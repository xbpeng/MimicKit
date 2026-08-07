# Copyright (c) 2026 Holiday Robotics
# SPDX-License-Identifier: MIT
# Adapted from https://github.com/Holiday-Robot/FlashSAC at 87edc906.
# See LICENSES/FlashSAC-MIT.txt.
"""Uniform replay buffer with FlashSAC's vectorized n-step returns."""

import collections
import os

import torch


class FlashSACReplayBuffer:
    """A device-resident replay buffer for batched MimicKit environments."""

    _FIELDS = (
        "observation",
        "next_observation",
        "action",
        "reward",
        "terminated",
        "truncated",
    )

    def __init__(
        self,
        observation_shape,
        action_shape,
        max_length,
        min_length,
        sample_batch_size,
        n_step,
        gamma,
        device,
        observation_dtype=torch.float32,
    ):
        max_length = int(max_length)
        min_length = int(min_length)
        sample_batch_size = int(sample_batch_size)
        n_step = int(n_step)
        if max_length <= 0:
            raise ValueError("buffer_max_length must be positive")
        if min_length <= 0:
            raise ValueError("buffer_min_length must be positive")
        if min_length > max_length:
            raise ValueError("buffer_min_length cannot exceed buffer_max_length")
        if n_step <= 0:
            raise ValueError("n_step must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("discount must be in [0, 1]")

        self._device = torch.device(device)
        self._max_length = max_length
        self._min_length = min_length
        self._sample_batch_size = sample_batch_size
        self._n_step = n_step
        self._gamma = float(gamma)
        self._observation_shape = tuple(observation_shape)
        self._action_shape = tuple(action_shape)
        self._observation_dtype = observation_dtype
        # Allocate on first use so loading a policy for evaluation does not also
        # allocate a multi-million-transition replay buffer.
        self._storage = None
        self._n_step_queue = collections.deque(maxlen=n_step)
        self._current_index = 0
        self._length = 0

    def __len__(self):
        return self._length

    @property
    def device(self):
        return self._device

    @property
    def pending_length(self):
        return len(self._n_step_queue)

    def can_sample(self):
        return self._length >= self._min_length

    def clear(self):
        self.clear_pending()
        self._current_index = 0
        self._length = 0

    def clear_pending(self):
        self._n_step_queue.clear()

    def add(self, transition):
        missing = set(self._FIELDS).difference(transition)
        if missing:
            raise KeyError("Replay transition is missing fields: {}".format(sorted(missing)))

        batch_size = transition["reward"].shape[0]
        for field in self._FIELDS:
            if transition[field].shape[0] != batch_size:
                raise ValueError("All replay fields must have the same leading batch dimension")

        self._allocate_storage()
        owned_transition = {
            field: transition[field]
            .detach()
            .to(device=self._device, dtype=self._storage[field].dtype, copy=True)
            for field in self._FIELDS
        }

        # Queue full vector-environment steps. This is equivalent to maintaining one
        # n-step queue per environment because environments occupy fixed batch slots.
        self._n_step_queue.append(owned_transition)
        if len(self._n_step_queue) < self._n_step:
            return

        n_step_transition = self._get_n_step_transition()
        self._write_batch(n_step_transition)

    def sample(self, sample_indices=None):
        if not self.can_sample():
            raise RuntimeError(
                "Replay buffer has {} transitions; {} are required".format(self._length, self._min_length)
            )
        if self._storage is None:
            raise RuntimeError("Cannot sample an unallocated replay buffer")
        if sample_indices is None:
            sample_indices = torch.randint(
                low=0,
                high=self._length,
                size=(self._sample_batch_size,),
                device=self._device,
            )
        else:
            sample_indices = torch.as_tensor(sample_indices, dtype=torch.long, device=self._device)
        return {field: values[sample_indices] for field, values in self._storage.items()}

    def state_dict(self):
        valid_storage = None
        if self._storage is not None:
            if self._length == self._max_length:
                # A full buffer already has compact underlying storages. Detached
                # views avoid a second multi-GB allocation on the replay device.
                valid_storage = {
                    field: values.detach() for field, values in self._storage.items()
                }
            else:
                # A partial slice still references its full underlying storage in
                # torch serialization. Make compact host copies so checkpoint size
                # scales with valid data without duplicating GPU memory.
                valid_storage = {
                    field: values[: self._length]
                    .detach()
                    .to(device="cpu", copy=True)
                    for field, values in self._storage.items()
                }
        return {
            "storage": valid_storage,
            "current_index": self._current_index,
            "length": self._length,
            "config": {
                "max_length": self._max_length,
                "min_length": self._min_length,
                "sample_batch_size": self._sample_batch_size,
                "n_step": self._n_step,
                "gamma": self._gamma,
            },
        }

    def load_state_dict(self, state):
        config = state.get("config", {})
        expected = {
            "max_length": self._max_length,
            "min_length": self._min_length,
            "sample_batch_size": self._sample_batch_size,
            "n_step": self._n_step,
            "gamma": self._gamma,
        }
        for key, value in expected.items():
            if key in config and config[key] != value:
                raise ValueError(
                    "Replay buffer {} mismatch: checkpoint has {}, configured {}".format(
                        key, config[key], value
                    )
                )

        length = int(state["length"])
        current_index = int(state["current_index"])
        if not 0 <= length <= self._max_length:
            raise ValueError("Replay checkpoint length is out of bounds")
        if not 0 <= current_index < self._max_length:
            raise ValueError("Replay checkpoint current_index is out of bounds")
        storage = state["storage"]
        if length > 0:
            if storage is None:
                raise ValueError("Replay checkpoint has data length but no storage")
            self._allocate_storage()
            for field in self._FIELDS:
                expected_shape = (length, *self._storage[field].shape[1:])
                if tuple(storage[field].shape) != expected_shape:
                    raise ValueError("Replay field '{}' has an incompatible shape".format(field))
                # copy_ performs the host-to-device transfer directly into the
                # destination slice; an intermediate .to(cuda) tensor would
                # transiently duplicate an entire replay field on the GPU.
                self._storage[field][:length].copy_(storage[field])
        self._current_index = current_index
        self._length = length
        # Match the public implementation: simulator state is not checkpointed,
        # so in-flight n-step transitions cannot be resumed consistently.
        self._n_step_queue.clear()

    def save(self, path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        # Stage compact checkpoint tensors in host memory. Mapping directly to
        # CUDA would coexist with the newly allocated full-capacity buffer and
        # can transiently require roughly twice the replay's GPU memory.
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def get_observations(self):
        if self._storage is None:
            return torch.empty(
                (0, *self._observation_shape),
                dtype=self._observation_dtype,
                device=self._device,
            )
        return self._storage["observation"][: self._length]

    def _allocate_storage(self):
        if self._storage is not None:
            return
        pin_memory = self._device.type == "cpu" and torch.cuda.is_available()
        common = {"device": self._device, "pin_memory": pin_memory}
        self._storage = {
            "observation": torch.empty(
                (self._max_length, *self._observation_shape),
                dtype=self._observation_dtype,
                **common,
            ),
            "next_observation": torch.empty(
                (self._max_length, *self._observation_shape),
                dtype=self._observation_dtype,
                **common,
            ),
            "action": torch.empty(
                (self._max_length, *self._action_shape), dtype=torch.float32, **common
            ),
            "reward": torch.empty(self._max_length, dtype=torch.float32, **common),
            "terminated": torch.empty(self._max_length, dtype=torch.float32, **common),
            "truncated": torch.empty(self._max_length, dtype=torch.float32, **common),
        }

    def _get_n_step_transition(self):
        current = self._n_step_queue[-1]
        reward = current["reward"].clone()
        next_observation = current["next_observation"].clone()
        terminated = current["terminated"].clone()
        truncated = current["truncated"].clone()

        for index in reversed(range(self._n_step - 1)):
            transition = self._n_step_queue[index]
            done = torch.logical_or(transition["terminated"].bool(), transition["truncated"].bool())
            reward = transition["reward"] + self._gamma * reward * (~done).to(reward.dtype)
            terminated[done] = transition["terminated"][done]
            truncated[done] = transition["truncated"][done]
            next_observation[done] = transition["next_observation"][done]

        first = self._n_step_queue[0]
        return {
            "observation": first["observation"],
            "next_observation": next_observation,
            "action": first["action"],
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }

    def _write_batch(self, transition):
        batch_size = transition["reward"].shape[0]
        end_index = self._current_index + batch_size
        if end_index <= self._max_length:
            indices = slice(self._current_index, end_index)
        else:
            indices = (
                torch.arange(batch_size, device=self._device) + self._current_index
            ) % self._max_length
        for field in self._FIELDS:
            self._storage[field][indices] = transition[field].to(
                device=self._device, dtype=self._storage[field].dtype
            )
        self._current_index = (self._current_index + batch_size) % self._max_length
        self._length = min(self._length + batch_size, self._max_length)

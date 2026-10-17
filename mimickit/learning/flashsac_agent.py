# Copyright (c) 2026 Holiday Robotics
# SPDX-License-Identifier: MIT
# Adapted from https://github.com/Holiday-Robot/FlashSAC at 87edc906.
# See LICENSES/FlashSAC-MIT.txt.
"""Single-device FlashSAC agent adapted to the MimicKit lifecycle.

The algorithmic core follows the official FlashSAC implementation at commit
87edc9061150ae9e962dd84e6544e27a1554b3ab. MimicKit-specific code handles
action scaling, terminal flags, vector-environment update accounting, logging,
and checkpointing. See ``LICENSES/FlashSAC-MIT.txt`` for license terms.
"""

import inspect
import math
import os
import time
import warnings

import gymnasium.spaces as spaces
import numpy as np
import torch

import envs.base_env as base_env
import learning.base_agent as base_agent
import learning.flashsac_model as flashsac_model
import learning.flashsac_replay_buffer as flashsac_replay_buffer
from util.logger import Logger
import util.mp_util as mp_util


class FlashSACAgent(base_agent.BaseAgent):
    """Reference-faithful FlashSAC for one MimicKit process/device."""

    _UPDATE_METRICS = (
        "actor_loss",
        "actor_entropy",
        "actor_mean_action",
        "temperature_value",
        "temperature_loss",
        "critic_loss",
        "critic_max_entropy_bonus",
    )

    def __init__(self, config, env, device):
        if mp_util.enable_mp():
            raise NotImplementedError(
                "FlashSAC currently supports one MimicKit process/device only; "
                "pass exactly one entry to --devices."
            )
        self._inside_train_model = False
        super().__init__(config=config, env=env, device=device)

    def _load_params(self, config):
        super()._load_params(config)

        self._buffer_max_length = int(config["buffer_max_length"])
        self._buffer_min_length = int(config["buffer_min_length"])
        self._buffer_device = config.get("buffer_device", "auto")
        self._sample_batch_size = int(config["sample_batch_size"])
        self._updates_per_env_step = float(config["updates_per_env_step"])
        self._n_step = int(config.get("n_step", 1))

        self._normalize_reward = bool(config.get("normalize_reward", True))
        self._normalized_G_max = float(config.get("normalized_G_max", 5.0))
        self._observation_max_abs = float(config.get("observation_max_abs", math.inf))

        self._learning_rate_init = float(config.get("learning_rate_init", 3e-4))
        self._learning_rate_peak = float(config.get("learning_rate_peak", 3e-4))
        self._learning_rate_end = float(config.get("learning_rate_end", self._learning_rate_peak))
        self._learning_rate_warmup_step = int(config.get("learning_rate_warmup_step", 0))
        self._learning_rate_decay_step = int(config.get("learning_rate_decay_step", 1_000_000))

        self._actor_num_blocks = int(config.get("actor_num_blocks", 2))
        self._actor_hidden_dim = int(config.get("actor_hidden_dim", 128))
        self._actor_bc_alpha = float(config.get("actor_bc_alpha", 0.0))
        self._actor_noise_zeta_mu = float(config.get("actor_noise_zeta_mu", 2.0))
        self._actor_noise_zeta_max = int(config.get("actor_noise_zeta_max", 16))
        self._actor_update_period = int(config.get("actor_update_period", 2))

        self._critic_num_blocks = int(config.get("critic_num_blocks", 2))
        self._critic_hidden_dim = int(config.get("critic_hidden_dim", 256))
        self._critic_num_bins = int(config.get("critic_num_bins", 101))
        self._critic_min_v = float(config.get("critic_min_v", -self._normalized_G_max))
        self._critic_max_v = float(config.get("critic_max_v", self._normalized_G_max))
        self._critic_target_update_tau = float(config.get("critic_target_update_tau", 0.01))

        self._temp_initial_value = float(config.get("temp_initial_value", 0.01))
        self._temp_target_sigma = float(config.get("temp_target_sigma", 0.15))

        self._use_compile = bool(config.get("use_compile", False))
        self._compile_mode = flashsac_model.resolve_compile_mode(
            config.get("compile_mode", "auto")
        )
        self._use_tf32 = bool(config.get("use_tf32", True))
        self._load_optimizer = bool(config.get("load_optimizer", True))
        self._load_reward_normalizer = bool(config.get("load_reward_normalizer", True))
        self._save_replay_buffer = bool(config.get("save_replay_buffer", False))
        self._load_replay_buffer = config.get("load_replay_buffer", "")

        self._torch_device = torch.device(self._device)
        self._update_accumulator = 0.0
        self._update_step = 0
        self._env_step_count = 0
        self._resume_iteration = None
        self._validate_config()
        self._configure_cuda_backend()

    def _configure_cuda_backend(self):
        if self._torch_device.type != "cuda" or not torch.cuda.is_available():
            return
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = self._use_tf32
        torch.backends.cudnn.allow_tf32 = self._use_tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if self._use_tf32 else "highest")

    def _validate_config(self):
        if self._steps_per_iter <= 0 or self._iters_per_output <= 0:
            raise ValueError("steps_per_iter and iters_per_output must be positive")
        if self._test_episodes < 0:
            raise ValueError("test_episodes must be non-negative")
        if self._buffer_max_length <= 0:
            raise ValueError("buffer_max_length must be positive")
        if not 0 < self._buffer_min_length <= self._buffer_max_length:
            raise ValueError("buffer_min_length must be in (0, buffer_max_length]")
        if self._sample_batch_size <= 0:
            raise ValueError("sample_batch_size must be positive")
        if self._updates_per_env_step < 0:
            raise ValueError("updates_per_env_step must be non-negative")
        if self._n_step <= 0:
            raise ValueError("n_step must be positive")
        if not 0.0 <= self._discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if self._normalized_G_max <= 0:
            raise ValueError("normalized_G_max must be positive")
        if self._observation_max_abs <= 0:
            raise ValueError("observation_max_abs must be positive")
        if self._learning_rate_peak <= 0:
            raise ValueError("learning_rate_peak must be positive")
        if self._learning_rate_init < 0 or self._learning_rate_end < 0:
            raise ValueError("learning-rate endpoints must be non-negative")
        if self._learning_rate_warmup_step < 0:
            raise ValueError("learning_rate_warmup_step must be non-negative")
        if self._learning_rate_decay_step <= self._learning_rate_warmup_step:
            raise ValueError("learning_rate_decay_step must exceed learning_rate_warmup_step")
        if self._actor_num_blocks < 0 or self._critic_num_blocks < 0:
            raise ValueError("actor_num_blocks and critic_num_blocks must be non-negative")
        if self._actor_hidden_dim <= 0 or self._critic_hidden_dim <= 0:
            raise ValueError("actor_hidden_dim and critic_hidden_dim must be positive")
        if self._actor_bc_alpha < 0:
            raise ValueError("actor_bc_alpha must be non-negative")
        if self._actor_noise_zeta_mu <= 0 or self._actor_noise_zeta_max <= 0:
            raise ValueError("actor noise zeta parameters must be positive")
        if self._actor_update_period <= 0:
            raise ValueError("actor_update_period must be positive")
        if self._critic_num_bins < 2:
            raise ValueError("critic_num_bins must be at least 2")
        if self._critic_min_v >= self._critic_max_v:
            raise ValueError("critic_min_v must be less than critic_max_v")
        if not 0.0 <= self._critic_target_update_tau <= 1.0:
            raise ValueError("critic_target_update_tau must be in [0, 1]")
        if self._temp_initial_value <= 0 or self._temp_target_sigma <= 0:
            raise ValueError("temperature parameters must be positive")
        if self._use_mixed_precision and self._torch_device.type != "cuda":
            raise ValueError("FlashSAC mixed precision currently requires a CUDA device")

    def _build_model(self, config):
        del config
        observation_space = self._env.get_obs_space()
        action_space = self._env.get_action_space()
        if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) != 1:
            raise ValueError("FlashSAC requires a one-dimensional Box observation space")
        if not isinstance(action_space, spaces.Box) or len(action_space.shape) != 1:
            raise ValueError("FlashSAC requires a one-dimensional Box action space")
        if not np.all(np.isfinite(action_space.low)) or not np.all(np.isfinite(action_space.high)):
            raise ValueError("FlashSAC requires finite action bounds")
        if not np.all(action_space.high > action_space.low):
            raise ValueError("Every FlashSAC action upper bound must exceed its lower bound")

        self._observation_dim = observation_space.shape[0]
        self._action_dim = action_space.shape[0]
        self._temp_target_entropy = 0.5 * self._action_dim * math.log(
            2.0 * math.pi * math.e * self._temp_target_sigma**2
        )

        self._actor = flashsac_model.FlashSACActor(
            num_blocks=self._actor_num_blocks,
            input_dim=self._observation_dim,
            hidden_dim=self._actor_hidden_dim,
            action_dim=self._action_dim,
        ).to(self._torch_device)
        self._critic = flashsac_model.FlashSACDoubleCritic(
            num_blocks=self._critic_num_blocks,
            input_dim=self._observation_dim + self._action_dim,
            hidden_dim=self._critic_hidden_dim,
            num_bins=self._critic_num_bins,
            min_v=self._critic_min_v,
            max_v=self._critic_max_v,
        ).to(self._torch_device)
        self._target_critic = flashsac_model.FlashSACDoubleCritic(
            num_blocks=self._critic_num_blocks,
            input_dim=self._observation_dim + self._action_dim,
            hidden_dim=self._critic_hidden_dim,
            num_bins=self._critic_num_bins,
            min_v=self._critic_min_v,
            max_v=self._critic_max_v,
        ).to(self._torch_device)
        self._target_critic.load_state_dict(self._critic.state_dict())
        self._temperature = flashsac_model.FlashSACTemperature(
            self._temp_initial_value
        ).to(self._torch_device)

        compile_mode = self._compile_mode if self._use_compile else None
        flashsac_model.prepare_module_parameter_normalization(self._actor, compile_mode)
        flashsac_model.prepare_module_parameter_normalization(self._critic, compile_mode)
        flashsac_model.prepare_module_parameter_normalization(
            self._target_critic, compile_mode
        )
        flashsac_model.normalize_module_parameters(self._actor)
        flashsac_model.normalize_module_parameters(self._critic)
        flashsac_model.normalize_module_parameters(self._target_critic)
        flashsac_model.prepare_ema_update(
            target_module=self._target_critic,
            source_module=self._critic,
            tau=self._critic_target_update_tau,
            compile_mode=compile_mode,
        )
        self._target_critic.requires_grad_(False)

        if self._use_compile:
            flashsac_model.compile_update_helpers()
            flashsac_model.compile_method(self._actor, "forward", self._compile_mode)
            flashsac_model.compile_method(
                self._actor, "get_mean_and_std", self._compile_mode
            )
            flashsac_model.compile_method(self._critic, "forward", self._compile_mode)
            flashsac_model.compile_method(
                self._target_critic, "forward", self._compile_mode
            )
            flashsac_model.compile_method(
                self._temperature, "forward", self._compile_mode
            )

    def _build_optimizer(self, config):
        del config
        schedule = flashsac_model.warmup_cosine_decay_scheduler(
            init_value=self._learning_rate_init,
            peak_value=self._learning_rate_peak,
            end_value=self._learning_rate_end,
            warmup_steps=self._learning_rate_warmup_step,
            decay_steps=self._learning_rate_decay_step,
        )
        adam_kwargs = {"lr": self._learning_rate_peak}
        if "fused" in inspect.signature(torch.optim.Adam).parameters:
            adam_kwargs["fused"] = self._torch_device.type == "cuda" and torch.cuda.is_available()

        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), **adam_kwargs)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), **adam_kwargs)
        self._temperature_optimizer = torch.optim.Adam(self._temperature.parameters(), **adam_kwargs)

        def lr_lambda(step):
            return schedule(step) / self._learning_rate_peak

        self._actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._actor_optimizer, lr_lambda=lr_lambda
        )
        self._critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._critic_optimizer, lr_lambda=lr_lambda
        )
        self._temperature_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._temperature_optimizer, lr_lambda=lr_lambda
        )
        self._grad_scaler = flashsac_model.build_grad_scaler(
            device=self._torch_device, enabled=self._use_mixed_precision
        )

    def _build_exp_buffer(self, config):
        del config
        if self._buffer_device in (None, "", "auto"):
            buffer_device = self._torch_device
        elif str(self._buffer_device) == "cuda" and self._torch_device.type == "cuda":
            buffer_device = self._torch_device
        else:
            buffer_device = torch.device(self._buffer_device)
        if self._buffer_max_length < self.get_num_envs():
            raise ValueError("buffer_max_length must be at least the number of environments")

        observation_space = self._env.get_obs_space()
        action_space = self._env.get_action_space()
        self._replay_buffer = flashsac_replay_buffer.FlashSACReplayBuffer(
            observation_shape=observation_space.shape,
            action_shape=action_space.shape,
            max_length=self._buffer_max_length,
            min_length=self._buffer_min_length,
            sample_batch_size=self._sample_batch_size,
            n_step=self._n_step,
            gamma=self._discount,
            device=buffer_device,
            observation_dtype=torch.float32,
        )
        self._reward_normalizer = None
        if self._normalize_reward:
            self._reward_normalizer = flashsac_model.RewardNormalizer(
                gamma=self._discount,
                G_max=self._normalized_G_max,
                device=self._torch_device,
            )

    def _get_exp_buffer_length(self):
        # BaseAgent requires this hook, but FlashSAC owns a replay buffer instead.
        return 1

    def _sync_optimizer(self):
        if mp_util.enable_mp():
            raise NotImplementedError("FlashSAC optimizer synchronization is not implemented")

    def _need_normalizer_update(self):
        # FlashSAC's input BatchNorm consumes raw observations, as in the reference.
        return False

    def _init_iter(self):
        self._iter_update_sums = {}
        self._iter_update_counts = {}
        self._iter_done_fail_count = torch.zeros((), device=self._torch_device)
        self._iter_done_succ_count = torch.zeros((), device=self._torch_device)
        self._iter_done_time_count = torch.zeros((), device=self._torch_device)
        self._iter_nonfinite_env_count = torch.zeros((), device=self._torch_device)
        self._iter_outlier_env_count = torch.zeros((), device=self._torch_device)
        self._iter_random_action_steps = 0

    def _init_train(self):
        if self._resume_iteration is None:
            self._iter = 0
        else:
            self._iter = self._resume_iteration
            self._resume_iteration = None
        self._sample_count = self._env_step_count
        self._train_start_env_step_count = self._env_step_count
        self._train_return_tracker.reset()
        self._test_return_tracker.reset()
        # Do not reset optimizer counters or the fractional update accumulator:
        # they are part of exact checkpoint resume state.

    def _update_sample_count(self):
        return int(self._env_step_count)

    def _build_zeta_cdf(self):
        # Match the reference implementation's CPU construction before moving
        # the small, immutable CDF to the training device.
        ns = torch.arange(1, self._actor_noise_zeta_max + 1, dtype=torch.float32)
        probabilities = ns.pow(-self._actor_noise_zeta_mu)
        return torch.cumsum(probabilities / probabilities.sum(), dim=0).to(self._torch_device)

    def _reset_noise_state(self):
        self._zeta_cdf = self._build_zeta_cdf()
        self._cached_noise = torch.randn(
            (self.get_num_envs(), self._action_dim), device=self._torch_device
        )
        # The public implementation deliberately uses one repeat interval shared
        # by all vector environments, while retaining a separate noise vector.
        self._cur_noise_repeat_n = torch.tensor(1, dtype=torch.int32, device=self._torch_device)
        self._cur_noise_repeat_count = torch.tensor(0, dtype=torch.int32, device=self._torch_device)

    def _sample_integer_from_cdf(self):
        sample = torch.rand((), device=self._torch_device)
        index = torch.argmax((sample < self._zeta_cdf).to(torch.int32))
        return (index + 1).to(torch.int32)

    def _sample_policy_unit_action(self, observation, training):
        observation = observation.to(device=self._torch_device, dtype=torch.float32)
        mean, std = self._actor.get_mean_and_std(observations=observation, training=False)
        if not training:
            return torch.tanh(mean)

        reinitialize = torch.logical_or(
            self._cur_noise_repeat_count == 0,
            self._cur_noise_repeat_count >= self._cur_noise_repeat_n,
        )
        new_noise = torch.randn_like(mean)
        new_repeat_n = self._sample_integer_from_cdf()
        self._cached_noise = torch.where(reinitialize, new_noise, self._cached_noise)
        self._cur_noise_repeat_n = torch.where(
            reinitialize, new_repeat_n, self._cur_noise_repeat_n
        )
        self._cur_noise_repeat_count = torch.where(
            reinitialize,
            torch.zeros_like(self._cur_noise_repeat_count),
            self._cur_noise_repeat_count,
        )
        action = torch.tanh(mean + std * self._cached_noise)
        self._cur_noise_repeat_count += 1
        return action

    def _sample_random_action(self, num_envs):
        unit_action = 2.0 * torch.rand(
            (num_envs, self._action_dim), dtype=torch.float32, device=self._torch_device
        ) - 1.0
        return unit_action, self._a_norm.unnormalize(unit_action)

    def _decide_action(self, obs, info):
        del info
        with torch.no_grad():
            unit_action = self._sample_policy_unit_action(
                obs, training=self._mode == base_agent.AgentMode.TRAIN
            )
            env_action = self._a_norm.unnormalize(unit_action)
        return env_action, {"unit_action": unit_action}

    def _decide_train_action(self, observation):
        if not self._replay_buffer.can_sample():
            self._iter_random_action_steps += 1
            return self._sample_random_action(observation.shape[0])
        env_action, action_info = self._decide_action(observation, None)
        return action_info["unit_action"], env_action

    def _convert_done_flags(self, done):
        fail = done == base_env.DoneFlags.FAIL.value
        success = done == base_env.DoneFlags.SUCC.value
        timeout = done == base_env.DoneFlags.TIME.value
        terminated = torch.logical_or(fail, success)
        truncated = timeout
        return terminated.float(), truncated.float(), fail, success, timeout

    def _step_env(self, action, fallback_observation=None):
        next_observation, reward, done, info = super()._step_env(action)

        flat_observation = next_observation.reshape(next_observation.shape[0], -1)
        finite_observation = torch.isfinite(flat_observation).all(dim=1)
        finite_reward = torch.isfinite(reward.reshape(reward.shape[0], -1)).all(dim=1)
        outlier_observation = torch.logical_and(
            finite_observation,
            flat_observation.abs().amax(dim=1) > self._observation_max_abs,
        )
        invalid_observation = torch.logical_or(
            torch.logical_not(finite_observation), outlier_observation
        )
        invalid_reward = torch.logical_not(finite_reward)
        invalid = torch.logical_or(invalid_observation, invalid_reward)

        # The reference update evaluates every next observation before applying
        # the terminal mask. Turn invalid simulator rows into explicit, finite
        # failure transitions so replay, BatchNorm, and categorical projection
        # cannot be poisoned, then let MimicKit reset those environments.
        if fallback_observation is None:
            fallback_observation = torch.zeros_like(next_observation)
        observation_mask = invalid_observation.reshape(
            (invalid.shape[0],) + (1,) * (next_observation.ndim - 1)
        )
        next_observation = torch.where(
            observation_mask,
            fallback_observation.to(next_observation),
            next_observation,
        )
        fail_reward = torch.full_like(reward, float(self._env.get_reward_fail()))
        reward_mask = torch.logical_not(finite_reward).reshape(
            (invalid.shape[0],) + (1,) * (reward.ndim - 1)
        )
        reward = torch.where(reward_mask, fail_reward, reward)
        done = torch.where(
            invalid,
            torch.full_like(done, base_env.DoneFlags.FAIL.value),
            done,
        )
        if self._mode == base_agent.AgentMode.TRAIN:
            nonfinite = torch.logical_or(
                torch.logical_not(finite_observation), invalid_reward
            )
            self._iter_nonfinite_env_count += nonfinite.sum()
            self._iter_outlier_env_count += outlier_observation.sum()
        return next_observation, reward, done, info

    def _record_transition(self, observation, unit_action, reward, done, next_observation):
        terminated, truncated, fail, success, timeout = self._convert_done_flags(done)
        transition = {
            "observation": observation,
            "next_observation": next_observation,
            "action": unit_action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
        self._replay_buffer.add(transition)
        if self._reward_normalizer is not None:
            self._reward_normalizer.update_reward_stats(
                reward=reward.to(self._torch_device, dtype=torch.float32),
                terminated=terminated.to(self._torch_device),
                truncated=truncated.to(self._torch_device),
            )

        self._iter_done_fail_count += fail.sum()
        self._iter_done_succ_count += success.sum()
        self._iter_done_time_count += timeout.sum()

    def _prepare_batch(self, batch):
        batch = {
            key: value.to(self._torch_device, non_blocking=True)
            for key, value in batch.items()
        }
        batch["actor_observation"] = batch["observation"]
        batch["actor_next_observation"] = batch["next_observation"]
        if self._reward_normalizer is not None:
            batch["reward"] = self._reward_normalizer.normalize_rewards(batch["reward"])
        return batch

    def _update_networks(self):
        batch = self._prepare_batch(self._replay_buffer.sample())
        update_info = {}
        if self._update_step % self._actor_update_period == 0:
            actor_info = flashsac_model.update_actor(
                actor=self._actor,
                critic=self._critic,
                temperature=self._temperature,
                optimizer=self._actor_optimizer,
                scheduler=self._actor_scheduler,
                batch=batch,
                bc_alpha=self._actor_bc_alpha,
                device=self._torch_device,
                use_amp=self._use_mixed_precision,
                grad_scaler=self._grad_scaler,
            )
            temperature_info = flashsac_model.update_temperature(
                temperature=self._temperature,
                optimizer=self._temperature_optimizer,
                scheduler=self._temperature_scheduler,
                entropy=actor_info["actor_entropy"],
                target_entropy=self._temp_target_entropy,
            )
            update_info.update(actor_info)
            update_info.update(temperature_info)

        critic_info = flashsac_model.update_critic(
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            optimizer=self._critic_optimizer,
            scheduler=self._critic_scheduler,
            batch=batch,
            min_v=self._critic_min_v,
            max_v=self._critic_max_v,
            num_bins=self._critic_num_bins,
            gamma=self._discount,
            n_step=self._n_step,
            device=self._torch_device,
            use_amp=self._use_mixed_precision,
            grad_scaler=self._grad_scaler,
        )
        flashsac_model.update_target_network(
            target_critic=self._target_critic,
            critic=self._critic,
            tau=self._critic_target_update_tau,
        )
        update_info.update(critic_info)
        self._update_step += 1
        return update_info

    def _accumulate_update_info(self, update_info):
        for key, value in update_info.items():
            value = value.detach() if torch.is_tensor(value) else value
            self._iter_update_sums[key] = self._iter_update_sums.get(key, 0.0) + value
            self._iter_update_counts[key] = self._iter_update_counts.get(key, 0) + 1

    def _finalize_train_iter_info(self):
        info = {key: 0.0 for key in self._UPDATE_METRICS}
        for key, value in self._iter_update_sums.items():
            info[key] = value / self._iter_update_counts[key]
        info.update(
            {
                "replay_size": float(len(self._replay_buffer)),
                "update_steps": float(self._update_step),
                "env_steps": float(self._env_step_count),
                "random_action_rate": self._iter_random_action_steps / self._steps_per_iter,
                "done_fail_count": self._iter_done_fail_count,
                "done_succ_count": self._iter_done_succ_count,
                "done_time_count": self._iter_done_time_count,
                "nonfinite_env_count": self._iter_nonfinite_env_count,
                "outlier_env_count": self._iter_outlier_env_count,
            }
        )
        return info

    def _train_iter(self):
        self._init_iter()
        self.train()
        self.set_mode(base_agent.AgentMode.TRAIN)

        for _ in range(self._steps_per_iter):
            # MimicKit simulation environments may reuse and mutate their observation
            # buffer during step(), so the transition must own the pre-step value.
            observation = self._curr_obs.detach().to(dtype=torch.float32).clone()
            with torch.no_grad():
                unit_action, env_action = self._decide_train_action(observation)
                next_observation, reward, done, _ = self._step_env(
                    env_action, fallback_observation=observation
                )
                self._train_return_tracker.update(reward, done)
                self._record_transition(
                    observation=observation,
                    unit_action=unit_action,
                    reward=reward,
                    done=done,
                    next_observation=next_observation,
                )
                self._curr_obs, self._curr_info = self._reset_done_envs(done)
                self._env_step_count += self.get_num_envs()

            if self._replay_buffer.can_sample():
                self._update_accumulator += self._updates_per_env_step * self.get_num_envs()
                while self._update_accumulator >= 1.0:
                    update_info = self._update_networks()
                    self._accumulate_update_info(update_info)
                    self._update_accumulator -= 1.0

        info = self._finalize_train_iter_info()
        info["mean_return"] = self._train_return_tracker.get_mean_return().item()
        info["mean_ep_len"] = self._train_return_tracker.get_mean_ep_len().item()
        info["num_eps"] = self._train_return_tracker.get_episodes()
        return info

    def _clear_rollout_state(self):
        self._replay_buffer.clear_pending()
        if self._reward_normalizer is not None:
            self._reward_normalizer.reset_returns()
        self._reset_noise_state()

    def test_model(self, num_episodes):
        restore_train_mode = self._inside_train_model
        self._clear_rollout_state()
        try:
            return super().test_model(num_episodes)
        finally:
            self._clear_rollout_state()
            if restore_train_mode:
                self.train()
                self.set_mode(base_agent.AgentMode.TRAIN)

    def train_model(self, max_samples, out_dir, save_int_models, logger_type):
        if self._load_replay_buffer:
            self._replay_buffer.load(self._load_replay_buffer)
            Logger.print("[FlashSAC] Loaded replay buffer from {}".format(self._load_replay_buffer))
        # Loading replay never restores in-flight n-step transitions because
        # MimicKit does not checkpoint the matching simulator state.
        self._clear_rollout_state()
        self._inside_train_model = True
        try:
            return super().train_model(max_samples, out_dir, save_int_models, logger_type)
        finally:
            self._inside_train_model = False

    def _log_train_info(self, train_info, test_info, env_diag_info, start_time):
        super()._log_train_info(train_info, test_info, env_diag_info, start_time)
        # Samples are cumulative across checkpoint resume, while wall time starts
        # with this process. Replace only the derived throughput metric.
        elapsed = max(time.time() - start_time, 1e-8)
        current_run_samples = self._sample_count - self._train_start_env_step_count
        self._logger.log(
            "Samples_Per_Second",
            current_run_samples / elapsed,
            collection="1_Info",
            quiet=True,
        )

    def save(self, out_file):
        self._save_checkpoint(out_file, include_replay=self._save_replay_buffer)

    def _save_checkpoint(self, out_file, include_replay):
        if not mp_util.is_root_proc():
            return
        state = {
            "flashsac_format_version": 2,
            "observation_preprocessing": "raw",
            "normalize_reward": self._normalize_reward,
            "module_state_dict": self.state_dict(),
            "actor_optimizer": self._actor_optimizer.state_dict(),
            "critic_optimizer": self._critic_optimizer.state_dict(),
            "temperature_optimizer": self._temperature_optimizer.state_dict(),
            "actor_scheduler": self._actor_scheduler.state_dict(),
            "critic_scheduler": self._critic_scheduler.state_dict(),
            "temperature_scheduler": self._temperature_scheduler.state_dict(),
            "grad_scaler": self._grad_scaler.state_dict(),
            "update_step": self._update_step,
            "env_step_count": self._env_step_count,
            "update_accumulator": self._update_accumulator,
            "iteration": self._iter,
            "reward_normalizer": (
                self._reward_normalizer.state_dict()
                if self._reward_normalizer is not None
                else None
            ),
        }
        torch.save(state, out_file)
        if include_replay:
            self._replay_buffer.save(out_file + ".replay.pt")

    def _output_train_model(self, iter, out_model_file, int_out_dir):
        # Keep one replay sidecar next to the rolling main checkpoint. Writing the
        # same multi-GB replay again for every intermediate weight snapshot would
        # multiply I/O and storage without adding replay history.
        self._save_checkpoint(out_model_file, include_replay=self._save_replay_buffer)
        if int_out_dir:
            int_model_file = os.path.join(int_out_dir, "model_{:010d}.pt".format(iter))
            self._save_checkpoint(int_model_file, include_replay=False)

    def load(self, in_file):
        state = torch.load(in_file, map_location=self._torch_device)
        self._resume_iteration = None
        if "module_state_dict" not in state:
            warning = (
                "Plain FlashSAC state_dict checkpoint has no normalize_reward contract; "
                "loading weights for backward compatibility."
            )
            if self._load_optimizer:
                warning += (
                    " load_optimizer=True cannot be honored because optimizer and counter "
                    "state is absent; this is a weights-only load."
                )
            warnings.warn(warning, RuntimeWarning, stacklevel=2)
            self.load_state_dict(state)
        else:
            format_version = state.get("flashsac_format_version")
            if format_version is None:
                raise ValueError(
                    "Legacy FlashSAC checkpoints may use external observation normalization "
                    "and cannot be loaded safely by the raw-observation MimicKit agent."
                )
            if format_version not in (1, 2):
                raise ValueError("Unsupported FlashSAC checkpoint format version")
            if state.get("observation_preprocessing") != "raw":
                raise ValueError(
                    "This FlashSAC checkpoint does not declare raw-observation preprocessing. "
                    "Legacy CATO checkpoints trained with external observation normalization "
                    "are not compatible with the reference-faithful MimicKit agent."
                )
            if "normalize_reward" not in state:
                warnings.warn(
                    "FlashSAC checkpoint does not declare the normalize_reward contract; "
                    "loading it for backward compatibility.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                checkpoint_normalize_reward = state["normalize_reward"]
                if not isinstance(checkpoint_normalize_reward, bool):
                    raise ValueError(
                        "FlashSAC checkpoint normalize_reward contract must be a boolean"
                    )
                if checkpoint_normalize_reward != self._normalize_reward:
                    raise ValueError(
                        "FlashSAC checkpoint normalize_reward mismatch: checkpoint={}, "
                        "configured={}".format(
                            checkpoint_normalize_reward, self._normalize_reward
                        )
                    )
            self.load_state_dict(state["module_state_dict"])
            if self._load_optimizer:
                self._actor_optimizer.load_state_dict(state["actor_optimizer"])
                self._critic_optimizer.load_state_dict(state["critic_optimizer"])
                self._temperature_optimizer.load_state_dict(state["temperature_optimizer"])
                self._actor_scheduler.load_state_dict(state["actor_scheduler"])
                self._critic_scheduler.load_state_dict(state["critic_scheduler"])
                self._temperature_scheduler.load_state_dict(state["temperature_scheduler"])
                self._grad_scaler.load_state_dict(state["grad_scaler"])
                self._update_step = int(state["update_step"])
                self._env_step_count = int(state["env_step_count"])
                self._update_accumulator = float(state["update_accumulator"])
                checkpoint_iteration = state.get("iteration")
                if checkpoint_iteration is not None:
                    checkpoint_iteration = int(checkpoint_iteration)
                    if checkpoint_iteration < 0:
                        raise ValueError(
                            "FlashSAC checkpoint iteration must be non-negative"
                        )
                    self._iter = checkpoint_iteration
                    self._resume_iteration = checkpoint_iteration + 1
            normalizer_state = state.get("reward_normalizer")
            if (
                self._load_reward_normalizer
                and self._reward_normalizer is not None
                and normalizer_state is not None
            ):
                self._reward_normalizer.load_state_dict(normalizer_state)
        self._sync_optimizer()
        Logger.print("Loaded FlashSAC model parameters from {}".format(in_file))

    def _update_model(self):
        # BaseAgent requires this abstract hook; updates happen online in _train_iter.
        return {}

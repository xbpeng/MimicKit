import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import gymnasium.spaces as spaces
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimickit"))

import envs.base_env as base_env  # noqa: E402
import learning.agent_builder as agent_builder  # noqa: E402
import learning.base_agent as base_agent  # noqa: E402
import learning.flashsac_agent as flashsac_agent  # noqa: E402
import util.mp_util as mp_util  # noqa: E402


class MutableVectorEnv(base_env.BaseEnv):
    """Small Box environment that deliberately reuses its observation tensor."""

    def __init__(self, num_envs=4, horizon=3):
        super().__init__(visualize=False)
        self._num_envs = num_envs
        self._horizon = horizon
        self._action_space = spaces.Box(
            low=np.array([-2.0, 0.5], dtype=np.float32),
            high=np.array([3.0, 2.5], dtype=np.float32),
            dtype=np.float32,
        )
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self._observation = torch.zeros(num_envs, 3)
        self._episode_step = torch.zeros(num_envs, dtype=torch.long)

    def get_num_envs(self):
        return self._num_envs

    def get_obs_space(self):
        return self._observation_space

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs)
        self._observation[env_ids] = 0.0
        self._episode_step[env_ids] = 0
        return self._observation, {}

    def step(self, action):
        self._episode_step += 1
        # In-place mutation is intentional: replay must have cloned the old value.
        self._observation[:, 0].add_(1.0)
        self._observation[:, 1] = action[:, 0]
        self._observation[:, 2] = action[:, 1]
        reward = 1.0 - 0.01 * action.square().sum(dim=-1)
        done = torch.zeros(self._num_envs, dtype=torch.long)
        done[self._episode_step >= self._horizon] = base_env.DoneFlags.TIME.value
        return self._observation, reward, done, {}


class FakeGradScaler:
    """CPU-only scaler double for checkpoint serialization tests."""

    def __init__(self, state):
        self._state = copy.deepcopy(state)

    def state_dict(self):
        return copy.deepcopy(self._state)

    def load_state_dict(self, state):
        self._state = copy.deepcopy(state)


def _config(n_step=1, **overrides):
    config = {
        "agent_name": "FlashSAC",
        "discount": 0.99,
        "steps_per_iter": 2,
        "iters_per_output": 1,
        "test_episodes": 1,
        "use_mixed_precision": False,
        "buffer_max_length": 64,
        "buffer_min_length": 8,
        "buffer_device": "cpu",
        "sample_batch_size": 8,
        "updates_per_env_step": 0.25,
        "n_step": n_step,
        "normalize_reward": True,
        "normalized_G_max": 5.0,
        "observation_max_abs": 1000.0,
        "learning_rate_init": 3e-4,
        "learning_rate_peak": 3e-4,
        "learning_rate_end": 1.5e-4,
        "learning_rate_warmup_step": 0,
        "learning_rate_decay_step": 100,
        "actor_num_blocks": 1,
        "actor_hidden_dim": 16,
        "actor_bc_alpha": 0.0,
        "actor_noise_zeta_mu": 2.0,
        "actor_noise_zeta_max": 4,
        "actor_update_period": 2,
        "critic_num_blocks": 1,
        "critic_hidden_dim": 16,
        "critic_num_bins": 11,
        "critic_min_v": -5.0,
        "critic_max_v": 5.0,
        "critic_target_update_tau": 0.01,
        "temp_initial_value": 0.01,
        "temp_target_sigma": 0.15,
        "use_compile": False,
        "load_optimizer": True,
        "load_reward_normalizer": True,
        "save_replay_buffer": False,
        "load_replay_buffer": "",
    }
    config.update(overrides)
    return config


class FlashSACAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_num_procs = mp_util.global_num_procs
        cls._old_device = mp_util.global_mp_device
        mp_util.global_num_procs = 1
        mp_util.global_mp_device = "cpu"

    @classmethod
    def tearDownClass(cls):
        mp_util.global_num_procs = cls._old_num_procs
        mp_util.global_mp_device = cls._old_device

    def _assert_nested_equal(self, actual, expected):
        if torch.is_tensor(expected):
            self.assertTrue(torch.is_tensor(actual))
            torch.testing.assert_close(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self._assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for actual_value, expected_value in zip(actual, expected):
                self._assert_nested_equal(actual_value, expected_value)
        else:
            self.assertEqual(actual, expected)

    def _build_agent(self, n_step=1):
        env = MutableVectorEnv()
        agent = flashsac_agent.FlashSACAgent(config=_config(n_step=n_step), env=env, device="cpu")
        agent._reset_noise_state()
        agent._curr_obs, agent._curr_info = env.reset()
        agent._init_train()
        return agent, env

    def test_builder_and_lazy_replay(self):
        env = MutableVectorEnv()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as stream:
            yaml.safe_dump(_config(), stream)
            path = stream.name
        try:
            agent = agent_builder.build_agent(path, env=env, device="cpu")
        finally:
            os.unlink(path)
        self.assertIsInstance(agent, flashsac_agent.FlashSACAgent)
        self.assertIsNone(agent._replay_buffer._storage)
        unit_action = torch.tensor([[-1.0, 0.5], [0.25, 1.0]])
        env_action = agent._a_norm.unnormalize(unit_action)
        reconstructed = agent._a_norm.normalize(env_action)
        torch.testing.assert_close(reconstructed, unit_action)

    def test_warmup_first_update_and_finite_losses(self):
        torch.manual_seed(7)
        agent, _ = self._build_agent()
        first = agent._train_iter()
        self.assertEqual(len(agent._replay_buffer), 8)
        self.assertEqual(agent._update_step, 1)
        self.assertEqual(first["random_action_rate"], 1.0)
        for key in ("actor_loss", "critic_loss", "temperature_loss"):
            self.assertTrue(torch.isfinite(torch.as_tensor(first[key])).item(), key)

        second = agent._train_iter()
        self.assertEqual(agent._update_step, 3)
        self.assertTrue(torch.isfinite(torch.as_tensor(second["critic_loss"])).item())
        self.assertEqual(agent._iter_update_counts["critic_loss"], 2)
        self.assertEqual(agent._iter_update_counts["actor_loss"], 1)
        for parameter in agent._actor.parameters():
            self.assertTrue(torch.isfinite(parameter).all())
        for parameter in agent._critic.parameters():
            self.assertTrue(torch.isfinite(parameter).all())

    def test_terminal_observation_is_owned_before_reset(self):
        agent, _ = self._build_agent()
        agent._train_iter()
        # The first vector step stored next_observation[:, 0] == 1. A later env
        # step/reset mutated the shared env tensor, but replay must retain 1.
        sample = agent._replay_buffer.sample(sample_indices=[0])
        torch.testing.assert_close(sample["next_observation"][0, 0], torch.tensor(1.0))

    def test_nonfinite_env_output_becomes_finite_failure_transition(self):
        agent, env = self._build_agent()
        agent._init_iter()
        original_step = env.step

        def step_with_invalid_rows(action):
            observation, reward, done, info = original_step(action)
            observation = observation.clone()
            reward = reward.clone()
            observation[1, 0] = torch.nan
            reward[2] = torch.inf
            observation[3, 0] = 1001.0
            return observation, reward, done, info

        env.step = step_with_invalid_rows
        next_observation, reward, done, _ = agent._step_env(torch.zeros(4, 2))

        self.assertTrue(torch.isfinite(next_observation).all())
        self.assertTrue(torch.isfinite(reward).all())
        torch.testing.assert_close(next_observation[1], torch.zeros(3))
        self.assertEqual(reward[1].item(), 1.0)
        self.assertEqual(reward[2].item(), env.get_reward_fail())
        self.assertEqual(done[1].item(), base_env.DoneFlags.FAIL.value)
        self.assertEqual(done[2].item(), base_env.DoneFlags.FAIL.value)
        self.assertEqual(done[3].item(), base_env.DoneFlags.FAIL.value)
        self.assertEqual(agent._iter_nonfinite_env_count.item(), 2)
        self.assertEqual(agent._iter_outlier_env_count.item(), 1)

        agent._record_transition(
            observation=torch.ones(4, 3),
            unit_action=torch.ones(4, 2),
            reward=reward,
            done=done,
            next_observation=next_observation,
        )
        replay = agent._replay_buffer._storage
        torch.testing.assert_close(replay["next_observation"][1], torch.zeros(3))
        torch.testing.assert_close(
            replay["next_observation"][2], torch.tensor([1.0, 0.0, 0.0])
        )
        torch.testing.assert_close(replay["next_observation"][3], torch.zeros(3))
        for row in (1, 2, 3):
            torch.testing.assert_close(replay["observation"][row], torch.ones(3))
            torch.testing.assert_close(replay["action"][row], torch.ones(2))
            self.assertEqual(replay["terminated"][row].item(), 1.0)

    def test_deterministic_test_action_does_not_advance_noise(self):
        agent, env = self._build_agent()
        observation, _ = env.reset()
        agent.set_mode(base_agent.AgentMode.TEST)
        count = agent._cur_noise_repeat_count.clone()
        first, _ = agent._decide_action(observation, None)
        second, _ = agent._decide_action(observation, None)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(agent._cur_noise_repeat_count, count)

    def test_noise_repeat_interval_is_scalar_and_shared(self):
        agent, _ = self._build_agent()
        mean = torch.zeros(4, 2)
        std = torch.ones(4, 2)
        agent._cur_noise_repeat_n.fill_(2)
        agent._cur_noise_repeat_count.zero_()
        repeat = torch.tensor(2, dtype=torch.int32)
        with mock.patch.object(agent._actor, "get_mean_and_std", return_value=(mean, std)):
            with mock.patch.object(agent, "_sample_integer_from_cdf", return_value=repeat):
                first = agent._sample_policy_unit_action(torch.zeros(4, 3), training=True)
                first_noise = agent._cached_noise.clone()
                second = agent._sample_policy_unit_action(torch.zeros(4, 3), training=True)
                second_noise = agent._cached_noise.clone()
                third = agent._sample_policy_unit_action(torch.zeros(4, 3), training=True)
        torch.testing.assert_close(second_noise, first_noise)
        torch.testing.assert_close(second, first)
        self.assertEqual(agent._cur_noise_repeat_n.ndim, 0)
        self.assertFalse(torch.equal(agent._cached_noise, second_noise))
        self.assertFalse(torch.equal(third, second))

    def test_evaluation_boundary_preserves_replay_and_stats(self):
        agent, env = self._build_agent(n_step=3)
        agent._train_iter()
        replay_length = len(agent._replay_buffer)
        rms_count = agent._reward_normalizer.G_rms.count.clone()
        reward_max = agent._reward_normalizer.G_r_max.clone()
        self.assertGreater(agent._replay_buffer.pending_length, 0)
        self.assertGreater(torch.count_nonzero(agent._reward_normalizer.G_r).item(), 0)

        agent._inside_train_model = True
        agent.test_model(1)
        agent._inside_train_model = False

        self.assertEqual(len(agent._replay_buffer), replay_length)
        self.assertEqual(agent._replay_buffer.pending_length, 0)
        torch.testing.assert_close(agent._reward_normalizer.G_r, torch.zeros_like(agent._reward_normalizer.G_r))
        torch.testing.assert_close(agent._reward_normalizer.G_rms.count, rms_count)
        torch.testing.assert_close(agent._reward_normalizer.G_r_max, reward_max)
        self.assertEqual(agent._mode, base_agent.AgentMode.TRAIN)
        self.assertEqual(env._mode, base_env.EnvMode.TRAIN)

    def test_direct_evaluation_remains_in_test_mode(self):
        agent, env = self._build_agent()
        agent.test_model(1)
        self.assertEqual(agent._mode, base_agent.AgentMode.TEST)
        self.assertEqual(env._mode, base_env.EnvMode.TEST)

    def test_checkpoint_round_trip_restores_complete_training_state(self):
        torch.manual_seed(11)
        agent, _ = self._build_agent()
        agent._train_iter()
        agent._update_accumulator = 0.375
        agent._iter = 7
        agent._grad_scaler = FakeGradScaler(
            {"scale": torch.tensor(32.0), "growth_tracker": 17}
        )

        optimizers = (
            agent._actor_optimizer,
            agent._critic_optimizer,
            agent._temperature_optimizer,
        )
        self.assertTrue(all(optimizer.state_dict()["state"] for optimizer in optimizers))

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            agent.save(path)
            checkpoint = torch.load(path, map_location="cpu")
            self.assertEqual(checkpoint["flashsac_format_version"], 2)
            self.assertIs(checkpoint["normalize_reward"], True)
            self.assertEqual(checkpoint["iteration"], 7)

            restored_env = MutableVectorEnv()
            restored = flashsac_agent.FlashSACAgent(
                config=_config(), env=restored_env, device="cpu"
            )
            restored._grad_scaler = FakeGradScaler(
                {"scale": torch.tensor(1.0), "growth_tracker": 0}
            )
            restored.load(path)

        self.assertEqual(restored._update_step, agent._update_step)
        self.assertEqual(restored._env_step_count, agent._env_step_count)
        self.assertEqual(restored._update_accumulator, 0.375)
        self.assertEqual(restored._iter, 7)

        for actual, expected in zip(
            (
                restored._actor_optimizer.state_dict(),
                restored._critic_optimizer.state_dict(),
                restored._temperature_optimizer.state_dict(),
            ),
            (
                agent._actor_optimizer.state_dict(),
                agent._critic_optimizer.state_dict(),
                agent._temperature_optimizer.state_dict(),
            ),
        ):
            self._assert_nested_equal(actual, expected)

        for actual, expected in zip(
            (
                restored._actor_scheduler,
                restored._critic_scheduler,
                restored._temperature_scheduler,
            ),
            (
                agent._actor_scheduler,
                agent._critic_scheduler,
                agent._temperature_scheduler,
            ),
        ):
            self._assert_nested_equal(actual.state_dict(), expected.state_dict())
            self.assertEqual(actual.get_last_lr(), expected.get_last_lr())

        self._assert_nested_equal(
            restored._grad_scaler.state_dict(), agent._grad_scaler.state_dict()
        )
        self._assert_nested_equal(
            restored._reward_normalizer.state_dict(),
            agent._reward_normalizer.state_dict(),
        )

        restored._init_train()
        self.assertEqual(restored._update_accumulator, 0.375)
        self.assertEqual(restored._iter, 8)

        observation = torch.randn(4, 3)
        agent._reset_noise_state()
        restored._reset_noise_state()
        agent.set_mode(base_agent.AgentMode.TEST)
        restored.set_mode(base_agent.AgentMode.TEST)
        expected, _ = agent._decide_action(observation, None)
        actual, _ = restored._decide_action(observation, None)
        torch.testing.assert_close(actual, expected)

    def test_load_optimizer_false_is_weights_only_for_training_state(self):
        torch.manual_seed(13)
        agent, _ = self._build_agent()
        agent._train_iter()
        agent._update_accumulator = 0.625
        agent._iter = 11
        agent._grad_scaler = FakeGradScaler(
            {"scale": torch.tensor(64.0), "growth_tracker": 23}
        )

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            agent.save(path)
            restored = flashsac_agent.FlashSACAgent(
                config=_config(load_optimizer=False),
                env=MutableVectorEnv(),
                device="cpu",
            )
            restored._grad_scaler = FakeGradScaler(
                {"scale": torch.tensor(2.0), "growth_tracker": 3}
            )
            initial_optimizer_states = (
                copy.deepcopy(restored._actor_optimizer.state_dict()),
                copy.deepcopy(restored._critic_optimizer.state_dict()),
                copy.deepcopy(restored._temperature_optimizer.state_dict()),
            )
            initial_scheduler_states = (
                copy.deepcopy(restored._actor_scheduler.state_dict()),
                copy.deepcopy(restored._critic_scheduler.state_dict()),
                copy.deepcopy(restored._temperature_scheduler.state_dict()),
            )
            initial_scaler_state = restored._grad_scaler.state_dict()
            restored.load(path)

        self._assert_nested_equal(restored.state_dict(), agent.state_dict())
        for actual, expected in zip(
            (
                restored._actor_optimizer.state_dict(),
                restored._critic_optimizer.state_dict(),
                restored._temperature_optimizer.state_dict(),
            ),
            initial_optimizer_states,
        ):
            self._assert_nested_equal(actual, expected)
        for actual, expected in zip(
            (
                restored._actor_scheduler.state_dict(),
                restored._critic_scheduler.state_dict(),
                restored._temperature_scheduler.state_dict(),
            ),
            initial_scheduler_states,
        ):
            self._assert_nested_equal(actual, expected)
        self._assert_nested_equal(restored._grad_scaler.state_dict(), initial_scaler_state)
        self._assert_nested_equal(
            restored._reward_normalizer.state_dict(),
            agent._reward_normalizer.state_dict(),
        )
        self.assertEqual(restored._update_step, 0)
        self.assertEqual(restored._env_step_count, 0)
        self.assertEqual(restored._update_accumulator, 0.0)
        self.assertEqual(restored._iter, 0)
        restored._init_train()
        self.assertEqual(restored._iter, 0)

    def test_normalize_reward_checkpoint_mismatch_is_rejected(self):
        agent, _ = self._build_agent()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            agent.save(path)
            restored = flashsac_agent.FlashSACAgent(
                config=_config(normalize_reward=False),
                env=MutableVectorEnv(),
                device="cpu",
            )
            with self.assertRaisesRegex(ValueError, "normalize_reward mismatch"):
                restored.load(path)

    def test_version_one_checkpoint_warns_and_missing_iteration_falls_back(self):
        agent, _ = self._build_agent()
        agent._train_iter()
        agent._iter = 9
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            agent.save(path)
            checkpoint = torch.load(path, map_location="cpu")
            checkpoint["flashsac_format_version"] = 1
            checkpoint.pop("normalize_reward")
            checkpoint.pop("iteration")
            torch.save(checkpoint, path)

            restored = flashsac_agent.FlashSACAgent(
                config=_config(), env=MutableVectorEnv(), device="cpu"
            )
            with self.assertWarnsRegex(RuntimeWarning, "normalize_reward contract"):
                restored.load(path)

        self.assertEqual(restored._update_step, agent._update_step)
        self.assertEqual(restored._iter, 0)
        restored._init_train()
        self.assertEqual(restored._iter, 0)

    def test_plain_state_dict_warns_when_optimizer_restore_was_requested(self):
        agent, _ = self._build_agent()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "weights.pt")
            torch.save(agent.state_dict(), path)
            restored = flashsac_agent.FlashSACAgent(
                config=_config(), env=MutableVectorEnv(), device="cpu"
            )
            with self.assertWarnsRegex(
                RuntimeWarning, "load_optimizer=True.*weights-only"
            ):
                restored.load(path)

        self._assert_nested_equal(restored.state_dict(), agent.state_dict())
        self.assertEqual(restored._update_step, 0)
        self.assertEqual(restored._env_step_count, 0)

    def test_resumed_throughput_uses_only_current_run_samples(self):
        class CaptureLogger:
            def __init__(self):
                self.values = {}

            def log(self, key, value, **kwargs):
                del kwargs
                self.values[key] = value

        agent, _ = self._build_agent()
        agent._env_step_count = 80
        agent._init_train()
        agent._sample_count = 100
        agent._iter = 0
        agent._logger = CaptureLogger()
        train_info = {"mean_return": 0.0, "mean_ep_len": 0.0, "num_eps": 0}
        test_info = {"mean_return": 0.0, "mean_ep_len": 0.0, "num_eps": 0}
        with mock.patch("learning.flashsac_agent.time.time", return_value=110.0):
            agent._log_train_info(train_info, test_info, {}, start_time=100.0)
        self.assertEqual(agent._logger.values["Samples"], 100)
        self.assertEqual(agent._logger.values["Samples_Per_Second"], 2.0)

    def test_multi_process_fails_before_construction(self):
        old_num_procs = mp_util.global_num_procs
        mp_util.global_num_procs = 2
        try:
            with self.assertRaisesRegex(NotImplementedError, "one MimicKit process"):
                flashsac_agent.FlashSACAgent(
                    config=_config(), env=MutableVectorEnv(), device="cpu"
                )
        finally:
            mp_util.global_num_procs = old_num_procs

    def test_legacy_normalized_checkpoint_fails_clearly(self):
        agent, _ = self._build_agent()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "legacy.pt")
            torch.save({"module_state_dict": agent.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "external observation normalization"):
                agent.load(path)


if __name__ == "__main__":
    unittest.main()

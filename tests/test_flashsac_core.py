import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimickit"))

import learning.flashsac_model as flashsac_model  # noqa: E402
import learning.flashsac_replay_buffer as flashsac_replay_buffer  # noqa: E402


def _transition(observation, next_observation, reward, terminated=0.0, truncated=0.0):
    observation = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
    next_observation = torch.as_tensor(next_observation, dtype=torch.float32).reshape(1, -1)
    return {
        "observation": observation,
        "next_observation": next_observation,
        "action": torch.zeros(1, 1),
        "reward": torch.tensor([reward], dtype=torch.float32),
        "terminated": torch.tensor([terminated], dtype=torch.float32),
        "truncated": torch.tensor([truncated], dtype=torch.float32),
    }


class FlashSACCoreTest(unittest.TestCase):
    def test_safe_tanh_log_jacobian(self):
        value = torch.linspace(-8.0, 8.0, 101)
        actual = flashsac_model.safe_tanh_log_det_jacobian(value)
        expected = torch.log1p(-torch.tanh(value.double()).square()).float()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        extreme = flashsac_model.safe_tanh_log_det_jacobian(
            torch.tensor([-1000.0, 1000.0])
        )
        self.assertTrue(torch.isfinite(extreme).all())

    def test_categorical_projection_preserves_probability_mass(self):
        logits = torch.tensor([[0.0, 1.0, -1.0, 0.25, 0.5]])
        log_probs = torch.log_softmax(logits, dim=-1)
        projected = flashsac_model._compute_categorical_td_target(
            target_log_probs=log_probs,
            reward=torch.tensor([0.4]),
            done=torch.tensor([0.0]),
            actor_entropy=torch.tensor([-0.2]),
            gamma=0.99,
            num_bins=5,
            min_v=-1.0,
            max_v=1.0,
        )
        torch.testing.assert_close(projected.sum(dim=-1), torch.ones(1), atol=1e-6, rtol=1e-6)
        self.assertTrue((projected >= 0).all())

        terminal = flashsac_model._compute_categorical_td_target(
            target_log_probs=log_probs,
            reward=torch.tensor([0.5]),
            done=torch.tensor([1.0]),
            actor_entropy=torch.tensor([100.0]),
            gamma=0.99,
            num_bins=5,
            min_v=-1.0,
            max_v=1.0,
        )
        expected = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
        torch.testing.assert_close(terminal, expected)

    def test_parameter_projection_and_target_ema(self):
        source = flashsac_model.FlashSACActor(1, 3, 8, 2)
        target = flashsac_model.FlashSACActor(1, 3, 8, 2)
        flashsac_model.normalize_module_parameters(source)
        for module in source.modules():
            if hasattr(module, "w") and isinstance(module.w, torch.nn.Linear):
                norms = module.w.weight.norm(dim=-1)
                torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-6, rtol=1e-6)

        target_before = [parameter.detach().clone() for parameter in target.parameters()]
        source_values = [parameter.detach().clone() for parameter in source.parameters()]
        flashsac_model.ema_update_module(target, source, tau=0.25)
        for before, source_value, actual in zip(target_before, source_values, target.parameters()):
            torch.testing.assert_close(actual, before.lerp(source_value, 0.25))

    def test_reward_normalizer_reset_preserves_statistics(self):
        normalizer = flashsac_model.RewardNormalizer(
            gamma=0.99, G_max=5.0, device="cpu"
        )
        normalizer.update_reward_stats(
            reward=torch.tensor([1.0, 2.0]),
            terminated=torch.zeros(2),
            truncated=torch.zeros(2),
        )
        count = normalizer.G_rms.count.clone()
        maximum = normalizer.G_r_max.clone()
        self.assertGreater(torch.count_nonzero(normalizer.G_r).item(), 0)
        normalizer.reset_returns()
        torch.testing.assert_close(normalizer.G_r, torch.zeros(1))
        torch.testing.assert_close(normalizer.G_rms.count, count)
        torch.testing.assert_close(normalizer.G_r_max, maximum)
        normalizer.update_reward_stats(
            reward=torch.ones(3),
            terminated=torch.zeros(3),
            truncated=torch.zeros(3),
        )
        self.assertEqual(tuple(normalizer.G_r.shape), (3,))

    def test_replay_n_step_terminal_boundary_and_ownership(self):
        replay = flashsac_replay_buffer.FlashSACReplayBuffer(
            observation_shape=(1,),
            action_shape=(1,),
            max_length=16,
            min_length=1,
            sample_batch_size=1,
            n_step=3,
            gamma=0.5,
            device="cpu",
        )
        first = _transition([0.0], [10.0], 1.0)
        replay.add(first)
        first["observation"].fill_(999.0)
        first["reward"].fill_(999.0)
        replay.add(_transition([10.0], [20.0], 2.0, terminated=1.0))
        replay.add(_transition([20.0], [30.0], 4.0))

        sample = replay.sample(sample_indices=[0])
        torch.testing.assert_close(sample["observation"], torch.tensor([[0.0]]))
        torch.testing.assert_close(sample["next_observation"], torch.tensor([[20.0]]))
        torch.testing.assert_close(sample["reward"], torch.tensor([2.0]))
        torch.testing.assert_close(sample["terminated"], torch.tensor([1.0]))
        self.assertEqual(replay.pending_length, 3)
        replay.clear_pending()
        self.assertEqual(replay.pending_length, 0)
        self.assertEqual(len(replay), 1)

    def test_replay_timeout_boundary(self):
        replay = flashsac_replay_buffer.FlashSACReplayBuffer(
            observation_shape=(1,),
            action_shape=(1,),
            max_length=8,
            min_length=1,
            sample_batch_size=1,
            n_step=2,
            gamma=0.9,
            device="cpu",
        )
        replay.add(_transition([0.0], [1.0], 1.0, truncated=1.0))
        replay.add(_transition([1.0], [2.0], 7.0))
        sample = replay.sample(sample_indices=[0])
        torch.testing.assert_close(sample["reward"], torch.tensor([1.0]))
        torch.testing.assert_close(sample["next_observation"], torch.tensor([[1.0]]))
        torch.testing.assert_close(sample["terminated"], torch.tensor([0.0]))
        torch.testing.assert_close(sample["truncated"], torch.tensor([1.0]))

    def test_replay_wrap_and_compact_checkpoint(self):
        replay = flashsac_replay_buffer.FlashSACReplayBuffer(
            observation_shape=(2,),
            action_shape=(1,),
            max_length=100_000,
            min_length=1,
            sample_batch_size=1,
            n_step=1,
            gamma=0.99,
            device="cpu",
        )
        replay.add(_transition([1.0, 2.0], [2.0, 3.0], 1.0))
        replay.add(_transition([3.0, 4.0], [4.0, 5.0], 2.0))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "replay.pt")
            replay.save(path)
            self.assertLess(os.path.getsize(path), 100_000)
            restored = flashsac_replay_buffer.FlashSACReplayBuffer(
                observation_shape=(2,),
                action_shape=(1,),
                max_length=100_000,
                min_length=1,
                sample_batch_size=1,
                n_step=1,
                gamma=0.99,
                device="cpu",
            )
            restored.load(path)
        self.assertEqual(len(restored), len(replay))
        for index in range(len(replay)):
            expected = replay.sample(sample_indices=[index])
            actual = restored.sample(sample_indices=[index])
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key])

    def test_replay_checkpoint_drops_pending_n_step_transitions(self):
        replay = flashsac_replay_buffer.FlashSACReplayBuffer(
            observation_shape=(1,),
            action_shape=(1,),
            max_length=8,
            min_length=1,
            sample_batch_size=1,
            n_step=3,
            gamma=0.99,
            device="cpu",
        )
        replay.add(_transition([0.0], [1.0], 1.0))
        self.assertEqual(replay.pending_length, 1)

        restored = flashsac_replay_buffer.FlashSACReplayBuffer(
            observation_shape=(1,),
            action_shape=(1,),
            max_length=8,
            min_length=1,
            sample_batch_size=1,
            n_step=3,
            gamma=0.99,
            device="cpu",
        )
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(restored.pending_length, 0)

    def test_scheduler_boundaries(self):
        schedule = flashsac_model.warmup_cosine_decay_scheduler(
            init_value=0.0,
            peak_value=1.0,
            end_value=0.2,
            warmup_steps=2,
            decay_steps=6,
        )
        self.assertEqual(schedule(0), 0.0)
        self.assertEqual(schedule(2), 1.0)
        self.assertAlmostEqual(schedule(6), 0.2)
        self.assertAlmostEqual(schedule(100), 0.2)

    def test_replay_ring_wrap_preserves_next_write_position(self):
        replay = flashsac_replay_buffer.FlashSACReplayBuffer(
            observation_shape=(1,),
            action_shape=(1,),
            max_length=5,
            min_length=1,
            sample_batch_size=1,
            n_step=1,
            gamma=0.99,
            device="cpu",
        )
        for value in range(7):
            replay.add(_transition([value], [value + 1], float(value)))
        physical = replay.sample(sample_indices=range(5))["observation"].flatten()
        torch.testing.assert_close(physical, torch.tensor([5.0, 6.0, 2.0, 3.0, 4.0]))

        restored = flashsac_replay_buffer.FlashSACReplayBuffer(
            observation_shape=(1,),
            action_shape=(1,),
            max_length=5,
            min_length=1,
            sample_batch_size=1,
            n_step=1,
            gamma=0.99,
            device="cpu",
        )
        restored.load_state_dict(replay.state_dict())
        restored.add(_transition([7.0], [8.0], 7.0))
        physical = restored.sample(sample_indices=range(5))["observation"].flatten()
        torch.testing.assert_close(physical, torch.tensor([5.0, 6.0, 7.0, 3.0, 4.0]))


if __name__ == "__main__":
    unittest.main()

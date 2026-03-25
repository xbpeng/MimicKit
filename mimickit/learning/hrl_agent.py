"""Hierarchical RL agent for ASE task training.

Trains a High-Level Controller (HLC) that outputs latent skill vectors,
which drive a frozen Low-Level Controller (LLC) trained with ASE.

The HLC is a standard PPO policy where:
  - Observation = LLC obs + task obs (e.g., target location, heading direction)
  - Action = latent vector (64-dim, L2-normalized to unit sphere)
  - Reward = task_reward * task_w + disc_reward * disc_w

The LLC is loaded from a pre-trained ASE checkpoint and frozen.
Each HLC step runs the LLC for `llc_steps` environment steps.

Usage:
    python mimickit/run.py \\
        --arg_file args/hrl_steering_humanoid_sword_shield_args.txt \\
        --mode train
"""

import copy
import numpy as np
import torch
import yaml

import learning.ppo_agent as ppo_agent
import learning.ppo_model as ppo_model
import learning.ase_agent as ase_agent
import learning.ase_model as ase_model
import learning.base_agent as base_agent
import util.torch_util as torch_util


class HRLAgent(ppo_agent.PPOAgent):
    """PPO agent that outputs latent vectors to drive a frozen ASE LLC."""

    def __init__(self, config, env, device):
        # Load LLC config before super().__init__ so we know the latent dim
        self._llc_config_path = config["llc_config"]
        with open(self._llc_config_path, 'r') as f:
            self._llc_config = yaml.safe_load(f)
        self._latent_dim = self._llc_config["latent_dim"]

        super().__init__(config, env, device)

        # Build and freeze the LLC
        llc_checkpoint = config["llc_checkpoint"]
        assert llc_checkpoint, "llc_checkpoint must be specified"
        self._llc_steps = config.get("llc_steps", 5)
        self._build_llc(llc_checkpoint)

        # Task obs size (appended to LLC obs by the task env)
        self._task_obs_size = self._get_task_obs_size()

        # Reward weights
        self._task_reward_w = config.get("task_reward_weight", 0.5)
        self._disc_reward_w = config.get("disc_reward_weight", 0.5)

        return

    def _get_task_obs_size(self):
        """Infer task obs size: total obs - LLC obs."""
        total_obs = self._env.get_obs_space().shape[0]
        llc_obs = self._llc_agent._obs_norm._mean.shape[0]
        task_size = total_obs - llc_obs
        assert task_size >= 0, f"Task obs size is negative: {total_obs} - {llc_obs}"
        return task_size

    # ------------------------------------------------------------------
    # Override action space: HLC outputs latent vectors, not motor torques
    # ------------------------------------------------------------------

    def _get_action_size(self):
        return self._latent_dim

    # ------------------------------------------------------------------
    # Override env stepping: run LLC for llc_steps per HLC step
    # ------------------------------------------------------------------

    def _step_env(self, action):
        """HLC action = latent vector. Run LLC for llc_steps, accumulate rewards."""
        # Normalize latent to unit sphere
        z = torch.nn.functional.normalize(action, dim=-1)

        total_reward = torch.zeros(action.shape[0], device=self._device)
        total_disc_reward = torch.zeros(action.shape[0], device=self._device)
        done_any = torch.zeros(action.shape[0], dtype=torch.bool, device=self._device)

        obs = None
        info = {}

        for t in range(self._llc_steps):
            # Compute LLC action from current obs + latent
            llc_obs = self._get_llc_obs()
            llc_action = self._compute_llc_action(llc_obs, z)

            # Step environment with LLC action
            obs, r, done, info = self._env.step(llc_action)

            total_reward += r
            done_any = done_any | done

            # Compute discriminator reward for style preservation
            if hasattr(self._env, 'fetch_disc_obs') and hasattr(self._llc_agent, '_calc_disc_rewards'):
                try:
                    disc_obs = self._env.fetch_disc_obs(num_samples=1)
                    disc_r = self._llc_agent._calc_disc_rewards(disc_obs)
                    total_disc_reward += disc_r.squeeze(-1)
                except:
                    pass

        # Average rewards over LLC steps
        total_reward /= self._llc_steps
        total_disc_reward /= self._llc_steps

        # Combine task + disc rewards
        combined_reward = self._task_reward_w * total_reward + self._disc_reward_w * total_disc_reward

        return obs, combined_reward, done_any, info

    def _get_llc_obs(self):
        """Get the LLC portion of the observation (without task obs)."""
        full_obs = self._env._compute_obs()
        if self._task_obs_size > 0:
            llc_obs = full_obs[..., :-self._task_obs_size]
        else:
            llc_obs = full_obs
        return llc_obs

    def _compute_llc_action(self, llc_obs, z):
        """Run frozen LLC: (obs, latent) → action."""
        with torch.no_grad():
            norm_obs = self._llc_agent._obs_norm.normalize(llc_obs)
            norm_action_dist = self._llc_agent._model.eval_actor(norm_obs, z)
            norm_a = norm_action_dist.mode  # deterministic
            a = self._llc_agent._a_norm.unnormalize(norm_a)
        return a

    # ------------------------------------------------------------------
    # Build and freeze the LLC
    # ------------------------------------------------------------------

    def _build_llc(self, checkpoint_path):
        """Load a pre-trained ASE agent as the frozen LLC."""
        # Build a minimal ASE agent config from the LLC config
        llc_config = copy.deepcopy(self._llc_config)

        # Create a temporary ASE agent to load the LLC
        # We need to construct it carefully to avoid re-initializing the env
        self._llc_agent = _load_frozen_llc(
            llc_config, checkpoint_path, self._env, self._device
        )
        print(f"Loaded frozen LLC from {checkpoint_path}")
        print(f"  LLC obs_dim={self._llc_agent._obs_norm._mean.shape[0]}")
        print(f"  LLC act_dim={self._llc_agent._a_norm._mean.shape[0]}")
        print(f"  Latent dim={self._latent_dim}")
        print(f"  LLC steps per HLC step={self._llc_steps}")
        return


def _load_frozen_llc(config, checkpoint_path, env, device):
    """Load an ASE agent from checkpoint and freeze all parameters."""
    agent = ase_agent.ASEAgent(config=config, env=env, device=device)
    agent.load(checkpoint_path)
    agent.eval()

    # Freeze all LLC parameters
    for param in agent._model.parameters():
        param.requires_grad = False
    agent._model.eval()

    return agent

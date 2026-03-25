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
        # latent_dim may be at top level or under model
        self._latent_dim = (self._llc_config.get("latent_dim")
                           or self._llc_config.get("model", {}).get("latent_dim")
                           or config.get("latent_dim", 64))

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

    def get_action_size(self):
        return self._latent_dim

    def _patch_env_action_space(self):
        """Temporarily replace env action space with latent dim."""
        try:
            import gymnasium.spaces as spaces
        except ImportError:
            import gym.spaces as spaces
        self._orig_get_action_space = self._env.get_action_space
        latent_space = spaces.Box(low=-1, high=1, shape=(self._latent_dim,), dtype=np.float32)
        self._env.get_action_space = lambda: latent_space

    def _unpatch_env_action_space(self):
        self._env.get_action_space = self._orig_get_action_space

    def _build_model(self, config):
        self._patch_env_action_space()
        super()._build_model(config)
        self._unpatch_env_action_space()

    def _build_normalizers(self):
        self._patch_env_action_space()
        super()._build_normalizers()
        self._unpatch_env_action_space()

    # ------------------------------------------------------------------
    # Override env stepping: run LLC for llc_steps per HLC step
    # ------------------------------------------------------------------

    def _step_env(self, action):
        """HLC action = latent vector. Run LLC for llc_steps, accumulate rewards."""
        # Normalize latent to unit sphere
        z = torch.nn.functional.normalize(action, dim=-1)

        total_reward = torch.zeros(action.shape[0], device=self._device)
        done_any = torch.zeros(action.shape[0], dtype=torch.bool, device=self._device)

        obs = None
        info = {}
        llc_obs_dim = self._llc_agent._obs_norm._mean.shape[0]

        import time as _time
        _t0 = _time.time()
        for t in range(self._llc_steps):
            # Get current obs for LLC (strip task obs)
            if obs is None:
                # First step: use the obs from the rollout loop
                full_obs = self._curr_obs
            else:
                full_obs = obs
            llc_obs = full_obs[..., :llc_obs_dim]

            if t == 0 and not hasattr(self, '_llc_debug_printed'):
                self._llc_debug_printed = True
                print(f"HRL DEBUG: full_obs.shape={full_obs.shape}, llc_obs_dim={llc_obs_dim}, llc_obs.shape={llc_obs.shape}, z.shape={z.shape}")

            # Compute LLC action
            llc_action = self._compute_llc_action(llc_obs, z)

            # Step environment with LLC action
            obs, r, done, info = self._env.step(llc_action)

            total_reward += r
            done_any = done_any | done

        # Average rewards over LLC steps
        total_reward /= self._llc_steps

        if not hasattr(self, '_step_count'):
            self._step_count = 0
            self._step_time = 0
        self._step_count += 1
        self._step_time += _time.time() - _t0
        if self._step_count % 32 == 0:
            avg = self._step_time / self._step_count * 1000
            print(f"  HRL _step_env: {avg:.1f}ms avg ({self._llc_steps} LLC steps × {action.shape[0]} envs)")
            self._step_count = 0
            self._step_time = 0

        return obs, total_reward, done_any, info

    def _get_llc_obs(self):
        """Get the LLC portion of the observation (without task obs).

        The task env appends task_obs to the regular obs. The LLC
        only needs the first `llc_obs_dim` dimensions.
        """
        # Use the last observation from the env (set by step/reset)
        full_obs = self._env.get_obs()
        llc_obs_dim = self._llc_agent._obs_norm._mean.shape[0]
        llc_obs = full_obs[..., :llc_obs_dim]
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
    """Load an ASE agent from checkpoint and freeze all parameters.

    The LLC was trained without task obs, so we wrap the env to hide
    the task obs dimensions during LLC construction.
    """
    import torch

    # Load checkpoint to get the LLC's obs dim
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    llc_obs_dim = ckpt["_obs_norm._mean"].shape[0]
    total_obs_dim = env.get_obs_space().shape[0]
    task_obs_size = total_obs_dim - llc_obs_dim

    # Create a wrapper env that reports the LLC's obs space (without task obs)
    class LLCEnvWrapper:
        """Thin wrapper that hides task obs from the LLC during construction."""
        def __init__(self, env, llc_obs_dim):
            self._env = env
            self._llc_obs_dim = llc_obs_dim
        def __getattr__(self, name):
            if name in ('_env', '_llc_obs_dim'):
                return object.__getattribute__(self, name)
            return getattr(self._env, name)
        def get_obs_space(self):
            try:
                import gym.spaces as spaces
            except ImportError:
                import gymnasium.spaces as spaces
            orig = self._env.get_obs_space()
            return spaces.Box(low=orig.low[:self._llc_obs_dim],
                              high=orig.high[:self._llc_obs_dim],
                              dtype=orig.dtype)

    wrapped_env = LLCEnvWrapper(env, llc_obs_dim)
    agent = ase_agent.ASEAgent(config=config, env=wrapped_env, device=device)
    agent.load(checkpoint_path)
    agent.eval()

    # Freeze all LLC parameters
    for param in agent._model.parameters():
        param.requires_grad = False
    agent._model.eval()

    print(f"  LLC obs_dim={llc_obs_dim}, task_obs_size={task_obs_size}")
    return agent

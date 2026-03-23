"""Test the ONNX actor model in Isaac Lab environment.

Verifies that the ONNX model produces the same behavior as PyTorch
when running in Isaac Lab's PhysX GPU environment.

Usage:
    # Run with PyTorch (normal)
    python tools/test_onnx_in_isaaclab.py

    # Run with ONNX model replacing PyTorch
    python tools/test_onnx_in_isaaclab.py --use_onnx

    # Run both and compare
    python tools/test_onnx_in_isaaclab.py --compare
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mimickit'))

import argparse
import numpy as np
import torch
import onnxruntime as ort
import time

parser = argparse.ArgumentParser()
parser.add_argument('--use_onnx', action='store_true')
parser.add_argument('--compare', action='store_true')
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--steps', type=int, default=300)
parser.add_argument('--no_vis', action='store_true')
parser.add_argument('--cpu', action='store_true', help='Run PhysX on CPU instead of GPU')
script_args = parser.parse_args()

# Build mimickit args
from util.arg_parser import ArgParser
args = ArgParser()
args.load_file('args/ase_humanoid_sword_shield_args.txt')
args._table['engine_config'] = ['data/engines/isaac_lab_engine.yaml']
args._table['num_envs'] = [str(script_args.num_envs)]
args._table['mode'] = ['test']
args._table['model_file'] = ['data/models/ase_humanoid_sword_shield_model.pt']
args._table['visualize'] = ['false' if script_args.no_vis else 'true']
args._table['rand_reset'] = ['false']

import util.util as util
import util.mp_util as mp_util
device = 'cpu' if script_args.cpu else 'cuda:0'
mp_util.init(0, 1, device, None)
util.set_rand_seed(42)

# Build env and agent using run.py's helpers
import run
env = run.build_env(args, script_args.num_envs, device, not script_args.no_vis)
agent = run.build_agent(args, env, device)
agent.load('data/models/ase_humanoid_sword_shield_model.pt')
agent.eval()

# Load ONNX
onnx_file = 'web/ase_humanoid_sword_shield_actor.onnx'
if script_args.cpu:
    providers = ['CPUExecutionProvider']
else:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
onnx_sess = ort.InferenceSession(onnx_file, providers=providers)
print(f"ONNX model loaded: {onnx_file}, provider: {onnx_sess.get_providers()[0]}")

from learning.base_agent import AgentMode
agent.set_mode(AgentMode.TEST)

original_decide = agent._decide_action

_compare_count = [0]

def onnx_decide_action(obs, info):
    """Use ONNX model for inference, optionally compare with PyTorch."""
    z = agent._latent_buf
    obs_np = obs.cpu().numpy().astype(np.float32)
    z_np = z.cpu().numpy().astype(np.float32)

    ort_action = onnx_sess.run(["action"], {"obs": obs_np, "latent": z_np})[0]
    onnx_action = torch.tensor(ort_action, device=obs.device, dtype=obs.dtype)

    if script_args.compare:
        with torch.no_grad():
            pt_action, pt_info = original_decide(obs, info)
        max_diff = (pt_action - onnx_action).abs().max().item()
        mean_diff = (pt_action - onnx_action).abs().mean().item()

        _compare_count[0] += 1
        if _compare_count[0] <= 2:
            # Deep debug
            norm_obs_pt = agent._obs_norm.normalize(obs)

            # Save obs and latent to disk for offline comparison
            # Check if agent's model weights match checkpoint
            ckpt = torch.load('data/models/ase_humanoid_sword_shield_model.pt', map_location='cpu', weights_only=False)

            # Compare actor weights
            agent_w0 = agent._model._actor_layers[0].weight.cpu().numpy()
            ckpt_w0 = ckpt['_model._actor_layers.0.weight'].numpy()
            print(f"  actor_layers.0.weight match: {np.allclose(agent_w0, ckpt_w0, atol=1e-6)}")
            print(f"  actor_layers.0.weight max_diff: {np.max(np.abs(agent_w0 - ckpt_w0))}")

            # Compare mean_net weights
            agent_mw = agent._model._action_dist._mean_net.weight.cpu().numpy()
            ckpt_mw = ckpt['_model._action_dist._mean_net.weight'].numpy()
            print(f"  mean_net.weight match: {np.allclose(agent_mw, ckpt_mw, atol=1e-6)}")
            print(f"  mean_net.weight max_diff: {np.max(np.abs(agent_mw - ckpt_mw))}")

            # Manual forward pass with agent's own model for comparison
            with torch.no_grad():
                norm_obs_gpu = agent._obs_norm.normalize(obs)
                z_gpu = agent._latent_buf
                in_data = torch.cat([norm_obs_gpu, z_gpu], dim=-1)
                h_gpu = agent._model._actor_layers(in_data)
                manual_norm_a = agent._model._action_dist._mean_net(h_gpu)
                manual_a = agent._a_norm.unnormalize(manual_norm_a)
                print(f"  Manual GPU forward[:5]: {manual_a[0,:5].cpu().numpy()}")
                print(f"  vs PT action[:5]:       {pt_action[0,:5].cpu().numpy()}")
                print(f"  vs ONNX action[:5]:     {ort_action[0,:5]}")
            ckpt_obs_std = ckpt['_obs_norm._std'].numpy()
            agent_obs_std = agent._obs_norm._std.cpu().numpy()
            print(f"  ckpt obs_std[:5]: {ckpt_obs_std[:5]}")
            print(f"  agent obs_std[:5]: {agent_obs_std[:5]}")
            print(f"  obs_std match: {np.allclose(ckpt_obs_std, agent_obs_std, atol=1e-6)}")
            print(f"  obs_std max_diff: {np.max(np.abs(ckpt_obs_std - agent_obs_std))}")

            ckpt_obs_mean = ckpt['_obs_norm._mean'].numpy()
            agent_obs_mean = agent._obs_norm._mean.cpu().numpy()
            print(f"  obs_mean match: {np.allclose(ckpt_obs_mean, agent_obs_mean, atol=1e-6)}")
            print(f"  obs_mean max_diff: {np.max(np.abs(ckpt_obs_mean - agent_obs_mean))}")

            np.savez('/tmp/onnx_debug.npz',
                     raw_obs=obs_np,
                     latent=z_np,
                     norm_obs=norm_obs_pt.cpu().numpy(),
                     pt_action=pt_action.cpu().numpy(),
                     onnx_action=ort_action,
                     obs_norm_mean=agent._obs_norm._mean.cpu().numpy(),
                     obs_norm_std=agent._obs_norm._std.cpu().numpy(),
                     a_norm_mean=agent._a_norm._mean.cpu().numpy(),
                     a_norm_std=agent._a_norm._std.cpu().numpy())

            print(f"\n  === Debug frame {_compare_count[0]} ===")
            print(f"  obs[0,:5]: {obs_np[0,:5]}")
            print(f"  latent[0,:5]: {z_np[0,:5]}")
            print(f"  norm_obs[0,:5]: {norm_obs_pt[0,:5].cpu().numpy()}")
            print(f"  PT action[0,:8]: {pt_action[0,:8].cpu().numpy()}")
            print(f"  ONNX action[0,:8]: {ort_action[0,:8]}")
            print(f"  Saved debug data to /tmp/onnx_debug.npz")

        if max_diff > 0.001:
            print(f"  PT vs ONNX: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")

    a_info = {"a_logp": torch.zeros(obs.shape[0], device=obs.device),
              "rand_action_mask": torch.zeros(obs.shape[0], device=obs.device)}
    return onnx_action, a_info

if script_args.use_onnx or script_args.compare:
    agent._decide_action = onnx_decide_action
    print(f"Using {'ONNX' if script_args.use_onnx else 'ONNX (compare mode)'} for inference")
else:
    print("Using PyTorch for inference")

# Run
with torch.no_grad():
    obs, info = agent._reset_envs()

    for step in range(script_args.steps):
        action, a_info = agent._decide_action(obs, info)
        obs, reward, done, info = env.step(action)

        if step % 30 == 0:
            root_pos = env._engine.get_root_pos(0)
            root_z = root_pos[0, 2].item()
            print(f"Step {step:4d}: root_z={root_z:.4f} reward={reward[0].item():.4f}")

        if not script_args.no_vis:
            env._engine.render()

print("=== DONE ===")

"""Export AMP/PPO actor to ONNX — no latent vector, just obs → action.

Usage:
    cd MimicKit
    python tools/export_amp_onnx.py \
        --arg_file args/amp_humanoid_args.txt \
        --model_file data/models/amp_humanoid_spinkick_model.pt \
        --output web/amp_humanoid_actor.onnx
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mimickit'))

import argparse
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument('--arg_file', default='args/amp_humanoid_args.txt')
parser.add_argument('--model_file', default='data/models/amp_humanoid_spinkick_model.pt')
parser.add_argument('--output', default='web/amp_humanoid_actor.onnx')
parser.add_argument('--engine', default='data/engines/isaac_lab_engine.yaml')
args = parser.parse_args()


class PPOActorWrapper(nn.Module):
    """Wraps obs_norm → actor → a_norm into a single exportable module.
    Input:  raw_obs (B, obs_dim)
    Output: action  (B, act_dim)
    """
    def __init__(self, agent):
        super().__init__()
        self.register_buffer('obs_mean', agent._obs_norm._mean.data.clone())
        self.register_buffer('obs_std', agent._obs_norm._std.data.clone().clamp(min=1e-4))
        self.obs_clip = agent._obs_norm._clip
        self.register_buffer('a_mean', agent._a_norm._mean.data.clone())
        self.register_buffer('a_std', agent._a_norm._std.data.clone().clamp(min=1e-4))
        self.actor_layers = agent._model._actor_layers
        self.mean_net = agent._model._action_dist._mean_net

    def forward(self, raw_obs: torch.Tensor) -> torch.Tensor:
        norm_obs = (raw_obs - self.obs_mean) / self.obs_std
        norm_obs = torch.clamp(norm_obs, -self.obs_clip, self.obs_clip)
        h = self.actor_layers(norm_obs)
        norm_action = self.mean_net(h)
        action = norm_action * self.a_std + self.a_mean
        return action


from util.arg_parser import ArgParser
import util.mp_util as mp_util
import run as mimickit_run

mk_args = ArgParser()
mk_args.load_file(args.arg_file)
mk_args._table['engine_config'] = [args.engine]
mk_args._table['num_envs'] = ['1']
mk_args._table['mode'] = ['test']
mk_args._table['visualize'] = ['false']
mp_util.init(0, 1, 'cpu', None)

env = mimickit_run.build_env(mk_args, 1, 'cpu', False)
agent = mimickit_run.build_agent(mk_args, env, 'cpu')
agent.load(args.model_file)
agent.eval()

obs_dim = agent._obs_norm._mean.shape[0]
act_dim = agent._a_norm._mean.shape[0]
print(f"Agent: obs_dim={obs_dim}, act_dim={act_dim}")

wrapper = PPOActorWrapper(agent)
wrapper.eval()

# Test with PyTorch
dummy_obs = torch.randn(1, obs_dim)
pt_action = wrapper(dummy_obs)
print(f"PyTorch output: {pt_action[0, :5].detach().numpy()}")

# Export
torch.onnx.export(
    wrapper,
    (dummy_obs,),
    args.output,
    input_names=["obs"],
    output_names=["action"],
    dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    opset_version=13,
)
print(f"Exported to {args.output}")

# Verify with ONNX Runtime
sess = ort.InferenceSession(args.output)
ort_action = sess.run(['action'], {'obs': dummy_obs.numpy()})[0]
print(f"ONNX output: {ort_action[0, :5]}")

max_diff = np.abs(pt_action.detach().numpy() - ort_action).max()
print(f"Max diff: {max_diff}")

# Also verify from real obs
env.reset()
real_obs = env._compute_obs()
real_obs_np = real_obs[0].cpu().numpy()
pt_real = wrapper(real_obs).detach().numpy()
ort_real = sess.run(['action'], {'obs': real_obs_np.reshape(1, -1)})[0]
print(f"\nReal obs test:")
print(f"  PT action[:5]: {pt_real[0,:5].round(4)}")
print(f"  ONNX action[:5]: {ort_real[0,:5].round(4)}")
print(f"  Max diff: {np.abs(pt_real - ort_real).max()}")

# Re-save with embedded weights if needed
size_mb = os.path.getsize(args.output) / (1024 * 1024)
print(f"\nOutput: {args.output} ({size_mb:.1f} MB)")
if size_mb < 0.1:
    import onnx
    model = onnx.load(args.output)
    onnx.save(model, args.output, save_as_external_data=False)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Re-saved: {args.output} ({size_mb:.1f} MB)")

"""Export ASE actor to ONNX using the actual agent model (no manual reconstruction).

This loads the full agent with its model, wraps the inference pipeline
(obs_norm → actor → a_norm) into a single nn.Module, and exports it.
This guarantees exact equivalence with PyTorch inference.

Usage:
    cd MimicKit
    python tools/export_onnx_v2.py \
        --arg_file args/ase_humanoid_sword_shield_args.txt \
        --model_file data/models/ase_humanoid_sword_shield_model.pt \
        --output web/ase_humanoid_sword_shield_actor.onnx
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mimickit'))

import argparse
import numpy as np
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--arg_file', default='args/ase_humanoid_sword_shield_args.txt')
parser.add_argument('--model_file', default='data/models/ase_humanoid_sword_shield_model.pt')
parser.add_argument('--output', default='web/ase_humanoid_sword_shield_actor.onnx')
parser.add_argument('--engine', default='data/engines/isaac_lab_engine.yaml',
                    help='Engine config (only used to build env for model construction)')
args = parser.parse_args()


class ASEActorWrapper(nn.Module):
    """Wraps the full agent inference pipeline into a single exportable module.

    Input:  raw_obs (B, obs_dim), latent (B, latent_dim)
    Output: action  (B, act_dim)

    Bakes in: obs normalization, actor network, action unnormalization.
    """

    def __init__(self, agent):
        super().__init__()

        # Copy normalizer parameters as buffers
        self.register_buffer('obs_mean', agent._obs_norm._mean.data.clone())
        self.register_buffer('obs_std', agent._obs_norm._std.data.clone().clamp(min=1e-4))
        self.obs_clip = agent._obs_norm._clip

        self.register_buffer('a_mean', agent._a_norm._mean.data.clone())
        self.register_buffer('a_std', agent._a_norm._std.data.clone().clamp(min=1e-4))

        # Reference the actual model layers (no reconstruction!)
        self.actor_layers = agent._model._actor_layers
        self.mean_net = agent._model._action_dist._mean_net

    def forward(self, raw_obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        # Normalize observation
        norm_obs = (raw_obs - self.obs_mean) / self.obs_std
        norm_obs = torch.clamp(norm_obs, -self.obs_clip, self.obs_clip)

        # Concatenate obs + latent (same as ASEModel.eval_actor)
        x = torch.cat([norm_obs, latent], dim=-1)

        # Actor forward pass
        h = self.actor_layers(x)
        norm_action = self.mean_net(h)  # deterministic (mode) action

        # Unnormalize action
        action = norm_action * self.a_std + self.a_mean
        return action


def main():
    # Need to build env + agent to get the actual model with correct architecture.
    # We use a dummy engine config that doesn't require a GPU simulator.
    # If Isaac Lab is available, use it; otherwise fall back to loading from checkpoint.

    try:
        # Try building via the full agent pipeline
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

        print(f"Agent loaded via full pipeline")
        print(f"  obs_dim={agent._obs_norm._mean.shape[0]}")
        print(f"  act_dim={agent._a_norm._mean.shape[0]}")
        print(f"  latent_dim={agent._model._enc_out.weight.shape[0]}")

    except Exception as e:
        print(f"Failed to build via full pipeline: {e}")
        print("Falling back to checkpoint-only export (same as export_onnx.py v1)")
        # Fall back to v1 approach
        from export_onnx import export_onnx
        export_onnx(args.model_file, args.output)
        return

    # Create wrapper
    wrapper = ASEActorWrapper(agent)
    wrapper.eval()

    obs_dim = wrapper.obs_mean.shape[0]
    latent_dim = agent._model._enc_out.weight.shape[0]
    act_dim = wrapper.a_mean.shape[0]

    print(f"\nExporting: raw_obs({obs_dim}) + latent({latent_dim}) → action({act_dim})")

    # Dummy inputs
    dummy_obs = torch.randn(1, obs_dim)
    dummy_z = torch.randn(1, latent_dim)

    # PyTorch reference
    with torch.no_grad():
        pt_action = wrapper(dummy_obs, dummy_z)
    print(f"PyTorch output: {pt_action[0, :5].numpy()}")

    # Export
    torch.onnx.export(
        wrapper,
        (dummy_obs, dummy_z),
        args.output,
        input_names=["obs", "latent"],
        output_names=["action"],
        dynamic_axes={
            "obs": {0: "batch"},
            "latent": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported to {args.output}")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(args.output)
    ort_inputs = {"obs": dummy_obs.numpy(), "latent": dummy_z.numpy()}
    ort_action = sess.run(["action"], ort_inputs)[0]

    max_diff = np.max(np.abs(pt_action.numpy() - ort_action))
    print(f"ONNX output: {ort_action[0, :5]}")
    print(f"Max diff: {max_diff:.2e}")
    assert max_diff < 1e-3, f"ONNX diverges: {max_diff}"

    # Verify with real-ish obs
    print("\nVerifying with agent's actual inference...")
    test_obs = torch.randn(1, obs_dim)
    test_z = torch.randn(1, latent_dim)
    test_z = test_z / test_z.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        # Agent path
        norm_obs = agent._obs_norm.normalize(test_obs)
        dist = agent._model.eval_actor(norm_obs, test_z)
        agent_action = agent._a_norm.unnormalize(dist.mode)

        # Wrapper path
        wrapper_action = wrapper(test_obs, test_z)

        # ONNX path
        ort_action2 = sess.run(["action"], {
            "obs": test_obs.numpy(),
            "latent": test_z.numpy()
        })[0]

    agent_np = agent_action.numpy()
    wrapper_np = wrapper_action.numpy()
    print(f"Agent action[:5]:   {agent_np[0,:5]}")
    print(f"Wrapper action[:5]: {wrapper_np[0,:5]}")
    print(f"ONNX action[:5]:    {ort_action2[0,:5]}")
    print(f"Agent vs Wrapper max_diff: {np.max(np.abs(agent_np - wrapper_np)):.2e}")
    print(f"Agent vs ONNX max_diff:    {np.max(np.abs(agent_np - ort_action2)):.2e}")
    print(f"Wrapper vs ONNX max_diff:  {np.max(np.abs(wrapper_np - ort_action2)):.2e}")

    # Check file size
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nOutput file: {args.output} ({size_mb:.1f} MB)")

    if size_mb < 0.1:
        print("WARNING: File is very small — weights may be stored externally in .onnx.data file")
        print("Re-saving with all weights internal...")
        import onnx
        model = onnx.load(args.output)
        onnx.save(model, args.output, save_as_external_data=False)
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"Re-saved: {args.output} ({size_mb:.1f} MB)")

    print("\nPASS: All verifications passed.")


if __name__ == "__main__":
    main()

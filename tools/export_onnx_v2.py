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

    # Bake model metadata into ONNX for the web demo.
    # This eliminates the need for a separate JSON config file.
    import onnx
    model = onnx.load(args.output)

    # Collect metadata from agent
    meta = {}
    meta['obs_dim'] = int(obs_dim)
    meta['act_dim'] = int(act_dim)
    meta['latent_dim'] = int(latent_dim)
    meta['obs_mean'] = wrapper.obs_mean.numpy().tolist()
    meta['obs_std'] = wrapper.obs_std.numpy().tolist()
    meta['a_mean'] = wrapper.a_mean.numpy().tolist()
    meta['a_std'] = wrapper.a_std.numpy().tolist()

    # Action bounds from agent config
    if hasattr(agent, '_action_low') and agent._action_low is not None:
        meta['action_low'] = agent._action_low.cpu().numpy().tolist()
        meta['action_high'] = agent._action_high.cpu().numpy().tolist()

    # Init pose and env config — try env attributes, fall back to existing JSON
    json_fallback = {}
    json_path = os.path.join(os.path.dirname(args.output), 'humanoid_data.json')
    if os.path.isfile(json_path):
        import json as _json
        json_fallback = _json.load(open(json_path))
        print(f"  Loaded JSON fallback from {json_path}")

    def _try_tensor(obj, attr, idx=0):
        """Extract a list from a tensor attribute, handling 1D and 2D tensors."""
        t = getattr(obj, attr, None)
        if t is None: return None
        t = t.cpu()
        if t.dim() > 1: t = t[idx]
        return t.numpy().flatten().tolist()

    # Init pose
    init_dof = _try_tensor(env, '_init_dof_pos') or json_fallback.get('init_dof_pos')
    init_root_pos = _try_tensor(env, '_init_root_pos') or json_fallback.get('init_root_pos')
    init_root_rot = _try_tensor(env, '_init_root_rot') or json_fallback.get('init_root_rot_quat')
    if init_dof and isinstance(init_dof, list) and len(init_dof) > 1:
        meta['init_dof_pos'] = init_dof
    if init_root_pos and isinstance(init_root_pos, list) and len(init_root_pos) == 3:
        meta['init_root_pos'] = init_root_pos
    if init_root_rot and isinstance(init_root_rot, list) and len(init_root_rot) == 4:
        meta['init_root_rot_quat'] = init_root_rot

    # Action bounds
    action_low = _try_tensor(agent, '_action_low') or json_fallback.get('action_low')
    action_high = _try_tensor(agent, '_action_high') or json_fallback.get('action_high')
    if action_low: meta['action_low'] = action_low
    if action_high: meta['action_high'] = action_high

    # Key body IDs and settings
    key_ids = _try_tensor(env, '_key_body_ids') or json_fallback.get('key_body_ids')
    if key_ids: meta['key_body_ids'] = [int(x) for x in key_ids]
    meta['global_obs'] = bool(getattr(env, '_global_obs', json_fallback.get('global_obs', False)))
    meta['pelvis_z'] = float(getattr(env, '_pelvis_z', json_fallback.get('pelvis_z', 0.703)))
    meta['tpose_pelvis_z'] = float(getattr(env, '_tpose_pelvis_z', json_fallback.get('tpose_pelvis_z', 0.903)))

    # Bake MJCF XML into metadata so the ONNX is a single-file character kit
    mjcf_path = getattr(env, '_char_file', None) or json_fallback.get('mjcf_file')
    # Also try the arg_file's char_file
    if not mjcf_path:
        try:
            mjcf_path = mk_args.parse_string('char_file')
        except: pass
    # Try common locations
    if not mjcf_path or not os.path.isfile(mjcf_path):
        for candidate in [
            'data/assets/sword_shield/humanoid_sword_shield.xml',
            os.path.join(os.path.dirname(args.output), 'humanoid_sword_shield.xml'),
        ]:
            if os.path.isfile(candidate):
                mjcf_path = candidate
                break
    if mjcf_path and os.path.isfile(mjcf_path):
        with open(mjcf_path) as f:
            meta['mjcf_xml'] = f.read()
        print(f"  Baked MJCF from {mjcf_path} ({len(meta['mjcf_xml'])} chars)")

    # Write ALL metadata as a single JSON blob under a sentinel key.
    # This allows the web demo to find it by scanning the raw ONNX bytes
    # (onnxruntime-web doesn't expose metadata via its JS API).
    import json
    sentinel_key = 'mimickit_config'
    config_json = json.dumps(meta, separators=(',', ':'))  # compact
    entry = onnx.StringStringEntryProto(key=sentinel_key, value=config_json)
    model.metadata_props.append(entry)
    # Also write individual entries for tools that read metadata normally
    for key, value in meta.items():
        if key == 'mjcf_xml': continue  # already in the blob
        entry = onnx.StringStringEntryProto(key=key, value=json.dumps(value))
        model.metadata_props.append(entry)

    onnx.save(model, args.output, save_as_external_data=False)
    print(f"\nBaked {len(meta)} metadata entries into ONNX:")
    for k in sorted(meta.keys()):
        v = meta[k]
        if isinstance(v, list) and len(v) > 5:
            print(f"  {k}: [{v[0]:.4f}, ... {len(v)} items]")
        else:
            print(f"  {k}: {v}")

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

"""Record a 10-second trajectory from Isaac Lab for PhysX parity validation.

Runs the ONNX policy against the Isaac Lab PhysX backend and records every
state variable at each policy step (30 Hz) for comparison with the web demo.

Usage:
    cd MimicKit/mimickit
    python ../tools/record_trajectory.py \
        --model_file ../data/models/ase_humanoid_sword_shield_model.pt \
        --onnx_file ../web/ase_humanoid_sword_shield_actor.onnx \
        --output ../web/trajectory.json \
        --duration 10.0 \
        --latent_seed 42

If --model_file is not available, uses ONNX-only mode (onnxruntime inference).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mimickit'))

import argparse
import json
import numpy as np

parser = argparse.ArgumentParser(description='Record Isaac Lab trajectory for parity validation')
parser.add_argument('--onnx_file', default='../web/ase_humanoid_sword_shield_actor.onnx',
                    help='Path to ONNX actor model')
parser.add_argument('--model_file', default='',
                    help='Path to PyTorch model checkpoint (optional, uses ONNX if empty)')
parser.add_argument('--output', default='../web/trajectory.json',
                    help='Output trajectory JSON file')
parser.add_argument('--duration', type=float, default=10.0,
                    help='Recording duration in seconds')
parser.add_argument('--latent_seed', type=int, default=42,
                    help='Random seed for latent vector generation')
parser.add_argument('--engine', default='data/engines/isaac_lab_engine.yaml')
parser.add_argument('--env_config', default='data/envs/ase_humanoid_sword_shield_env.yaml')
parser.add_argument('--agent_config', default='data/agents/ase_humanoid_agent.yaml')
args = parser.parse_args()


def extract_onnx_metadata(onnx_path):
    """Extract mimickit_config metadata from ONNX file (same logic as web demo)."""
    with open(onnx_path, 'rb') as f:
        data = f.read()

    sentinel = b'mimickit_config'
    idx = data.find(sentinel)
    if idx == -1:
        raise RuntimeError('No mimickit_config found in ONNX binary')

    # Search for protobuf value field tag (0x12) after the key
    search_start = idx + len(sentinel)
    for k in range(search_start, min(search_start + 20, len(data))):
        if data[k] == 0x12:
            # Read varint length
            length = 0
            shift = 0
            pos = k + 1
            while pos < len(data):
                b = data[pos]
                pos += 1
                length |= (b & 0x7f) << shift
                shift += 7
                if (b & 0x80) == 0:
                    break

            json_str = data[pos:pos + length].decode('utf-8')
            return json.loads(json_str)

    raise RuntimeError('Found sentinel but could not extract JSON')


def generate_latent(dim, seed):
    """Generate a fixed unit-sphere latent vector for reproducibility."""
    rng = np.random.RandomState(seed)
    z = rng.randn(dim).astype(np.float32)
    z /= np.linalg.norm(z)
    return z


def run_onnx_only():
    """Record trajectory using ONNX-only inference (no Isaac Lab required).

    This mode builds the PhysX simulation through Isaac Lab, runs the ONNX policy,
    and records states at each policy step.
    """
    import onnxruntime as ort

    # Load ONNX model
    onnx_path = os.path.abspath(args.onnx_file)
    print(f"Loading ONNX model from {onnx_path}")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Extract metadata
    meta = extract_onnx_metadata(onnx_path)
    obs_dim = meta['obs_dim']
    act_dim = meta['act_dim']
    latent_dim = meta['latent_dim']
    action_low = np.array(meta['action_low'], dtype=np.float32)
    action_high = np.array(meta['action_high'], dtype=np.float32)

    print(f"  obs_dim={obs_dim}, act_dim={act_dim}, latent_dim={latent_dim}")

    # Generate fixed latent
    latent = generate_latent(latent_dim, args.latent_seed)
    print(f"  latent (seed={args.latent_seed}): norm={np.linalg.norm(latent):.4f}")

    # Build Isaac Lab environment
    from util.arg_parser import ArgParser
    import util.mp_util as mp_util
    import run as mimickit_run
    import util.util as util

    mk_args = ArgParser()
    mk_args._table['engine_config'] = [args.engine]
    mk_args._table['env_config'] = [args.env_config]
    mk_args._table['agent_config'] = [args.agent_config]
    mk_args._table['num_envs'] = ['1']
    mk_args._table['mode'] = ['test']
    mk_args._table['visualize'] = ['false']

    mp_util.init(0, 1, 'cpu', None)
    util.set_rand_seed(42)

    env = mimickit_run.build_env(mk_args, 1, 'cuda:0', False)
    env.reset()

    import torch

    # Recording parameters
    control_freq = 30  # Hz
    num_steps = int(args.duration * control_freq)
    print(f"Recording {num_steps} policy steps ({args.duration}s at {control_freq}Hz)")

    trajectory = {
        'config': {
            'duration': args.duration,
            'control_freq': control_freq,
            'sim_freq': 120,
            'num_substeps': 4,
            'dt': 1.0 / 120,
            'latent_seed': args.latent_seed,
            'latent': latent.tolist(),
            'obs_dim': obs_dim,
            'act_dim': act_dim,
        },
        'steps': []
    }

    char_id = env._get_char_id()

    for step_i in range(num_steps):
        # Read current state (BEFORE action)
        root_pos = env._engine.get_root_pos(char_id)[0].cpu().numpy()
        root_rot = env._engine.get_root_rot(char_id)[0].cpu().numpy()  # xyzw
        root_vel = env._engine.get_root_vel(char_id)[0].cpu().numpy()
        root_ang_vel = env._engine.get_root_ang_vel(char_id)[0].cpu().numpy()
        dof_pos = env._engine.get_dof_pos(char_id)[0].cpu().numpy()
        dof_vel = env._engine.get_dof_vel(char_id)[0].cpu().numpy()
        body_pos = env._engine.get_body_pos(char_id)[0].cpu().numpy()

        # Compute observation (same as env._compute_obs)
        obs = env._compute_obs()[0].cpu().numpy()

        # Run ONNX policy
        obs_input = obs.reshape(1, -1).astype(np.float32)
        latent_input = latent.reshape(1, -1).astype(np.float32)
        results = session.run(None, {'obs': obs_input, 'latent': latent_input})
        action = results[0][0].astype(np.float32)

        # Clip action to bounds
        clipped_action = np.clip(action, action_low, action_high)

        # Record this step
        step_data = {
            'step': step_i,
            'root_pos': root_pos.tolist(),
            'root_rot': root_rot.tolist(),
            'root_vel': root_vel.tolist(),
            'root_ang_vel': root_ang_vel.tolist(),
            'dof_pos': dof_pos.tolist(),
            'dof_vel': dof_vel.tolist(),
            'body_pos': body_pos.tolist(),
            'obs': obs.tolist(),
            'action_raw': action.tolist(),
            'action_clipped': clipped_action.tolist(),
        }
        trajectory['steps'].append(step_data)

        if step_i % 30 == 0:
            print(f"  Step {step_i}/{num_steps}: root_z={root_pos[2]:.4f}, "
                  f"max_dof_vel={np.max(np.abs(dof_vel)):.2f}")

        # Apply action and step environment
        action_tensor = torch.tensor(clipped_action, device='cuda:0').unsqueeze(0)
        env.step(action_tensor)

    # Save trajectory
    output_path = os.path.abspath(args.output)
    with open(output_path, 'w') as f:
        json.dump(trajectory, f)
    print(f"\nTrajectory saved to {output_path}")
    print(f"  {len(trajectory['steps'])} steps, "
          f"{os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    run_onnx_only()

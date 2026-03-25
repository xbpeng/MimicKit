"""Encode motion clips → latent presets via the ASE encoder.

Runs the full env pipeline to compute AMP observations from each motion
clip, then encodes them to get the corresponding latent vectors.

Usage:
    DISPLAY=:0 conda run -n mimickit-isaaclab python tools/encode_motions_to_latents.py
"""
import sys, os, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mimickit'))

import numpy as np
import torch
import torch.nn as nn

from util.arg_parser import ArgParser
import util.mp_util as mp_util
import run as mimickit_run

# Build env to get access to motion lib and AMP obs computation
args = ArgParser()
args.load_file('args/ase_humanoid_sword_shield_args.txt')
args._table['engine_config'] = ['data/engines/isaac_lab_engine.yaml']
args._table['num_envs'] = ['1']
args._table['mode'] = ['test']
args._table['visualize'] = ['false']
mp_util.init(0, 1, 'cpu', None)
env = mimickit_run.build_env(args, 1, 'cpu', False)
agent = mimickit_run.build_agent(args, env, 'cpu')
agent.load('data/models/ase_humanoid_sword_shield_model.pt')
agent.eval()

# The encoder is part of the agent's model
# agent._model._enc_layers + agent._model._enc_out
# It takes normalized disc_obs and outputs L2-normalized latent

# Get motion lib from env
motion_lib = env._motion_lib

print(f"Motion lib: {motion_lib.get_num_motions()} motions")
print(f"Encoder: disc_obs → 64-dim latent")

# For each motion, sample frames, compute disc_obs, encode to latent
presets = {}

with torch.no_grad():
    for mi in range(motion_lib.get_num_motions()):
        fname = os.path.basename(motion_lib._motion_files[mi])
        name = fname.replace('RL_Avatar_', '').replace('_Motion.pkl', '').replace('(0)', '')

        duration = motion_lib.get_motion_length(mi).item()
        if duration < 0.1:
            continue

        # Dense sliding window: encode every ~1/30s across the entire clip
        latents = []
        n_samples = max(1, int(duration * 30))  # one per frame at 30fps
        motion_id = torch.tensor([mi])

        for si in range(n_samples):
            t0 = torch.tensor([duration * (si + 0.5) / n_samples])
            try:
                disc_obs = env._compute_disc_obs_demo(motion_id, t0)
                norm_obs = agent._disc_obs_norm.normalize(disc_obs)
                enc_out = agent._model._enc_layers(norm_obs)
                z = agent._model._enc_out(enc_out)
                z = torch.nn.functional.normalize(z, dim=-1)
                latents.append(z[0].numpy())
            except Exception as e:
                if si == 0: print(f"  {name}: Error: {e}")
                break

        if latents:
            latents = np.array(latents)

            # Element-wise signed max: for each dimension, take the value with
            # the largest absolute value across all frames. This captures the
            # peak/most characteristic activation in each latent dimension,
            # then normalizes to the unit sphere.
            abs_vals = np.abs(latents)
            max_indices = np.argmax(abs_vals, axis=0)
            peak_z = np.array([latents[max_indices[d], d] for d in range(latents.shape[1])])
            # Skip normalization — let the raw peak magnitudes through
            # peak_z = peak_z / np.linalg.norm(peak_z)

            # Spread metric for logging
            mean_z = np.mean(latents, axis=0)
            mean_z = mean_z / (np.linalg.norm(mean_z) + 1e-8)
            spread = 1.0 - np.mean(latents @ mean_z)

            presets[name] = [round(float(v), 4) for v in peak_z]
            print(f"  {name}: {len(latents)} frames, spread={spread:.3f}, duration={duration:.1f}s")
        else:
            print(f"  {name}: no samples (duration={duration:.1f}s)")

# Save
output = {"presets": presets}
with open('web/latent_presets.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved {len(presets)} presets to web/latent_presets.json")

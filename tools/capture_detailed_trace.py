"""Capture detailed per-step trace from Isaac Lab CPU with zero latent.

Records full state, full obs, full action, AND the state AFTER each physics step,
so we can compare step-by-step with the web demo.

Starts from init_pose with zero velocities (matching web demo reset).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mimickit'))

import json
import numpy as np
import torch

from util.arg_parser import ArgParser
import util.mp_util as mp_util
import run as mimickit_run

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

from learning.base_agent import AgentMode
agent.set_mode(AgentMode.TEST)

# Zero latent
agent._latent_buf = torch.zeros(1, 64)

NUM_STEPS = 10

def get_state():
    """Get full physical state."""
    char_id = 0
    return {
        'root_pos': env._engine.get_root_pos(char_id)[0].cpu().numpy().tolist(),
        'root_rot': env._engine.get_root_rot(char_id)[0].cpu().numpy().tolist(),
        'root_vel': env._engine.get_root_vel(char_id)[0].cpu().numpy().tolist(),
        'root_ang_vel': env._engine.get_root_ang_vel(char_id)[0].cpu().numpy().tolist(),
        'dof_pos': env._engine.get_dof_pos(char_id)[0].cpu().numpy().tolist(),
        'dof_vel': env._engine.get_dof_vel(char_id)[0].cpu().numpy().tolist(),
    }

trace = []
with torch.no_grad():
    obs, info = agent._reset_envs()
    agent._latent_buf = torch.zeros(1, 64)

    # Force to init_pose with zero velocities (matching web demo)
    char_id = 0
    init_root_pos = env._init_root_pos.unsqueeze(0)
    init_root_rot = env._init_root_rot.unsqueeze(0)
    init_dof_pos = env._init_dof_pos.unsqueeze(0)
    env._engine.set_root_pos([0], char_id, init_root_pos)
    env._engine.set_root_rot([0], char_id, init_root_rot)
    env._engine.set_root_vel([0], char_id, 0.0)
    env._engine.set_root_ang_vel([0], char_id, 0.0)
    env._engine.set_dof_pos([0], char_id, init_dof_pos)
    env._engine.set_dof_vel([0], char_id, 0.0)

    # Recompute obs from the forced state
    obs = env._compute_obs()
    print(f"Forced init_pose: root_z={init_root_pos[0,2].item():.4f}")
    print(f"Init root_rot: {init_root_rot[0].cpu().numpy().round(4).tolist()}")
    print(f"Init dof_pos[:5]: {init_dof_pos[0,:5].cpu().numpy().round(4).tolist()}")

    for step in range(NUM_STEPS):
        state_before = get_state()
        full_obs = obs[0].cpu().numpy().tolist()

        # Get action from policy
        action, a_info = agent._decide_action(obs, info)
        action_list = action[0].cpu().numpy().tolist()

        # Step the environment (this applies the action and steps physics)
        obs, reward, done, info = env.step(action)

        state_after = get_state()

        entry = {
            'step': step,
            'state_before': state_before,
            'obs': full_obs,
            'action': action_list,
            'latent': [0.0] * 64,
            'state_after': state_after,
        }
        trace.append(entry)

        rz_before = state_before['root_pos'][2]
        rz_after = state_after['root_pos'][2]
        print(f"Step {step}: root_z {rz_before:.4f} -> {rz_after:.4f} (delta={rz_after-rz_before:.4f})")

output = {'num_steps': NUM_STEPS, 'steps': trace}
with open('web/detailed_trace.json', 'w') as f:
    json.dump(output, f)
print(f"\nSaved {NUM_STEPS} steps to web/detailed_trace.json")

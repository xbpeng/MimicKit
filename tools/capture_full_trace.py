"""Capture full observation, action, and state trace from Isaac Lab CPU.

Records ALL 158 observation dimensions, the full action vector, and
all physical state for the first N steps after reset with zero latent.

Output: web/full_trace.json
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
args._table['rand_reset'] = ['false']

mp_util.init(0, 1, 'cpu', None)
env = mimickit_run.build_env(args, 1, 'cpu', False)
agent = mimickit_run.build_agent(args, env, 'cpu')
agent.load('data/models/ase_humanoid_sword_shield_model.pt')
agent.eval()

from learning.base_agent import AgentMode
agent.set_mode(AgentMode.TEST)

# Zero latent for reproducibility
agent._latent_buf = torch.zeros(1, 64)

NUM_STEPS = 5

trace = []
with torch.no_grad():
    obs, info = agent._reset_envs()

    for step in range(NUM_STEPS + 1):
        char_id = 0
        # Full physical state
        root_pos = env._engine.get_root_pos(char_id)[0].cpu().numpy()
        root_rot = env._engine.get_root_rot(char_id)[0].cpu().numpy()  # [x,y,z,w]
        root_vel = env._engine.get_root_vel(char_id)[0].cpu().numpy()
        root_ang_vel = env._engine.get_root_ang_vel(char_id)[0].cpu().numpy()
        dof_pos = env._engine.get_dof_pos(char_id)[0].cpu().numpy()
        dof_vel = env._engine.get_dof_vel(char_id)[0].cpu().numpy()

        # Full observation vector (158 dims)
        full_obs = obs[0].cpu().numpy()

        entry = {
            'step': step,
            'root_pos': root_pos.tolist(),
            'root_rot': root_rot.tolist(),
            'root_vel': root_vel.tolist(),
            'root_ang_vel': root_ang_vel.tolist(),
            'dof_pos': dof_pos.tolist(),
            'dof_vel': dof_vel.tolist(),
            'obs': full_obs.tolist(),
            'latent': agent._latent_buf[0].cpu().numpy().tolist(),
        }

        if step < NUM_STEPS:
            action, a_info = agent._decide_action(obs, info)
            entry['action'] = action[0].cpu().numpy().tolist()
            obs, reward, done, info = env.step(action)

        trace.append(entry)
        print(f"Step {step}: root_z={root_pos[2]:.4f} obs[:5]={full_obs[:5].round(4).tolist()}")

# Also capture body names, DOF ordering info
meta = {
    'body_names': env._engine.get_obj_body_names(0),
    'obs_dim': int(full_obs.shape[0]),
    'act_dim': int(dof_pos.shape[0]),
    'dof_count': int(dof_pos.shape[0]),
}

output = {'meta': meta, 'steps': trace}
with open('web/full_trace.json', 'w') as f:
    json.dump(output, f)
print(f"\nSaved {len(trace)} steps to web/full_trace.json")
print(f"obs_dim={meta['obs_dim']} act_dim={meta['act_dim']}")

"""Capture raw states + obs from ONNX running in Isaac Lab.

For each step, saves the full physical state AND the obs computed by Python.
The web demo can then feed these states into its JS obs computation and compare.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mimickit'))

import json
import numpy as np
import torch
import onnxruntime as ort

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

onnx_sess = ort.InferenceSession('web/ase_humanoid_sword_shield_actor.onnx')

NUM_STEPS = 120
char_id = 0

trace = []
with torch.no_grad():
    obs, info = agent._reset_envs()
    # Use a random latent (like the real demo does) — samples a skill
    # The agent already sampled one during _reset_envs, keep it
    latent = agent._latent_buf.cpu().numpy().astype(np.float32)
    print(f"Latent: {latent[0,:4].round(4)}... (norm={np.linalg.norm(latent):.4f})")

    for step in range(NUM_STEPS):
        # Get raw state
        root_pos = env._engine.get_root_pos(char_id)[0].cpu().numpy()
        root_rot = env._engine.get_root_rot(char_id)[0].cpu().numpy()
        root_vel = env._engine.get_root_vel(char_id)[0].cpu().numpy()
        root_ang_vel = env._engine.get_root_ang_vel(char_id)[0].cpu().numpy()
        dof_pos = env._engine.get_dof_pos(char_id)[0].cpu().numpy()
        dof_vel = env._engine.get_dof_vel(char_id)[0].cpu().numpy()

        # Get body positions AND rotations (PhysX link poses)
        body_pos = env._engine.get_body_pos(char_id)[0].cpu().numpy()
        body_rot = env._engine.get_body_rot(char_id)[0].cpu().numpy()

        # Get the obs that Python computed
        full_obs = obs[0].cpu().numpy()

        # Get ONNX action
        action = onnx_sess.run(['action'], {
            'obs': full_obs.reshape(1, -1).astype(np.float32),
            'latent': latent
        })[0][0]

        entry = {
            'step': step,
            'root_pos': root_pos.tolist(),
            'root_rot': root_rot.tolist(),
            'root_vel': root_vel.tolist(),
            'root_ang_vel': root_ang_vel.tolist(),
            'dof_pos': dof_pos.tolist(),
            'dof_vel': dof_vel.tolist(),
            'body_pos': body_pos.tolist(),
            'body_rot': body_rot.tolist(),
            'obs': full_obs.tolist(),
            'action': action.tolist(),
            'latent': latent[0].tolist(),
        }
        trace.append(entry)

        # Step env
        action_tensor = torch.tensor(action, device='cpu', dtype=torch.float32).unsqueeze(0)
        obs, reward, done, info = env.step(action_tensor)

        rz = root_pos[2]
        print(f"Step {step}: root_z={rz:.4f}")

with open('web/obs_trace.json', 'w') as f:
    json.dump(trace, f)
print(f"Saved {len(trace)} steps to web/obs_trace.json")

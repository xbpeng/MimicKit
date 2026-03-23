"""Capture closed-loop trace: ONNX policy from init_pose with zero latent.

This matches what the web demo does: start from init_pose, run ONNX model,
apply actions, step physics. We can then directly compare obs and action
at each step.
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

# Load ONNX model (same as web demo)
onnx_sess = ort.InferenceSession('web/ase_humanoid_sword_shield_actor.onnx')

from learning.base_agent import AgentMode
agent.set_mode(AgentMode.TEST)

NUM_STEPS = 10

def get_state():
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

    # Force init_pose with zero velocities
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
    obs = env._compute_obs()

    # Zero latent
    latent = np.zeros((1, 64), dtype=np.float32)

    for step in range(NUM_STEPS):
        state = get_state()
        full_obs = obs[0].cpu().numpy()

        # Run ONNX model (same as web demo)
        onnx_action = onnx_sess.run(['action'], {
            'obs': full_obs.reshape(1, -1).astype(np.float32),
            'latent': latent
        })[0][0]

        entry = {
            'step': step,
            'state': state,
            'obs': full_obs.tolist(),
            'action': onnx_action.tolist(),
            'latent': latent[0].tolist(),
        }
        trace.append(entry)

        # Apply action through the env (which clips to bounds)
        action_tensor = torch.tensor(onnx_action, device='cpu', dtype=torch.float32).unsqueeze(0)
        obs, reward, done, info = env.step(action_tensor)

        rz = state['root_pos'][2]
        print(f"Step {step}: root_z={rz:.4f} obs[0]={full_obs[0]:.4f} action[:3]={onnx_action[:3].round(4).tolist()}")

output = {'num_steps': NUM_STEPS, 'steps': trace}
with open('web/closedloop_trace.json', 'w') as f:
    json.dump(output, f)
print(f"\nSaved {NUM_STEPS} steps to web/closedloop_trace.json")

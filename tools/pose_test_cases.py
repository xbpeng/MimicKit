"""Generate pose test cases: set specific DOF configs, capture full obs + action.

Creates a set of diagnostic poses (T-pose, init pose, single-joint rotations)
and records the full state, observation, and ONNX action for each.
Run in Isaac Lab CPU, then compare with web demo.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mimickit'))

import json
import numpy as np
import torch
import onnxruntime as ort

from util.arg_parser import ArgParser
import util.mp_util as mp_util
import util.torch_util as torch_util
import run

args = ArgParser()
args.load_file('args/ase_humanoid_sword_shield_args.txt')
args._table['engine_config'] = ['data/engines/isaac_lab_engine.yaml']
args._table['num_envs'] = ['1']
args._table['mode'] = ['test']
args._table['visualize'] = ['false']
mp_util.init(0, 1, 'cpu', None)
env = run.build_env(args, 1, 'cpu', False)
agent = run.build_agent(args, env, 'cpu')
agent.load('data/models/ase_humanoid_sword_shield_model.pt')
agent.eval()

onnx_sess = ort.InferenceSession('web/ase_humanoid_sword_shield_actor.onnx')
latent = np.zeros((1, 64), dtype=np.float32)

char_id = 0
init_root_pos = env._init_root_pos
init_root_rot = env._init_root_rot
init_dof_pos = env._init_dof_pos
num_dofs = init_dof_pos.shape[0]

def set_pose(root_pos, root_rot, dof_pos):
    """Set character to a specific pose and return full state + obs + action."""
    env._engine.set_root_pos([0], char_id, root_pos.unsqueeze(0))
    env._engine.set_root_rot([0], char_id, root_rot.unsqueeze(0))
    env._engine.set_root_vel([0], char_id, 0.0)
    env._engine.set_root_ang_vel([0], char_id, 0.0)
    env._engine.set_dof_pos([0], char_id, dof_pos.unsqueeze(0))
    env._engine.set_dof_vel([0], char_id, 0.0)

    obs = env._compute_obs()
    full_obs = obs[0].cpu().numpy()

    # ONNX action
    action = onnx_sess.run(['action'], {
        'obs': full_obs.reshape(1, -1).astype(np.float32),
        'latent': latent
    })[0][0]

    # Physical state
    rp = env._engine.get_root_pos(char_id)[0].cpu().numpy()
    rr = env._engine.get_root_rot(char_id)[0].cpu().numpy()
    dp = env._engine.get_dof_pos(char_id)[0].cpu().numpy()
    bp = env._engine.get_body_pos(char_id)[0].cpu().numpy()

    return {
        'root_pos': rp.tolist(),
        'root_rot': rr.tolist(),
        'dof_pos': dp.tolist(),
        'obs': full_obs.tolist(),
        'action': action.tolist(),
        'body_pos': bp.tolist(),
    }

# Reset once to initialize
env.reset()
agent._latent_buf = torch.zeros(1, 64)

test_cases = []

# Test 1: Init pose (combat stance)
print("Test 1: Init pose")
result = set_pose(init_root_pos, init_root_rot, init_dof_pos)
test_cases.append({'name': 'init_pose', **result})

# Test 2: T-pose (all DOFs = 0)
print("Test 2: T-pose (zero DOFs)")
zero_dof = torch.zeros(num_dofs)
# Use upright root rotation (identity-ish, facing forward)
upright_rot = torch.tensor([0.0, 0.0, 0.0, 1.0])
upright_pos = torch.tensor([0.0, 0.0, 0.9])  # Slightly higher for T-pose
result = set_pose(upright_pos, upright_rot, zero_dof)
test_cases.append({'name': 't_pose', **result})

# Test 3: Init pose with right arm raised (DOF 6 = right_shoulder_x = 1.5)
print("Test 3: Right arm raised")
arm_dof = init_dof_pos.clone()
arm_dof[6] = 1.5  # right_shoulder_x
result = set_pose(init_root_pos, init_root_rot, arm_dof)
test_cases.append({'name': 'right_arm_raised', **result})

# Test 4: Init pose with both knees bent (DOF 20, 27)
print("Test 4: Knees bent")
knee_dof = init_dof_pos.clone()
knee_dof[20] = 2.0  # right_knee
knee_dof[27] = 2.0  # left_knee
result = set_pose(init_root_pos, init_root_rot, knee_dof)
test_cases.append({'name': 'knees_bent', **result})

# Test 5: Init pose with character tilted 30 degrees forward
print("Test 5: Tilted forward 30deg")
tilt_angle = 0.5236  # 30 degrees
tilt_rot = torch_util.axis_angle_to_quat(
    torch.tensor([[0.0, 1.0, 0.0]]),
    torch.tensor([tilt_angle])
)[0]
tilted_root_rot = torch_util.quat_mul(tilt_rot.unsqueeze(0), init_root_rot.unsqueeze(0))[0]
result = set_pose(init_root_pos, tilted_root_rot, init_dof_pos)
test_cases.append({'name': 'tilted_forward_30deg', **result})

# Also step physics once from init_pose and record
print("Test 6: Init pose after 1 physics step")
set_pose(init_root_pos, init_root_rot, init_dof_pos)
# Apply the init_pose action as drive targets
action_t = torch.tensor(test_cases[0]['action']).unsqueeze(0)
obs, reward, done, info = env.step(action_t)
rp = env._engine.get_root_pos(char_id)[0].cpu().numpy()
rr = env._engine.get_root_rot(char_id)[0].cpu().numpy()
dp = env._engine.get_dof_pos(char_id)[0].cpu().numpy()
bp = env._engine.get_body_pos(char_id)[0].cpu().numpy()
full_obs = obs[0].cpu().numpy()
action2 = onnx_sess.run(['action'], {
    'obs': full_obs.reshape(1, -1).astype(np.float32),
    'latent': latent
})[0][0]
test_cases.append({
    'name': 'init_pose_after_1step',
    'root_pos': rp.tolist(),
    'root_rot': rr.tolist(),
    'dof_pos': dp.tolist(),
    'obs': full_obs.tolist(),
    'action': action2.tolist(),
    'body_pos': bp.tolist(),
})

output = {
    'num_dofs': num_dofs,
    'body_names': env._engine.get_obj_body_names(char_id),
    'key_body_ids': env._key_body_ids.cpu().numpy().tolist(),
    'test_cases': test_cases,
}

with open('web/pose_tests.json', 'w') as f:
    json.dump(output, f)

print(f"\nSaved {len(test_cases)} test cases to web/pose_tests.json")
for tc in test_cases:
    print(f"  {tc['name']}: obs[:3]={[round(v,4) for v in tc['obs'][:3]]} action[:3]={[round(v,4) for v in tc['action'][:3]]}")

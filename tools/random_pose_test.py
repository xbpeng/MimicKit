"""Generate random pose test: random root pos/rot/vel, random DOF pos/vel.

Sets a fully random state and captures the observation and action.
This tests every obs dimension with non-trivial values.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mimickit'))

import json
import numpy as np
import torch
import onnxruntime as ort

from util.arg_parser import ArgParser
import util.mp_util as mp_util
import run

np.random.seed(42)
torch.manual_seed(42)

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

char_id = 0
init_dof = env._init_dof_pos
num_dofs = init_dof.shape[0]

# Get DOF limits
dof_low, dof_high = env._engine.get_obj_dof_limits(0, char_id)
dof_low = np.array(dof_low).flatten()
dof_high = np.array(dof_high).flatten()

env.reset()
agent._latent_buf = torch.zeros(1, 64)

# Generate random state
# Root position: near standing height
root_pos = torch.tensor([
    np.random.uniform(-0.1, 0.1),
    np.random.uniform(-0.1, 0.1),
    np.random.uniform(0.6, 0.8)
])

# Root rotation: small random rotation from init
init_rot = env._init_root_rot.numpy()
# Add small random perturbation
angle = np.random.uniform(0, 0.3)
axis = np.random.randn(3)
axis = axis / np.linalg.norm(axis)
s = np.sin(angle / 2)
c = np.cos(angle / 2)
perturb_q = np.array([axis[0]*s, axis[1]*s, axis[2]*s, c])
# Multiply: perturb * init
def qm(a, b):
    ax,ay,az,aw = a
    bx,by,bz,bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz
    ])
root_rot = torch.tensor(qm(perturb_q, init_rot), dtype=torch.float32)

# Root velocity: small random
root_vel = torch.tensor(np.random.uniform(-0.5, 0.5, 3), dtype=torch.float32)
root_ang_vel = torch.tensor(np.random.uniform(-1.0, 1.0, 3), dtype=torch.float32)

# DOF positions: random within limits, biased toward init
dof_pos = np.zeros(num_dofs)
for i in range(num_dofs):
    lo = dof_low[i]
    hi = dof_high[i]
    mid = init_dof[i].item()
    # Random within 50% of range from init
    range_half = (hi - lo) * 0.25
    dof_pos[i] = np.clip(mid + np.random.uniform(-range_half, range_half), lo, hi)
dof_pos = torch.tensor(dof_pos, dtype=torch.float32)

# DOF velocities: small random
dof_vel = torch.tensor(np.random.uniform(-2.0, 2.0, num_dofs), dtype=torch.float32)

# Apply state
env._engine.set_root_pos([0], char_id, root_pos.unsqueeze(0))
env._engine.set_root_rot([0], char_id, root_rot.unsqueeze(0))
env._engine.set_root_vel([0], char_id, root_vel.unsqueeze(0))
env._engine.set_root_ang_vel([0], char_id, root_ang_vel.unsqueeze(0))
env._engine.set_dof_pos([0], char_id, dof_pos.unsqueeze(0))
env._engine.set_dof_vel([0], char_id, dof_vel.unsqueeze(0))

# Compute obs
obs = env._compute_obs()
full_obs = obs[0].cpu().numpy()

# ONNX action
latent = np.zeros((1, 64), dtype=np.float32)
action = onnx_sess.run(['action'], {
    'obs': full_obs.reshape(1, -1).astype(np.float32),
    'latent': latent
})[0][0]

# Readback actual state (may differ from what we set due to engine internals)
actual_root_pos = env._engine.get_root_pos(char_id)[0].cpu().numpy()
actual_root_rot = env._engine.get_root_rot(char_id)[0].cpu().numpy()
actual_root_vel = env._engine.get_root_vel(char_id)[0].cpu().numpy()
actual_root_ang_vel = env._engine.get_root_ang_vel(char_id)[0].cpu().numpy()
actual_dof_pos = env._engine.get_dof_pos(char_id)[0].cpu().numpy()
actual_dof_vel = env._engine.get_dof_vel(char_id)[0].cpu().numpy()

result = {
    'root_pos': actual_root_pos.tolist(),
    'root_rot': actual_root_rot.tolist(),
    'root_vel': actual_root_vel.tolist(),
    'root_ang_vel': actual_root_ang_vel.tolist(),
    'dof_pos': actual_dof_pos.tolist(),
    'dof_vel': actual_dof_vel.tolist(),
    'obs': full_obs.tolist(),
    'action': action.tolist(),
    'latent': latent[0].tolist(),
}

with open('web/random_pose_test.json', 'w') as f:
    json.dump(result, f)

print("Random pose test saved to web/random_pose_test.json")
print(f"root_pos: {actual_root_pos.round(4).tolist()}")
print(f"root_rot: {actual_root_rot.round(4).tolist()}")
print(f"root_vel: {actual_root_vel.round(4).tolist()}")
print(f"root_ang_vel: {actual_root_ang_vel.round(4).tolist()}")
print(f"dof_pos[:5]: {actual_dof_pos[:5].round(4).tolist()}")
print(f"dof_vel[:5]: {actual_dof_vel[:5].round(4).tolist()}")
print(f"obs[:13]: {full_obs[:13].round(4).tolist()}")
print(f"action[:5]: {action[:5].round(4).tolist()}")

# Verify obs has non-trivial values in all categories
print(f"\nObs non-zero check:")
print(f"  height (0): {full_obs[0]:.4f}")
print(f"  rot_tn max (1-6): {np.abs(full_obs[1:7]).max():.4f}")
print(f"  vel max (7-9): {np.abs(full_obs[7:10]).max():.4f}")
print(f"  angvel max (10-12): {np.abs(full_obs[10:13]).max():.4f}")
print(f"  jrot max (13-108): {np.abs(full_obs[13:109]).max():.4f}")
print(f"  dofvel max (109-139): {np.abs(full_obs[109:140]).max():.4f}")
print(f"  keypos max (140-157): {np.abs(full_obs[140:158]).max():.4f}")

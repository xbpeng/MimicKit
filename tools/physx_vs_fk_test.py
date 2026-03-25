"""Definitive test: PhysX link positions vs FK positions for key bodies.

Sets a non-init pose via the engine's proper reset path (write_to_sim + step),
then compares PhysX link positions vs FK positions.

This resolves: does the policy see PhysX link pos or FK pos?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mimickit'))

import json
import numpy as np
import torch

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

char_id = 0
init_dof = env._init_dof_pos
num_dofs = init_dof.shape[0]

dof_low, dof_high = env._engine.get_obj_dof_limits(0, char_id)
dof_low = np.array(dof_low).flatten()
dof_high = np.array(dof_high).flatten()

env.reset()

body_names = env._engine.get_obj_body_names(char_id)
key_body_ids = env._key_body_ids
key_names = [body_names[i] for i in key_body_ids.tolist()]
print(f"Key body names: {key_names}")
print(f"All body names: {body_names}")

from envs.char_env import compute_char_obs

def compare_at_pose(label, root_pos_t, root_rot_t, dof_pos_t):
    """Set state, step to propagate, then compare PhysX vs FK."""
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")

    # Set state via engine (this writes to data buffers + flags reset)
    env._engine.set_root_pos([0], char_id, root_pos_t.unsqueeze(0))
    env._engine.set_root_rot([0], char_id, root_rot_t.unsqueeze(0))
    env._engine.set_root_vel([0], char_id, torch.zeros(1, 3))
    env._engine.set_root_ang_vel([0], char_id, torch.zeros(1, 3))
    env._engine.set_dof_pos([0], char_id, dof_pos_t.unsqueeze(0))
    env._engine.set_dof_vel([0], char_id, torch.zeros(1, num_dofs))
    env._engine.set_body_vel([0], char_id, 0.0)
    env._engine.set_body_ang_vel([0], char_id, 0.0)

    # Step engine to propagate state to PhysX links
    # This calls _update_reset_objs() which writes to sim, then sim_step()
    env._engine.step()

    # Now read back everything
    root_pos_rb = env._engine.get_root_pos(char_id)[0:1]
    root_rot_rb = env._engine.get_root_rot(char_id)[0:1]
    dof_pos_rb = env._engine.get_dof_pos(char_id)[0:1]

    # PhysX link positions (what the env actually uses for obs)
    body_pos_physx = env._engine.get_body_pos(char_id)[0]
    key_pos_physx = body_pos_physx[key_body_ids].cpu().numpy()

    # FK positions
    joint_rot = env._kin_char_model.dof_to_rot(dof_pos_rb)
    fk_body_pos, _ = env._kin_char_model.forward_kinematics(root_pos_rb, root_rot_rb, joint_rot)
    key_pos_fk = fk_body_pos[0, key_body_ids].cpu().numpy()

    print(f"Root pos readback: {root_pos_rb[0].cpu().numpy().round(4).tolist()}")
    print(f"DOF set vs readback max: {(dof_pos_t - dof_pos_rb[0].cpu()).abs().max().item():.6f}")

    print(f"\n{'Body':<20} {'PhysX':<45} {'FK':<45} {'|diff|'}")
    for i, bid in enumerate(key_body_ids.tolist()):
        px = key_pos_physx[i]
        fk = key_pos_fk[i]
        d = np.abs(px - fk).max()
        print(f"  {body_names[bid]:<18} {str(px.round(4).tolist()):<43} {str(fk.round(4).tolist()):<43} {d:.6f}")

    diff = np.abs(key_pos_physx - key_pos_fk)
    print(f"  OVERALL MAX DIFF: {diff.max():.6f}")

    # Which matches the obs?
    obs = env._compute_obs()
    obs_kp = obs[0, 140:158].cpu().numpy()

    # Reconstruct obs with PhysX key_pos
    root_vel_rb = env._engine.get_root_vel(char_id)[0:1]
    root_ang_vel_rb = env._engine.get_root_ang_vel(char_id)[0:1]
    dof_vel_rb = env._engine.get_dof_vel(char_id)[0:1]

    obs_from_physx = compute_char_obs(
        root_pos=root_pos_rb, root_rot=root_rot_rb,
        root_vel=root_vel_rb, root_ang_vel=root_ang_vel_rb,
        joint_rot=joint_rot, dof_vel=dof_vel_rb,
        key_pos=body_pos_physx[key_body_ids].unsqueeze(0).to(root_pos_rb.device),
        global_obs=env._global_obs, root_height_obs=env._root_height_obs
    )
    obs_from_fk = compute_char_obs(
        root_pos=root_pos_rb, root_rot=root_rot_rb,
        root_vel=root_vel_rb, root_ang_vel=root_ang_vel_rb,
        joint_rot=joint_rot, dof_vel=dof_vel_rb,
        key_pos=fk_body_pos[0:1, key_body_ids].to(root_pos_rb.device),
        global_obs=env._global_obs, root_height_obs=env._root_height_obs
    )

    err_physx = np.abs(obs_kp - obs_from_physx[0, 140:158].cpu().numpy()).max()
    err_fk = np.abs(obs_kp - obs_from_fk[0, 140:158].cpu().numpy()).max()
    print(f"\n  Obs match with PhysX key_pos: {err_physx:.6f}")
    print(f"  Obs match with FK key_pos:    {err_fk:.6f}")

    return {
        'key_pos_physx': key_pos_physx.tolist(),
        'key_pos_fk': key_pos_fk.tolist(),
        'max_diff': float(diff.max()),
        'obs_key_pos': obs_kp.tolist(),
        'root_pos': root_pos_rb[0].cpu().numpy().tolist(),
        'root_rot': root_rot_rb[0].cpu().numpy().tolist(),
        'dof_pos': dof_pos_rb[0].cpu().numpy().tolist(),
    }

# TEST 1: Init pose
r1 = compare_at_pose(
    "TEST 1: INIT POSE",
    env._init_root_pos.clone(),
    env._init_root_rot.clone(),
    init_dof.clone(),
)

# TEST 2: Random pose
dof_pos_rand = torch.zeros(num_dofs)
for i in range(num_dofs):
    lo = dof_low[i]
    hi = dof_high[i]
    mid = init_dof[i].item()
    range_half = (hi - lo) * 0.25
    dof_pos_rand[i] = np.clip(mid + np.random.uniform(-range_half, range_half), lo, hi)

r2 = compare_at_pose(
    "TEST 2: RANDOM POSE (25% range from init)",
    torch.tensor([0.0, 0.0, 0.89]),
    env._init_root_rot.clone(),
    dof_pos_rand,
)

# TEST 3: Extreme pose
dof_pos_extreme = torch.zeros(num_dofs)
for i in range(num_dofs):
    lo = dof_low[i]
    hi = dof_high[i]
    mid = init_dof[i].item()
    range_half = (hi - lo) * 0.5
    dof_pos_extreme[i] = np.clip(mid + np.random.uniform(-range_half, range_half), lo, hi)

r3 = compare_at_pose(
    "TEST 3: EXTREME POSE (50% range from init)",
    torch.tensor([0.0, 0.0, 0.89]),
    env._init_root_rot.clone(),
    dof_pos_extreme,
)

# Summary
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Init pose PhysX-FK diff:    {r1['max_diff']:.6f}")
print(f"Random pose PhysX-FK diff:  {r2['max_diff']:.6f}")
print(f"Extreme pose PhysX-FK diff: {r3['max_diff']:.6f}")

if r2['max_diff'] < 0.001:
    print("\n>>> PhysX and FK AGREE at non-init poses <<<")
    print(">>> The web demo should use FK (they're equivalent) <<<")
else:
    print(f"\n>>> PhysX and FK DIVERGE at non-init poses (max {r2['max_diff']:.4f}) <<<")
    print(">>> The web demo MUST use PhysX link positions, not FK <<<")
    print(">>> The web bug is that PhysX links in WASM don't match Isaac Lab links <<<")

# Save for web
result = {
    'init_pose': r1,
    'random_pose': r2,
    'extreme_pose': r3,
    'body_names': body_names,
    'key_body_ids': key_body_ids.tolist(),
    'key_body_names': key_names,
}
with open('web/physx_vs_fk_test.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to web/physx_vs_fk_test.json")

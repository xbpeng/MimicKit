"""Dump all PhysX parameters for comparison with web demo.

Outputs: mass, inertia, COM, drive params, solver iters, etc.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mimickit'))

import numpy as np
import torch
from util.arg_parser import ArgParser
import util.mp_util as mp_util
import run

args = ArgParser()
args.load_file('args/ase_humanoid_sword_shield_args.txt')
args._table['engine_config'] = ['data/engines/isaac_lab_engine.yaml']
args._table['num_envs'] = ['1']
args._table['mode'] = ['test']
args._table['visualize'] = ['false']
mp_util.init(0, 1, 'cpu', None)
env = run.build_env(args, 1, 'cpu', False)

char_id = 0
obj = env._engine._objs[char_id]
body_names = env._engine.get_obj_body_names(char_id)

# Get articulation properties
view = obj.root_physx_view
meta_data = view.shared_metatype

print("=" * 70)
print("ARTICULATION PROPERTIES")
print("=" * 70)
print(f"Solver position iterations: {view.solver_position_iteration_counts[0].item()}")
print(f"Solver velocity iterations: {view.solver_velocity_iteration_counts[0].item()}")
print(f"Sleep threshold: {view.sleep_thresholds[0].item()}")

# Per-body properties
print(f"\n{'='*70}")
print(f"PER-BODY PROPERTIES (common order)")
print(f"{'='*70}")

body_sim2common = env._engine._body_order_sim2common[char_id]
sim_link_names = meta_data.link_names

# Get mass, inertia, COM from PhysX view
body_masses = view.get_body_masses().cpu().numpy()[0]  # [num_bodies]
body_inertias = view.get_body_inv_inertias().cpu().numpy()[0]  # [num_bodies, 9]
body_coms = view.get_body_com_transforms().cpu().numpy()[0]  # [num_bodies, 7] (pos + quat)

result = {'bodies': []}

for ci, si in enumerate(body_sim2common.tolist()):
    name = sim_link_names[si]
    mass = body_masses[si]
    inertia_inv = body_inertias[si]  # 3x3 flattened
    com = body_coms[si]  # [px, py, pz, qw, qx, qy, qz]

    print(f"\n{name} (common idx {ci}, sim idx {si}):")
    print(f"  mass: {mass:.6f}")
    print(f"  COM pos: [{com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f}]")
    print(f"  COM rot: [{com[3]:.6f}, {com[4]:.6f}, {com[5]:.6f}, {com[6]:.6f}]")
    # Inverse inertia to inertia
    inv_mat = inertia_inv.reshape(3, 3)
    diag_inv = np.diag(inv_mat)
    diag = np.where(diag_inv > 0, 1.0 / diag_inv, 0)
    print(f"  inertia diag: [{diag[0]:.6f}, {diag[1]:.6f}, {diag[2]:.6f}]")

    result['bodies'].append({
        'name': name,
        'common_idx': ci,
        'sim_idx': si,
        'mass': float(mass),
        'com_pos': com[:3].tolist(),
        'com_rot': com[3:].tolist(),
        'inertia_diag': diag.tolist(),
    })

# Drive parameters
print(f"\n{'='*70}")
print(f"DRIVE PARAMETERS")
print(f"{'='*70}")

max_dofs = view.max_dof_count
dof_stiffnesses = view.get_dof_stiffnesses().cpu().numpy()[0]
dof_dampings = view.get_dof_dampings().cpu().numpy()[0]
dof_max_forces = view.get_dof_max_forces().cpu().numpy()[0]
dof_armatures = view.get_dof_armatures().cpu().numpy()[0]

dof_sim2common = env._engine._dof_order_sim2common[char_id]

result['drives'] = []
for ci, si in enumerate(dof_sim2common.tolist()):
    print(f"  DOF {ci} (sim {si}): stiff={dof_stiffnesses[si]:.1f} damp={dof_dampings[si]:.1f} maxF={dof_max_forces[si]:.1f} armature={dof_armatures[si]:.6f}")
    result['drives'].append({
        'common_idx': ci,
        'sim_idx': si,
        'stiffness': float(dof_stiffnesses[si]),
        'damping': float(dof_dampings[si]),
        'max_force': float(dof_max_forces[si]),
        'armature': float(dof_armatures[si]),
    })

# Scene parameters
print(f"\n{'='*70}")
print(f"SCENE PARAMETERS")
print(f"{'='*70}")
print(f"Gravity: {env._engine._sim.cfg.gravity}")
print(f"DT: {env._engine._sim.cfg.dt}")
print(f"Substeps: {env._engine._sim.cfg.substeps}")

with open('web/physx_params.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to web/physx_params.json")

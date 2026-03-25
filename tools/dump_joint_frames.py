"""Dump PhysX articulation joint frames from Isaac Lab for comparison with web.

For each joint, prints the parent pose and child pose as seen by PhysX.
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
body_names = env._engine.get_obj_body_names(char_id)

# Get joint frames from PhysX view
obj = env._engine._objs[char_id]
view = obj.root_physx_view
meta = view.shared_metatype

# Joint properties - these are in sim order
joint_names = meta.joint_names
link_names = meta.link_names

print(f"Sim link order: {link_names}")
print(f"Common body order: {body_names}")
print(f"Joint names (sim order): {joint_names}")

# Body order mappings
s2c = env._engine._body_order_sim2common[char_id].tolist()
print(f"body_order_sim2common: {s2c}")

# Get joint parent/child poses from the USD
# We need to read these from the articulation directly
# Let's use the PhysX tensor API to get joint local poses
# Actually, let's read from the USD stage directly

sim = env._engine._sim
stage = sim.stage

# Find articulation prim
art_prim_path = "/World/envs/env_0/obj_0/robot"

from pxr import UsdPhysics, Usd, Gf

result = {'joints': []}

for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RevoluteJointAPI) or prim.HasAPI(UsdPhysics.SphericalJointAPI) or prim.HasAPI(UsdPhysics.Joint):
        path = str(prim.GetPath())
        if 'env_0' not in path:
            continue

        joint_api = UsdPhysics.Joint(prim)

        # Get local poses
        lr0 = prim.GetAttribute('physics:localRot0').Get()
        lr1 = prim.GetAttribute('physics:localRot1').Get()
        lp0 = prim.GetAttribute('physics:localPos0').Get()
        lp1 = prim.GetAttribute('physics:localPos1').Get()

        name = prim.GetName()

        entry = {
            'name': name,
            'path': path,
        }
        if lp0: entry['localPos0'] = list(lp0)
        if lp1: entry['localPos1'] = list(lp1)
        if lr0: entry['localRot0_wxyz'] = [lr0.GetReal()] + list(lr0.GetImaginary())
        if lr1: entry['localRot1_wxyz'] = [lr1.GetReal()] + list(lr1.GetImaginary())

        result['joints'].append(entry)

        lr0_str = f"[{lr0.GetReal():.4f}, {list(lr0.GetImaginary())}]" if lr0 else "None"
        lr1_str = f"[{lr1.GetReal():.4f}, {list(lr1.GetImaginary())}]" if lr1 else "None"
        lp0_str = f"{list(lp0)}" if lp0 else "None"
        lp1_str = f"{list(lp1)}" if lp1 else "None"

        print(f"\n{name}:")
        print(f"  localPos0={lp0_str}  localPos1={lp1_str}")
        print(f"  localRot0={lr0_str}")
        print(f"  localRot1={lr1_str}")

# Now compare with humanoid_data.json
print("\n" + "="*70)
print("COMPARISON WITH humanoid_data.json")
print("="*70)

with open('web/humanoid_data.json') as f:
    hdata = json.load(f)

for jdata in hdata['joints']:
    child = jdata['child_body']
    lp0 = jdata['localPos0']
    lr = jdata['localRot']  # [w, x, y, z]

    # Find matching USD joint
    match = None
    for uj in result['joints']:
        # Match by child body name in the path
        if child.replace('_', '') in uj['name'].replace('_', '').lower() or child in uj['path']:
            match = uj
            break

    if match and 'localRot0_wxyz' in match:
        usd_lr = match['localRot0_wxyz']
        json_lr = lr

        # Compare rotations (both wxyz)
        rot_diff = max(abs(usd_lr[i] - json_lr[i]) for i in range(4))

        usd_lp = match.get('localPos0', [0,0,0])
        pos_diff = max(abs(usd_lp[i] - lp0[i]) for i in range(min(len(usd_lp), len(lp0))))

        flag = " *** MISMATCH ***" if rot_diff > 0.01 or pos_diff > 0.01 else ""
        print(f"\n{child}:{flag}")
        print(f"  JSON localPos0={[round(v,4) for v in lp0]}")
        print(f"  USD  localPos0={[round(v,4) for v in usd_lp]}")
        print(f"  pos_diff={pos_diff:.6f}")
        print(f"  JSON localRot={[round(v,4) for v in json_lr]}")
        print(f"  USD  localRot={[round(v,4) for v in usd_lr]}")
        print(f"  rot_diff={rot_diff:.6f}")
    else:
        print(f"\n{child}: no USD match found")

with open('web/joint_frames.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to web/joint_frames.json")

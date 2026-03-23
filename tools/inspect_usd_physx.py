"""Inspect USD-level PhysX parameters from Isaac Lab scene."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mimickit'))

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

from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdPhysics, PhysxSchema

stage = get_current_stage()

print("=== COLLISION SHAPES ===")
count = 0
for prim in stage.Traverse():
    if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        api = PhysxSchema.PhysxCollisionAPI(prim)
        co = api.GetContactOffsetAttr().Get()
        ro = api.GetRestOffsetAttr().Get()
        print(f"{prim.GetPath().pathString}: contactOffset={co} restOffset={ro}")
        count += 1
        if count > 20:
            print("... (truncated)")
            break

print("\n=== SCENE ===")
for prim in stage.Traverse():
    if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
        print(f"Scene: {prim.GetPath().pathString}")
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if 'physx' in name.lower() or 'physics' in name.lower():
                val = attr.Get()
                if val is not None:
                    print(f"  {name}: {val}")

print("\n=== ARTICULATION ROOT ===")
for prim in stage.Traverse():
    if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        print(f"Articulation: {prim.GetPath().pathString}")
        for attr in prim.GetAttributes():
            name = attr.GetName()
            val = attr.Get()
            if val is not None and ('physx' in name.lower() or 'physics' in name.lower()):
                print(f"  {name}: {val}")

print("\n=== JOINTS (first 3) ===")
count = 0
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.DriveAPI):
        if count < 3:
            print(f"Joint: {prim.GetPath().pathString}")
            for attr in prim.GetAttributes():
                name = attr.GetName()
                val = attr.Get()
                if val is not None and ('drive' in name.lower() or 'physics' in name.lower() or 'physx' in name.lower()):
                    print(f"  {name}: {val}")
            count += 1

print("\n=== RIGID BODIES (first 3) ===")
count = 0
for prim in stage.Traverse():
    if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        if count < 3:
            api = PhysxSchema.PhysxRigidBodyAPI(prim)
            print(f"RigidBody: {prim.GetPath().pathString}")
            for attr in prim.GetAttributes():
                name = attr.GetName()
                val = attr.Get()
                if val is not None and ('physx' in name.lower()):
                    print(f"  {name}: {val}")
            count += 1

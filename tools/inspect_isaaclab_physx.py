"""Inspect all PhysX parameters from a running Isaac Lab scene.

Dumps articulation, link, joint, shape, and drive parameters so we can
compare with the web PhysX WASM setup and find configuration differences.

Usage:
    conda run -n mimickit-isaaclab python tools/inspect_isaaclab_physx.py [--cpu]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mimickit'))

import argparse
import json
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

device = 'cpu' if args.cpu else 'cuda:0'

from util.arg_parser import ArgParser
import util.mp_util as mp_util
import run as mimickit_run

mk_args = ArgParser()
mk_args.load_file('args/ase_humanoid_sword_shield_args.txt')
mk_args._table['engine_config'] = ['data/engines/isaac_lab_engine.yaml']
mk_args._table['num_envs'] = ['1']
mk_args._table['mode'] = ['test']
mk_args._table['visualize'] = ['false']
mk_args._table['rand_reset'] = ['false']

mp_util.init(0, 1, device, None)
env = mimickit_run.build_env(mk_args, 1, device, False)

# Access the Isaac Lab internals
engine = env._engine
obj = engine._objs[0]  # The articulation object

print("=" * 70)
print("ISAAC LAB PhysX SCENE INSPECTION")
print("=" * 70)

# --- Scene parameters ---
print("\n--- Scene Parameters ---")
sim = engine._sim
print(f"Device: {device}")
print(f"Timestep (control): {engine._timestep}")
print(f"Sim steps per control: {engine._sim_steps}")
print(f"Sim dt: {engine.get_sim_timestep()}")

# --- Articulation parameters ---
print("\n--- Articulation Parameters ---")
# Get properties via Isaac Lab APIs
art = obj.root_physx_view
if art is not None:
    # Try to get solver iteration counts
    try:
        solver_pos = art.get_solver_position_iteration_counts()
        solver_vel = art.get_solver_velocity_iteration_counts()
        print(f"Solver position iterations: {solver_pos[0].item() if hasattr(solver_pos[0], 'item') else solver_pos}")
        print(f"Solver velocity iterations: {solver_vel[0].item() if hasattr(solver_vel[0], 'item') else solver_vel}")
    except Exception as e:
        print(f"Solver iterations: error - {e}")

    # Articulation flags
    try:
        flags = art.get_enabled_self_collisions()
        print(f"Self-collision enabled: {flags}")
    except Exception as e:
        print(f"Self-collision: error - {e}")

# --- Rigid body (link) parameters ---
print("\n--- Rigid Body (Link) Parameters ---")
try:
    body_names = obj.body_names
    print(f"Number of bodies: {len(body_names)}")
    print(f"Body names: {body_names}")

    # Get mass properties
    masses = obj.root_physx_view.get_body_masses()
    print(f"\nMasses: {masses[0].cpu().numpy()}")

    # Get inertias
    try:
        inertias = obj.root_physx_view.get_body_inertias()
        print(f"Inertias shape: {inertias.shape}")
        for i, name in enumerate(body_names):
            print(f"  {name}: mass={masses[0,i].item():.4f} inertia={inertias[0,i].cpu().numpy()}")
    except Exception as e:
        print(f"Inertias: error - {e}")

    # Try to get damping, max velocity etc
    try:
        # These may be accessible through the articulation view
        props = obj.root_physx_view.get_body_properties()
        print(f"\nBody properties type: {type(props)}")
    except:
        pass

except Exception as e:
    print(f"Body params error: {e}")

# --- Joint parameters ---
print("\n--- Joint Parameters ---")
try:
    joint_names = obj.joint_names
    print(f"Number of joints: {len(joint_names)}")
    print(f"Joint names: {joint_names}")

    # Get joint stiffness and damping
    stiffness = obj.root_physx_view.get_dof_stiffnesses()
    damping = obj.root_physx_view.get_dof_dampings()
    max_forces = obj.root_physx_view.get_dof_max_forces()
    armatures = obj.root_physx_view.get_dof_armatures()

    print(f"\nPer-DOF properties:")
    dof_names = obj.joint_names  # DOF names
    for i in range(min(len(dof_names), stiffness.shape[1])):
        name = dof_names[i] if i < len(dof_names) else f"dof_{i}"
        print(f"  {name}: kp={stiffness[0,i].item():.1f} kd={damping[0,i].item():.1f} "
              f"maxF={max_forces[0,i].item():.1f} armature={armatures[0,i].item():.4f}")

    # Joint limits
    try:
        limits = obj.root_physx_view.get_dof_limits()
        print(f"\nJoint limits:")
        for i in range(min(len(dof_names), limits.shape[1])):
            name = dof_names[i] if i < len(dof_names) else f"dof_{i}"
            print(f"  {name}: [{limits[0,i,0].item():.4f}, {limits[0,i,1].item():.4f}]")
    except Exception as e:
        print(f"Joint limits error: {e}")

    # Friction
    try:
        frictions = obj.root_physx_view.get_dof_friction_coefficients()
        print(f"\nDOF friction: {frictions[0].cpu().numpy()}")
    except:
        print("DOF friction: not available")

    # Max velocities
    try:
        max_vel = obj.root_physx_view.get_dof_max_velocities()
        print(f"DOF max velocities: {max_vel[0].cpu().numpy()}")
    except:
        print("DOF max velocities: not available")

except Exception as e:
    print(f"Joint params error: {e}")

# --- Shape/collision parameters ---
print("\n--- Shape/Collision Parameters ---")
try:
    # Try to access contact offset and rest offset via USD
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdPhysics, PhysxSchema

    stage = get_current_stage()
    if stage:
        # Find articulation root prim
        art_prim_path = obj.cfg.prim_path
        print(f"Articulation prim path: {art_prim_path}")

        # Iterate over body prims to get collision shape params
        for body_name in body_names[:5]:  # Just check first 5
            # Find the body prim
            body_path = f"{art_prim_path}/{body_name}"
            prim = stage.GetPrimAtPath(body_path)
            if not prim.IsValid():
                # Try with different path patterns
                for p in stage.Traverse():
                    if p.GetName() == body_name:
                        prim = p
                        break

            if prim.IsValid():
                # Check for PhysX rigid body properties
                rb_api = PhysxSchema.PhysxRigidBodyAPI(prim)
                if rb_api:
                    try:
                        max_depen = rb_api.GetMaxDepenetrationVelocityAttr().Get()
                        ang_damp = rb_api.GetAngularDampingAttr().Get()
                        lin_damp = rb_api.GetLinearDampingAttr().Get()
                        max_lin_vel = rb_api.GetMaxLinearVelocityAttr().Get()
                        max_ang_vel = rb_api.GetMaxAngularVelocityAttr().Get()
                        sleep_thresh = rb_api.GetSleepThresholdAttr().Get()
                        print(f"\n  {body_name} rigid body:")
                        print(f"    maxDepenetration: {max_depen}")
                        print(f"    angularDamping: {ang_damp}")
                        print(f"    linearDamping: {lin_damp}")
                        print(f"    maxLinearVelocity: {max_lin_vel}")
                        print(f"    maxAngularVelocity: {max_ang_vel}")
                        print(f"    sleepThreshold: {sleep_thresh}")
                    except Exception as e:
                        print(f"  {body_name} rigid body props error: {e}")

                # Check collision shapes
                for child in prim.GetChildren():
                    if child.HasAPI(UsdPhysics.CollisionAPI):
                        collision = UsdPhysics.CollisionAPI(child)
                        print(f"  {body_name}/{child.GetName()} collision shape found")

                        # Get PhysX collision shape props
                        shape_api = PhysxSchema.PhysxCollisionAPI(child)
                        if shape_api:
                            try:
                                co = shape_api.GetContactOffsetAttr().Get()
                                ro = shape_api.GetRestOffsetAttr().Get()
                                print(f"    contactOffset: {co}")
                                print(f"    restOffset: {ro}")
                            except:
                                print(f"    contactOffset/restOffset: not set")
except Exception as e:
    print(f"Shape params error: {e}")

# --- Total mass ---
print("\n--- Summary ---")
try:
    total_mass = masses[0].sum().item()
    print(f"Total mass: {total_mass:.3f} kg")
except:
    pass

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

"""Replay Isaac Lab trace in ovphysx (CPU PhysX 5) and compare with web PhysX WASM.

This builds the exact same articulation as the web demo and applies the same
actions from the Isaac Lab trace, to determine if CPU PhysX matches GPU PhysX.
"""
import json
import numpy as np
import ovphysx as px

# Load humanoid data and trace
with open('web/humanoid_data.json') as f:
    hdata = json.load(f)
with open('web/isaac_trace.json') as f:
    trace = json.load(f)

# Create PhysX
foundation = px.create_foundation()
physics = px.create_physics(foundation)
scene_desc = px.PxSceneDesc(px.PxTolerancesScale())
scene_desc.gravity = px.PxVec3(0, 0, -9.81)
scene_desc.solver_type = px.PxSolverType.eTGS  # Match web demo
dispatcher = px.PxDefaultCpuDispatcherCreate(1)
scene_desc.cpu_dispatcher = dispatcher
scene_desc.filter_shader = px.PxDefaultSimulationFilterShader
scene_desc.bounce_threshold_velocity = 0.2
scene = physics.create_scene(scene_desc)

# Material
material = physics.create_material(1.0, 1.0, 0.0)

# Ground plane
ground_shape = physics.create_shape(px.PxBoxGeometry(50, 50, 0.5), material)
ground_pose = px.PxTransform(px.PxVec3(0, 0, -0.5), px.PxQuat(0, 0, 0, 1))
ground = physics.create_rigid_static(ground_pose)
ground.attach_shape(ground_shape)
scene.add_actor(ground)

# Create articulation
articulation = physics.create_articulation_reduced_coordinate()
articulation.set_solver_iteration_counts(4, 0)  # Match Isaac Lab
articulation.set_articulation_flag(px.PxArticulationFlag.eDISABLE_SELF_COLLISION, True)

body_link_map = {}
links = []
pelvis_z = hdata['pelvis_z']

for body in hdata['bodies']:
    wp = body['pos']
    pose = px.PxTransform(
        px.PxVec3(wp[0], wp[1], wp[2] + pelvis_z),
        px.PxQuat(0, 0, 0, 1)
    )
    parent = body_link_map.get(body.get('parent'))
    link = articulation.create_link(parent, pose)

    # Attach geom shapes
    for geom in body['geoms']:
        shape = None
        if geom['type'] == 'sphere':
            shape = physics.create_shape(px.PxSphereGeometry(geom['radius']), material)
            lp = geom['pos']
            shape.set_local_pose(px.PxTransform(px.PxVec3(lp[0], lp[1], lp[2]), px.PxQuat(0,0,0,1)))
        elif geom['type'] == 'capsule' and geom.get('fromto'):
            ft = geom['fromto']
            p0, p1 = np.array(ft[:3]), np.array(ft[3:])
            d = p1 - p0
            length = np.linalg.norm(d)
            half_h = max(length / 2, 0.001)
            shape = physics.create_shape(px.PxCapsuleGeometry(geom['radius'], half_h), material)
            mid = (p0 + p1) / 2
            direction = d / length if length > 0.001 else np.array([1,0,0])
            # Compute quaternion from [1,0,0] to direction
            ref = np.array([1.0, 0.0, 0.0])
            cross = np.cross(ref, direction)
            dot = np.dot(ref, direction)
            if np.linalg.norm(cross) < 1e-8:
                if dot > 0:
                    q = px.PxQuat(0, 0, 0, 1)
                else:
                    q = px.PxQuat(0, 0, 1, 0)
            else:
                w = 1 + dot
                q = px.PxQuat(cross[0], cross[1], cross[2], w)
                mag = np.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2 + w**2)
                q = px.PxQuat(cross[0]/mag, cross[1]/mag, cross[2]/mag, w/mag)
            shape.set_local_pose(px.PxTransform(px.PxVec3(mid[0], mid[1], mid[2]), q))
        elif geom['type'] == 'box':
            he = geom['halfExtents']
            shape = physics.create_shape(px.PxBoxGeometry(he[0], he[1], he[2]), material)
            lp = geom['pos']
            shape.set_local_pose(px.PxTransform(px.PxVec3(lp[0], lp[1], lp[2]), px.PxQuat(0,0,0,1)))

        if shape:
            link.attach_shape(shape)

    # Set mass properties
    if body.get('mass') and body.get('inertia') and body.get('com'):
        link.set_mass(body['mass'])
        link.set_mass_space_inertia_tensor(px.PxVec3(*body['inertia']))
        link.set_c_mass_local_pose(px.PxTransform(
            px.PxVec3(*body['com']), px.PxQuat(0,0,0,1)
        ))

    # Match Isaac Lab rigid body properties
    link.set_angular_damping(0.01)
    link.set_linear_damping(0.0)
    link.set_max_depenetration_velocity(10.0)
    link.set_max_linear_velocity(1000.0)
    link.set_max_angular_velocity(1000.0)

    body_link_map[body['name']] = link
    links.append({'name': body['name'], 'link': link})

# Configure joints
AXIS_MAP = {0: px.PxArticulationAxis.eTWIST, 1: px.PxArticulationAxis.eSWING1, 2: px.PxArticulationAxis.eSWING2}

for jdata in hdata['joints']:
    child_link = body_link_map[jdata['child_body']]
    joint = child_link.get_inbound_joint()

    if jdata['jointType'] == 'spherical':
        joint.set_joint_type(px.PxArticulationJointType.eSPHERICAL)
    else:
        joint.set_joint_type(px.PxArticulationJointType.eREVOLUTE)

    lp = jdata['localPos0']
    lr = jdata['localRot']  # [w, x, y, z]
    parent_pose = px.PxTransform(px.PxVec3(lp[0], lp[1], lp[2]), px.PxQuat(lr[1], lr[2], lr[3], lr[0]))
    child_pose = px.PxTransform(px.PxVec3(0, 0, 0), px.PxQuat(lr[1], lr[2], lr[3], lr[0]))
    joint.set_parent_pose(parent_pose)
    joint.set_child_pose(child_pose)

    if jdata['jointType'] == 'spherical':
        for i, ax in enumerate(jdata['axes']):
            phys_axis = AXIS_MAP[jdata['axisMap'][i]]
            joint.set_motion(phys_axis, px.PxArticulationMotion.eLIMITED)
            joint.set_limit(phys_axis, ax['range'][0], ax['range'][1])
            joint.set_drive(phys_axis, ax['stiffness'], ax['damping'], ax['maxForce'],
                          px.PxArticulationDriveType.eFORCE)
            if 'armature' in ax:
                joint.set_armature(phys_axis, ax['armature'])
        for i in range(len(jdata['axes']), 3):
            joint.set_motion(list(AXIS_MAP.values())[i], px.PxArticulationMotion.eLOCKED)
    else:
        ax = jdata['axes'][0]
        joint.set_motion(px.PxArticulationAxis.eTWIST, px.PxArticulationMotion.eLIMITED)
        joint.set_limit(px.PxArticulationAxis.eTWIST, ax['range'][0], ax['range'][1])
        joint.set_drive(px.PxArticulationAxis.eTWIST, ax['stiffness'], ax['damping'], ax['maxForce'],
                       px.PxArticulationDriveType.eFORCE)
        if 'armature' in ax:
            joint.set_armature(px.PxArticulationAxis.eTWIST, ax['armature'])
        joint.set_motion(px.PxArticulationAxis.eSWING1, px.PxArticulationMotion.eLOCKED)
        joint.set_motion(px.PxArticulationAxis.eSWING2, px.PxArticulationMotion.eLOCKED)

# Fixed joints
for fj in hdata['fixedJoints']:
    child_link = body_link_map[fj['child_body']]
    joint = child_link.get_inbound_joint()
    joint.set_joint_type(px.PxArticulationJointType.eFIX)
    lp = fj['localPos0']
    joint.set_parent_pose(px.PxTransform(px.PxVec3(lp[0], lp[1], lp[2]), px.PxQuat(0,0,0,1)))
    joint.set_child_pose(px.PxTransform(px.PxVec3(0,0,0), px.PxQuat(0,0,0,1)))

scene.add_articulation(articulation)
print(f"Articulation: {len(links)} links, {articulation.get_dofs()} DOFs")

# Set initial state from trace
s0 = trace[0]
rp = s0['root_pos']
rr = s0['root_rot']
rv = s0['root_vel']
rav = s0['root_ang_vel']

articulation.set_root_global_pose(px.PxTransform(
    px.PxVec3(rp[0], rp[1], rp[2]),
    px.PxQuat(rr[0], rr[1], rr[2], rr[3])
))
articulation.set_root_linear_velocity(px.PxVec3(rv[0], rv[1], rv[2]))
articulation.set_root_angular_velocity(px.PxVec3(rav[0], rav[1], rav[2]))

# Set DOF positions and velocities
dof_info = hdata['dofInfo']
for i, dof in enumerate(dof_info):
    cl = body_link_map[dof['child_body']]
    joint = cl.get_inbound_joint()
    ax = AXIS_MAP[dof['physx_axis']]
    joint.set_joint_position(ax, s0['dof_pos'][i])
    joint.set_joint_velocity(ax, s0['dof_vel'][i])

# Propagate FK via cache
cache = articulation.create_cache()
articulation.copy_internal_state_to_cache(cache)
articulation.apply_cache(cache)

# Replay
print(f"\n=== REPLAY: Isaac Lab actions in ovphysx CPU PhysX ===")
for step in range(min(10, len(trace))):
    tf = trace[step]

    # Read state
    root_pose = links[0]['link'].get_global_pose()
    root_vel = links[0]['link'].get_linear_velocity()
    web_root_z = root_pose.p.z

    # Read DOF positions
    web_dof = []
    web_dof_vel = []
    for i, dof in enumerate(dof_info):
        cl = body_link_map[dof['child_body']]
        joint = cl.get_inbound_joint()
        ax = AXIS_MAP[dof['physx_axis']]
        web_dof.append(joint.get_joint_position(ax))
        web_dof_vel.append(joint.get_joint_velocity(ax))

    isaac_root_z = tf['root_pos'][2]
    root_z_diff = abs(web_root_z - isaac_root_z)

    max_dof_diff = 0
    worst_dof = ''
    max_vel_diff = 0
    for i in range(min(len(web_dof), len(tf['dof_pos']))):
        d = abs(web_dof[i] - tf['dof_pos'][i])
        if d > max_dof_diff:
            max_dof_diff = d
            worst_dof = f"{dof_info[i]['child_body']}/{dof_info[i]['axis_name']}"
        dv = abs(web_dof_vel[i] - tf['dof_vel'][i])
        if dv > max_vel_diff:
            max_vel_diff = dv

    print(f"Step {step}: rootZ: ovphysx={web_root_z:.4f} isaac={isaac_root_z:.4f} diff={root_z_diff:.4f} "
          f"| maxDofDiff={max_dof_diff:.4f} ({worst_dof}) | maxDofVelDiff={max_vel_diff:.2f}")

    # Log per-DOF details for step 1
    if step <= 1:
        dof_details = []
        vel_details = []
        for i in range(min(len(web_dof), len(tf['dof_pos']))):
            d = abs(web_dof[i] - tf['dof_pos'][i])
            if d > 0.01:
                dof_details.append(f"{dof_info[i]['child_body']}/{dof_info[i]['axis_name']}: "
                                 f"ovphysx={web_dof[i]:.4f} isaac={tf['dof_pos'][i]:.4f} diff={d:.4f}")
            dv = abs(web_dof_vel[i] - tf['dof_vel'][i])
            if dv > 0.1:
                vel_details.append(f"{dof_info[i]['child_body']}/{dof_info[i]['axis_name']}: "
                                 f"ovphysx={web_dof_vel[i]:.3f} isaac={tf['dof_vel'][i]:.3f} diff={dv:.3f}")
        if dof_details:
            print(f"  DOF diffs > 0.01: {' | '.join(dof_details)}")
        if vel_details:
            print(f"  Vel diffs > 0.1: {' | '.join(vel_details)}")

    # Apply action
    if 'action' in tf:
        for i, dof in enumerate(dof_info):
            cl = body_link_map[dof['child_body']]
            joint = cl.get_inbound_joint()
            ax = AXIS_MAP[dof['physx_axis']]
            joint.set_drive_target(ax, tf['action'][i])

    # Step physics
    for _ in range(4):
        scene.simulate(1/120)
        scene.fetch_results(True)

print("=== DONE ===")

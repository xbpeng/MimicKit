# Learnings — Web PhysX Port

## THE FIX: SWING1 ↔ SWING2 Axis Swap (2026-03-24)

PhysX WASM (physx-js-webidl@2.7.2) interprets spherical joint SWING1 and SWING2 in the opposite order from Isaac Lab. The fix: swap `physx_axis` values 1↔2 for all DOFs in `dofInfo` at load time. This single change made the character stand and perform combat motions.

**How it was found:** Systematic brute-force testing of all sign/swap permutations in `negate_test.html`. The "swap S1/S2" config gave 0.0001 error vs Isaac Lab body positions.

**Why it matters:** The DOF values from Isaac Lab are correct expmap parameters (verified: Python FK matches Isaac Lab body positions to 0.000000). But PhysX WASM's internal DOF→rotation mapping uses swapped swing axes, causing ~0.35m position errors on leg joints that prevented the character from balancing.

---
# Previous Learnings & Hypotheses

## Architecture

The web demo has three layers:
1. **PhysX WASM** — physics simulation (gravity, contacts, PD drives)
2. **JS obs computation** — reads PhysX state, computes 158-dim observation vector
3. **ONNX Runtime** — runs the ASE actor model (obs + latent → action)

The policy loop: PhysX state → JS obs → ONNX action → setDriveTarget → PhysX step → repeat

## What We Proved

### 1. JS FK = Isaac Lab FK = PhysX FK (in Isaac Lab)
Forward kinematics computed in JavaScript using `fk_local_translations` and `fk_local_rotations` from `humanoid_data.json` matches Isaac Lab's `get_body_pos()` to 0.000000 for all 17 bodies, at all poses (init, random, extreme, dynamic). This was verified with `tools/physx_vs_fk_test.py`.

### 2. JS obs = Python obs for all dynamic states
`buildObservationFromState()` in JavaScript produces identical observations to Python's `compute_char_obs()` for all 30+ frames of dynamic motion (character moving, tilted, non-zero velocities). Max error: 0.000000 across all 158 dims. Tested via the "OBS TRACE TEST" in the audit.

### 3. ONNX model is identical between web and Isaac Lab
The same ONNX file runs in both environments. Tested with `tools/test_onnx_in_isaaclab.py --use_onnx` — character performs perfect combat motions in Isaac Lab using the web's ONNX file. Action error < 0.001 across all frames.

### 4. PhysX WASM link positions diverge from Isaac Lab for legs
When setting the same root_pos, root_rot, and dof_pos, PhysX WASM's `applyCache()` produces different world-space link positions for shin and foot bodies. Upper body is correct. Error is 0.25-0.50 meters on feet. This happens even with no ground plane.

### 5. The divergence is in the knee revolute joint
The first body in the chain that diverges is `right_shin` (child of the knee joint). `right_thigh` (parent of knee) is in the correct position. The knee joint has `localRot = [0.7071, 0, 0, 0.7071]` (90° around Z axis).

### 6. Elbow revolute joints work correctly
The elbow has `localRot = [0.7071, 0, -0.7071, 0]` (90° around -Y axis) and its child body (`right_lower_arm`) is positioned correctly. This proves that revolute joints WITH `localRot` CAN work in PhysX WASM — the issue is specific to certain rotation axes.

### 7. Drives alone cannot hold the character (same in both backends)
With just PD drives targeting init_pose and no policy, the character collapses to root_z≈0.34 in BOTH Isaac Lab and web PhysX. The RL policy is essential for balance. This is not a bug — the drives are intentionally too weak for static balance (the policy provides the corrective actions).

## Key Hypothesis

The `localRot` rotation on the knee revolute joint is being interpreted differently by PhysX WASM (physx-js-webidl@2.7.2) compared to Isaac Lab's native PhysX. Specifically:

- **Elbows**: `localRot` rotates joint frame 90° around Y-axis → PhysX WASM handles this correctly
- **Knees**: `localRot` rotates joint frame 90° around Z-axis → PhysX WASM handles this INCORRECTLY

The `localRot` determines how the MJCF hinge axis maps to PhysX's twist axis:
- Elbow: MJCF Z-axis → PhysX twist (X) via 90° Y rotation ✅
- Knee: MJCF Y-axis → PhysX twist (X) via 90° Z rotation ❌

This suggests a bug or behavioral difference in how physx-js-webidl handles the joint frame rotation when the rotation is around the Z-axis vs Y-axis for revolute joints.

## Things That Don't Matter

- **Solver iterations**: Tried 4/0, 32/1, various values — no effect on the link position error
- **Ground collision**: Completely removed ground — same errors
- **T-pose vs combat stance height**: Changed pelvis_z from 0.703 to 0.903 — same errors
- **DOF ordering**: The per-joint API correctly addresses each DOF by body name + axis
- **Observation heading frame**: The ASE model uses `global_obs=False` (heading-local), but this only affects obs computation which is verified correct

## Technical Details

### PhysX Joint Frame Convention
In PhysX reduced-coordinate articulations:
- `setParentPose(PxTransform(localPos0, localRot))` — joint anchor in parent body frame
- `setChildPose(PxTransform(0, localRot))` — joint anchor in child body frame (same rotation)
- For revolute joints, `setMotion(eTWIST, eLIMITED)` — the twist axis (X in joint frame) is the hinge axis
- The `localRot` rotates the joint frame so PhysX's X-axis aligns with the MJCF hinge axis

### FK Convention
In the JS FK (matching Python's `kin_char_model.forward_kinematics`):
- `fk_local_translations[j]` — parent-to-child offset in parent frame (= `localPos0`)
- `fk_local_rotations[j]` — local rotation of the joint frame (all identity [0,0,0,1] for this character)
- Joint rotation is `axisAngleToQuat(mjcf_axis, dof_value)` for hinges
- Body rotation = `parent_rot * local_rot * joint_rot`

The FK uses identity local rotations because MJCF joints don't have a joint frame concept — the hinge axis is specified directly. The `localRot` in the PhysX joint data is a conversion artifact from `compute_joint_frame()` in `export_humanoid_json.py`.

### DOF Order
- `dofInfo` in humanoid_data.json: common (MJCF depth-first) order, matches observation vector
- PhysX sim order: different from common order (Isaac Lab applies `dof_order_sim2common` mapping)
- Per-joint API (`setJointPosition(axis, value)`) addresses DOFs by body+axis, not by flat index
- Cache buffer uses sim order — `dof_order_common2sim` mapping needed for cache operations

### Body Order
- Web creates bodies in MJCF depth-first order (same as common order)
- Isaac Lab's PhysX uses sim order internally, but `get_body_pos()` returns common order
- Both orders match for the playback comparison

## Files Created This Session
- `tools/capture_obs_trace.py` — captures 120 frames with state + obs + body_pos
- `tools/physx_vs_fk_test.py` — proves PhysX = FK in Isaac Lab
- `tools/export_amp_onnx.py` — exports AMP model to ONNX
- `tools/export_go2_web.py` — exports Go2 quadruped for web
- `tools/dump_joint_frames.py` — dumps PhysX joint frames from Isaac Lab
- `tools/dump_physx_params.py` — dumps PhysX physics parameters
- `web/obs_viz.html` — observation vector visualizer (localStorage-based)
- `web/amp.html` — AMP humanoid demo
- `web/go2.html` — Go2 quadruped demo

# Web PhysX Port Debugging Status

## Goal
Port the ASE humanoid sword-shield character to run in the browser using PhysX WASM + ONNX Runtime + three.js. The character should balance and perform combat motions like it does in Isaac Lab.

## Current State
The character holds height for 2-3 frames then falls. The policy produces reasonable actions (±1.5 range) but can't maintain balance.

## What Works (Verified)
- **ONNX model**: Correct v2 export (`tools/export_onnx_v2.py`), verified in Isaac Lab GPU+CPU. Character balances perfectly in Isaac Lab CPU with ONNX.
- **Observation dims 0-139**: PERFECT match (0.0000 error) at init_pose — height, root rotation, joint rotations, DOF velocities all correct.
- **Action bounds clipping**: Correct (joint limit based, matching training env).
- **All PhysX parameters**: Matched from USD inspection (solver iterations 32/1, armature, damping, friction=0, TGS solver, etc.).

## The Remaining Bug: Key Body Positions (obs dims 140-157)

### The Evidence
The random pose test (`tools/random_pose_test.py` → `web/random_pose_test.json`) sets a fully random state with non-zero everything and compares observations:

| Obs Category | Dims | Max Error | Status |
|-------------|------|-----------|--------|
| root_height | 0 | 0.0000 | ✅ PERFECT |
| root_rot_tan_norm | 1-6 | 0.0000 | ✅ PERFECT |
| root_vel | 7-9 | 1.7227 | ❌ setRootVelocity broken (test-only issue) |
| root_ang_vel | 10-12 | 2.3768 | ❌ setRootAngVelocity broken (test-only issue) |
| joint_rot_tan_norm | 13-108 | 0.0000 | ✅ PERFECT |
| dof_vel | 109-139 | 0.0000 | ✅ PERFECT |
| key_body_pos | 140-157 | 0.5068 | ❌ BUG - FK diverges |

**The key body positions (18 dims for 6 key bodies × 3) are wrong.** This causes action errors of 1-2 units from the very first frame, which makes the character fall.

### The Paradox
- Joint rotations (the FK INPUT) match at 0.0000
- Root position and rotation match at 0.0000
- The FK algorithm is identical between JS and Python kin_char_model
- At init_pose (DOFs from env config), the FK matches PERFECTLY (0.0000 for all 158 dims)
- At the random pose (different DOFs), the FK DIVERGES (0.5 error on key bodies)

### What Isaac Lab Actually Does
```python
# char_env.py line 291-276
body_pos = self._engine.get_body_pos(char_id)[env_ids]  # THIS IS PHYSX LINK POSITIONS!
key_pos = body_pos[..., self._key_body_ids, :]
```

But `engine.get_body_pos()` in `isaac_lab_engine.py` line 291-300 reads:
```python
body_pos = obj.data.body_link_pose_w[:, :, :3]  # PhysX articulation link world positions
```

**Isaac Lab uses PhysX link positions, NOT kin_char_model FK!** But when we use PhysX link positions in the web demo, the init_pose shows 0.42 error (PhysX links ≠ FK). When we use FK, init_pose shows 0.0000 but random poses show 0.5 error.

### The Root Confusion
The init_pose FK matches Isaac Lab perfectly, but non-init poses diverge. This seems impossible since:
1. The FK inputs (joint rotations from DOF positions) match at 0.0000 for the random pose
2. The FK algorithm is identical
3. The root position matches

**The most likely explanation**: The random pose test's `setRootGlobalPose()` + per-joint `setJointPosition()` doesn't fully propagate in the PhysX WASM articulation, so the FK uses correct joint rotations but the PhysX link positions (which `get_body_pos` returns in Isaac Lab) are from a stale state. This would explain why FK matches at init (fresh build) but not at random pose (override attempt).

## How to Run Tests

### Isaac Lab side (generate reference data):
```bash
conda run -n mimickit-isaaclab python tools/random_pose_test.py
conda run -n mimickit-isaaclab python tools/capture_closedloop_trace.py
conda run -n mimickit-isaaclab python tools/capture_detailed_trace.py
conda run -n mimickit-isaaclab python tools/pose_test_cases.py
```

### Web side (compare):
1. `python -m http.server 8080 --directory web`
2. Open `http://localhost:8080`
3. Click "Audit Obs vs Isaac Lab" button
4. Check console for: RANDOM POSE TEST, POSE TEST CASES, CLOSED-LOOP COMPARISON

### Key test files:
- `web/random_pose_test.json` — random state with all non-zero fields
- `web/closedloop_trace.json` — 10 steps from init_pose with zero latent ONNX
- `web/detailed_trace.json` — 10 steps with state_before/state_after
- `web/pose_tests.json` — 6 specific poses

## Next Steps (Priority Order)

### 1. RESOLVE THE KEY BODY POSITION QUESTION
This is the #1 blocker. Need to determine definitively whether Isaac Lab uses:
- (a) PhysX link positions (`obj.data.body_link_pose_w`)
- (b) kin_char_model FK positions

To test: In Isaac Lab, after setting a non-init pose, print BOTH `engine.get_body_pos()` AND `kin_char_model.forward_kinematics()` for the key bodies and see which one matches `compute_char_obs()` output dims 140-157.

**Important**: The `_compute_obs` code in `char_env.py` calls `get_body_pos()` BEFORE computing `joint_rot = kin_char_model.dof_to_rot(dof_pos)`. The `body_pos` used for key bodies is from `get_body_pos()`, which is PhysX link positions. But the `joint_rot` used for joint rotation obs is from kin_char_model. So the obs uses a MIX of PhysX (for body positions) and kin_char_model (for joint rotations).

### 2. FIX THE RANDOM POSE STATE SETTING
The `setRootGlobalPose()` + `setJointPosition()` in the web demo may not fully propagate. Try:
- Stepping physics once (simulate(0) + fetchResults) after setting state
- Using the articulation cache API to set all state atomically
- Verifying by reading back ALL state (root pos, rot, vel, DOF pos, vel) after setting

### 3. IF KEY_POS SHOULD BE PHYSX LINKS
The PhysX link positions differ from FK by up to 5cm at init_pose. This means the policy was trained seeing PhysX positions, not FK. To use PhysX links:
- Need to understand WHY web PhysX links differ from Isaac Lab PhysX links
- May be a body ordering issue in the `links[]` array
- Check if `links[2].link.getGlobalPose()` gives the `head` position or something else

### 4. IF KEY_POS SHOULD BE FK
Need to figure out why FK gives 0.5 error at random pose when all inputs match. Possible:
- The `fk_local_translations` or `fk_parent_indices` in `humanoid_data.json` have a subtle error
- The JS expmap-to-quat or axis-angle-to-quat doesn't match Python for some edge cases
- There's an off-by-one in how `kinematicJoints` maps to body indices

### 5. VERIFY THE BODY ORDERING MAPPING
Isaac Lab body order: `['pelvis', 'torso', 'right_thigh', 'left_thigh', 'head', 'right_upper_arm', ...]`
Web body order: `['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'right_hand', 'sword', ...]`

These are DIFFERENT orderings, but the key_body_ids `[2, 5, 10, 13, 16, 6]` happen to map to the same body names in BOTH orderings. However, Isaac Lab applies a `body_order_sim2common` mapping. Need to verify that Isaac Lab's key_body_ids are in the COMMON (MJCF) order, not the sim order.

## File Reference
- `web/index.html` — main web demo (~2200 lines)
- `web/humanoid_data.json` — character data (bodies, joints, FK, obs/action normalizers)
- `web/ase_humanoid_sword_shield_actor.onnx` — v2 ONNX model (correct)
- `tools/export_onnx_v2.py` — correct ONNX export using actual agent model
- `tools/test_onnx_in_isaaclab.py` — verify ONNX in Isaac Lab (--cpu, --use_onnx flags)
- `tools/inspect_isaaclab_physx.py` — dump PhysX parameters
- `tools/inspect_usd_physx.py` — dump USD-level PhysX parameters
- `mimickit/envs/char_env.py` — Python observation computation (compute_char_obs)
- `mimickit/engines/isaac_lab_engine.py` — Isaac Lab engine (get_body_pos uses PhysX links)
- `mimickit/anim/kin_char_model.py` — kinematic model FK

## Environment
- Isaac Lab conda env: `mimickit-isaaclab`
- PhysX WASM: `physx-js-webidl@2.7.2` from CDN
- ONNX Runtime: `onnxruntime-web@1.21.0`
- three.js: `0.170.0`

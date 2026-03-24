# Web PhysX Port Debugging Status

## Goal
Port the ASE humanoid sword-shield character to run in the browser using PhysX WASM + ONNX Runtime + three.js. The character should balance and perform combat motions like it does in Isaac Lab.

## Current State — FIXED (2026-03-24)
**The character now stands and performs combat motions in the browser!**

### Root Cause
PhysX WASM (physx-js-webidl@2.7.2) interprets spherical joint SWING1 and SWING2 axes in the **opposite order** from Isaac Lab's convention. Swapping these axes at load time fixes all body position errors.

### The Fix (in `loadHumanoidData()`)
```javascript
for (const dof of humanoidData.dofInfo) {
    if (dof.physx_axis === 1) dof.physx_axis = 2;       // SWING1 → SWING2
    else if (dof.physx_axis === 2) dof.physx_axis = 1;  // SWING2 → SWING1
}
```

### Remaining Work
- Character doesn't stay balanced as long as Isaac Lab (may need tuning of solver iterations, drive params, or substep count)
- Ground plane and visual ground re-enabled
- Substep slider added to UI for tuning
- Verbose debug logging should be cleaned up

## Previous State (before fix)

## What Is VERIFIED CORRECT (0.000000 error)
- **ONNX model**: Works flawlessly in Isaac Lab (tested with `tools/test_onnx_in_isaaclab.py --use_onnx --steps 3000`)
- **JS obs computation**: Matches Python across 30+ frames of dynamic motion (tested via `buildObservationFromState()` against `obs_trace.json`)
- **JS FK**: Matches Isaac Lab body positions exactly for ALL 17 bodies at ALL poses (dynamic, not just init)
- **Action computation**: ONNX action from JS obs matches Isaac Lab action within <0.001 across all frames
- **Joint rotation quaternions**: JS expmap/axis-angle matches Python exactly
- **DOF position readback**: Per-joint API reads correct values in correct order

## THE BUG: PhysX WASM Link Positions Wrong for Legs

### Evidence
When playing back Isaac Lab trace data (setting root_pos, root_rot, dof_pos via per-joint API + applyCache):
- **Upper body (pelvis, torso, head, arms, sword, shield)**: PhysX link positions match Isaac Lab
- **Legs (shin, foot)**: PhysX link positions are WRONG by 0.3-0.5 meters
- **Thigh positions**: Appear correct — the error starts at the **knee joint**

### Key Data
```
Frame 0 BODY POS errors (with NO ground plane):
  pelvis:          0.0000  (correct)
  torso:           0.0000  (correct)
  head:            0.0000  (correct)
  right_upper_arm: 0.0000  (correct)
  right_lower_arm: 0.0000  (correct - elbow hinge works!)
  right_hand:      0.0000  (correct)
  ...
  right_thigh:     ~0.00   (correct - hip spherical works!)
  right_shin:      0.2529  (WRONG - knee hinge BROKEN)
  right_foot:      0.3516  (WRONG - error accumulates)
  left_shin:       0.2422  (WRONG)
  left_foot:       0.3089  (WRONG)
```

### What This Means
PhysX WASM's `setJointPosition(TWIST, value)` + `applyCache()` produces different body world positions for the knee joint than our JS FK and Isaac Lab's PhysX. The DOF values are correct (verified by reading them back), but PhysX WASM resolves the joint chain differently.

### NOT Caused By
- Ground collision (tested with ground completely removed — same errors)
- Obs computation (verified 0.000000 across dynamic frames)
- ONNX model (works perfectly in Isaac Lab)
- DOF ordering (verified per-joint API uses correct body/axis mapping)
- Joint localPos0 (position offsets match between FK and PhysX joints)

## THE HYPOTHESIS: Revolute Joint Frame Rotation

### Joint Frame Comparison
The knee and elbow are both **revolute (hinge) joints** with non-identity `localRot`:

| Joint | MJCF Axis | localRot (wxyz) | PhysX Link Positions |
|-------|-----------|-----------------|---------------------|
| right_elbow | [0,0,1] (Z) | [0.7071, 0, -0.7071, 0] (90 deg around -Y) | CORRECT |
| right_knee | [0,1,0] (Y) | [0.7071, 0, 0, 0.7071] (90 deg around Z) | WRONG |
| left_elbow | [0,0,1] (Z) | [0.7071, 0, -0.7071, 0] | CORRECT |
| left_knee | [0,1,0] (Y) | [0.7071, 0, 0, 0.7071] | WRONG |

**Key observation**: Elbows (90 deg around Y-axis joint frame rotation) work. Knees (90 deg around Z-axis joint frame rotation) don't. Both are revolute joints with `localRot` applied to both parent and child pose. The difference is the rotation axis of the joint frame.

### Possible Causes
1. **PhysX WASM bug**: The physx-js-webidl 2.7.2 WASM build may handle `localRot` on revolute joints differently than Isaac Lab's native PhysX 5.x for certain rotation axes.

2. **Parent vs child pose interpretation**: Both `setParentPose` and `setChildPose` use the same rotation. The USD file also has `localRot0 = localRot1`. But PhysX WASM might interpret revolute joint frames differently from spherical ones.

3. **Joint type interaction with localRot**: When `setJointType(eREVOLUTE)` is called, PhysX locks swing1/swing2 axes. If the `localRot` rotates the frame such that the twist axis mapping conflicts with how WASM PhysX resolves the revolute constraint, the locked axes may cause unexpected behavior.

4. **Axis mapping issue**: The knee's `localRot` maps PhysX X (twist) to MJCF Y. The elbow's maps PhysX X to MJCF Z. Perhaps WASM PhysX handles the twist-to-Y mapping differently from twist-to-Z.

## How to Fix

### Approach 1: Minimal Joint Test (Recommended First Step)
Build a minimal 3-link articulation in the web: pelvis, thigh, shin with just one revolute knee joint.
- Set knee DOF to various angles (0, 0.5, 1.0, 1.5 rad)
- Read back the shin link position from PhysX
- Compare against the known-correct FK position
- Test with different `localRot` values to see which ones PhysX WASM handles correctly
- Test with `localRot = identity` and adjust the joint axis via other means

### Approach 2: Try Different Joint Frame Conventions
Instead of using `localRot` to rotate the joint frame, try:
- Setting `localRot0 = identity` and `localRot1 = identity` for ALL joints
- Using `setMotion(eSWING1, eLIMITED)` instead of `eTWIST` for knee joints (mapping the hinge to a different PhysX axis that doesn't need rotation)
- Adjusting the joint limits and drive axes accordingly

### Approach 3: Use Spherical Joints for Everything
Replace all revolute joints with spherical joints that have 2 of 3 axes locked. This avoids the revolute-specific code path in PhysX WASM entirely. Isaac Lab's PhysX may do something similar internally.

### Approach 4: Identity localRot + Rotated localPos
Instead of rotating the joint frame via `localRot`, keep `localRot = identity` and rotate the `localPos0` offset to achieve the same physical configuration. This would require recomputing the joint anchor positions.

## Current State of the Code

### Ground is disabled
Both visual (three.js) and physical (PhysX) ground are commented out for debugging. To re-enable:
- In `initPhysX()`: uncomment `pxScene.addActor(groundActor)` and remove `groundRemoved = true`
- In `init3D()`: uncomment `worldGroup.add(groundMesh)` and `worldGroup.add(grid)`
- In `buildArticulation()`: uncomment `scheduleGroundRestore()`
- In `restoreGround()`: restore the original implementation

### Playback button works
"Playback Isaac Lab Trace" loads `obs_trace.json` (120 frames of combat motion) and:
- Sets PhysX articulation state from each frame
- Renders using PhysX link positions (showing the mismatch)
- Logs per-body position errors comparing web PhysX vs Isaac Lab
- Verifies obs and action computation (both 0.000000 error)

### applyCacheNoGround helper
All `applyCache` calls for state-setting go through `applyCacheNoGround()` which removes/restores ground around the cache application. Currently ground is fully disabled so this is moot.

### global_obs fix applied
Both AMP and ASE demos now correctly handle `global_obs=True` (world-frame obs, no heading rotation). The ASE demo uses `global_obs=False` (heading-local).

## File Reference

### Web Demo Files
- `web/index.html` — main web demo (~2700 lines)
- `web/humanoid_data.json` — character data (bodies, joints, FK, normalizers, DOF ordering)
- `web/ase_humanoid_sword_shield_actor.onnx` — ONNX model (verified correct)
- `web/obs_trace.json` — 120 frames of Isaac Lab trace with body_pos for comparison
- `web/obs_viz.html` — observation visualizer
- `web/amp.html` — AMP humanoid demo (simpler, no latent)
- `web/go2.html` — Go2 quadruped demo (with tar_obs)

### Test/Export Tools
- `tools/test_onnx_in_isaaclab.py` — run ONNX in Isaac Lab (`--use_onnx --steps 3000`)
- `tools/capture_obs_trace.py` — capture state + obs trace from Isaac Lab
- `tools/physx_vs_fk_test.py` — prove PhysX = FK in Isaac Lab
- `tools/export_onnx_v2.py` — export ASE ONNX model
- `tools/export_humanoid_json.py` — export MJCF character to JSON
- `tools/export_amp_onnx.py` — export AMP ONNX model
- `tools/dump_joint_frames.py` — dump joint frames from Isaac Lab

### Key Codebase Files
- `mimickit/envs/char_env.py` — Python observation computation (`compute_char_obs`)
- `mimickit/engines/isaac_lab_engine.py` — Isaac Lab engine
- `mimickit/anim/kin_char_model.py` — kinematic model FK
- `data/assets/sword_shield/humanoid_sword_shield.xml` — MJCF character
- `data/assets/sword_shield/humanoid_sword_shield_physx.usda` — USD/PhysX version

## How to Run

### Isaac Lab (reference):
```bash
# Visualize ASE with PyTorch model
DISPLAY=:0 conda run -n mimickit-isaaclab python mimickit/run.py \
  --arg_file args/ase_humanoid_sword_shield_args.txt \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --num_envs 1 --mode test \
  --model_file data/models/ase_humanoid_sword_shield_model.pt \
  --visualize true

# Visualize with ONNX model (same model web uses)
DISPLAY=:0 conda run -n mimickit-isaaclab python tools/test_onnx_in_isaaclab.py --use_onnx --steps 3000

# Capture obs trace for web comparison
conda run -n mimickit-isaaclab python tools/capture_obs_trace.py
```

### Web Demo:
```bash
python -m http.server 8080 --directory web
# Open http://localhost:8080/index.html
# Click "Playback Isaac Lab Trace" to see the joint frame mismatch
# Console shows per-body position errors
```

### Playback Controls
- "Playback Isaac Lab Trace" — steps through Isaac Lab frames, shows PhysX link positions vs Isaac Lab reference
- "Audit Obs vs Isaac Lab" — runs comprehensive obs/action/dynamics comparison
- Gravity slider — adjust gravity
- Ground is currently disabled for debugging

## Environment
- Isaac Lab conda env: `mimickit-isaaclab`
- PhysX WASM: `physx-js-webidl@2.7.2` from CDN
- ONNX Runtime: `onnxruntime-web@1.21.0`
- three.js: `0.170.0`
- DISPLAY=:0 needed for Isaac Lab GUI

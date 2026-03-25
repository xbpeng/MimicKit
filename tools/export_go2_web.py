"""Export everything needed for Go2 web demo.

Generates:
1. ONNX model (obs → action, no latent)
2. humanoid_data.json (character, joints, FK, normalizers)
3. Motion data embedded in the JSON for tar_obs computation

Uses Isaac Gym engine (which doesn't have joint limit validation issues).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mimickit'))

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

# ---- Step 1: Export ONNX model ----
print("=" * 60)
print("Step 1: Export ONNX model")
print("=" * 60)

ckpt = torch.load('data/models/deepmimic_go2_pace_model.pt', map_location='cpu', weights_only=False)

obs_dim = ckpt['_obs_norm._mean'].shape[0]  # 484
act_dim = ckpt['_a_norm._mean'].shape[0]    # 12
print(f"obs_dim={obs_dim}, act_dim={act_dim}")

class PPOActorWrapper(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.register_buffer('obs_mean', ckpt['_obs_norm._mean'].clone())
        self.register_buffer('obs_std', ckpt['_obs_norm._std'].clone().clamp(min=1e-4))
        self.obs_clip = 5.0
        self.register_buffer('a_mean', ckpt['_a_norm._mean'].clone())
        self.register_buffer('a_std', ckpt['_a_norm._std'].clone().clamp(min=1e-4))

        # Reconstruct actor layers from checkpoint
        self.actor_layers = nn.Sequential(
            nn.Linear(ckpt['_model._actor_layers.0.weight'].shape[1],
                      ckpt['_model._actor_layers.0.weight'].shape[0]),
            nn.ReLU(),
            nn.Linear(ckpt['_model._actor_layers.2.weight'].shape[1],
                      ckpt['_model._actor_layers.2.weight'].shape[0]),
            nn.ReLU(),
        )
        self.actor_layers[0].weight.data = ckpt['_model._actor_layers.0.weight']
        self.actor_layers[0].bias.data = ckpt['_model._actor_layers.0.bias']
        self.actor_layers[2].weight.data = ckpt['_model._actor_layers.2.weight']
        self.actor_layers[2].bias.data = ckpt['_model._actor_layers.2.bias']

        self.mean_net = nn.Linear(ckpt['_model._action_dist._mean_net.weight'].shape[1],
                                  ckpt['_model._action_dist._mean_net.weight'].shape[0])
        self.mean_net.weight.data = ckpt['_model._action_dist._mean_net.weight']
        self.mean_net.bias.data = ckpt['_model._action_dist._mean_net.bias']

    def forward(self, raw_obs: torch.Tensor) -> torch.Tensor:
        norm_obs = (raw_obs - self.obs_mean) / self.obs_std
        norm_obs = torch.clamp(norm_obs, -self.obs_clip, self.obs_clip)
        h = self.actor_layers(norm_obs)
        norm_action = self.mean_net(h)
        action = norm_action * self.a_std + self.a_mean
        return action

wrapper = PPOActorWrapper(ckpt)
wrapper.eval()

dummy_obs = torch.randn(1, obs_dim)
pt_action = wrapper(dummy_obs)

onnx_path = 'web/go2_actor.onnx'
torch.onnx.export(wrapper, (dummy_obs,), onnx_path,
                  input_names=["obs"], output_names=["action"],
                  dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
                  opset_version=13)

sess = ort.InferenceSession(onnx_path)
ort_action = sess.run(['action'], {'obs': dummy_obs.numpy()})[0]
max_diff = np.abs(pt_action.detach().numpy() - ort_action).max()
print(f"ONNX exported: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")
print(f"Max PT/ONNX diff: {max_diff}")

# Re-save with embedded weights if small
if os.path.getsize(onnx_path) < 100000:
    import onnx
    model = onnx.load(onnx_path)
    onnx.save(model, onnx_path, save_as_external_data=False)
    print(f"Re-saved: {os.path.getsize(onnx_path)/1024/1024:.1f} MB")

# ---- Step 2: Generate humanoid_data.json ----
print("\n" + "=" * 60)
print("Step 2: Generate character JSON")
print("=" * 60)

# Run the existing export tool
os.system(f'python tools/export_humanoid_json.py '
          f'--mjcf data/assets/go2/go2.xml '
          f'--model data/models/deepmimic_go2_pace_model.pt '
          f'--output web/go2_data.json '
          f'--pelvis_z 0.27 '
          f'--latent_dim 0')

# Load and augment
with open('web/go2_data.json') as f:
    d = json.load(f)

# Add init pose from env config
init_pose = [0, 0, 0.27, 0, 0, 0, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8]
d['init_root_pos'] = init_pose[:3]
d['init_root_rot_quat'] = [0.0, 0.0, 0.0, 1.0]  # identity
d['init_dof_pos'] = init_pose[6:]  # skip root pos(3) + root rot expmap(3)
d['key_body_ids'] = []  # Will be set from body names
d['obs_dim'] = obs_dim  # 484
d['act_dim'] = act_dim   # 12
d['latent_dim'] = 0
d['global_obs'] = True
d['root_height_obs'] = True
d['enable_tar_obs'] = True
d['tar_obs_steps'] = [1, 2, 3]

# Key bodies
key_body_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
body_names = [b['name'] for b in d['bodies']]
d['key_body_ids'] = [body_names.index(n) for n in key_body_names]
d['key_body_names'] = key_body_names
print(f"Key body IDs: {d['key_body_ids']}")
print(f"Body names: {body_names}")

# Build kinematicJoints
kinJoints = []
dof_idx = 0
for j in d['joints']:
    if j['jointType'] == 'spherical':
        kinJoints.append({
            'name': j.get('joint_name', j['child_body']),
            'type': 'SPHERICAL', 'dof_idx': dof_idx, 'dof_dim': 3,
            'axis': None, 'child_body': j['child_body']
        })
        dof_idx += 3
    elif j['jointType'] == 'revolute':
        kinJoints.append({
            'name': j.get('joint_name', j['child_body']),
            'type': 'HINGE', 'dof_idx': dof_idx, 'dof_dim': 1,
            'axis': [0.0, 0.0, 1.0], 'child_body': j['child_body']
        })
        dof_idx += 1
d['kinematicJoints'] = kinJoints
print(f"Kinematic joints: {len(kinJoints)}, total DOFs: {dof_idx}")

# FK data from kin_char_model — need to build it from the MJCF
# The export_humanoid_json.py should have already added fk_parent_indices etc.
if 'fk_parent_indices' not in d:
    print("WARNING: fk_parent_indices missing, FK won't work")

# Action bounds (from training: joint limits * multiplier)
# Go2 has actuator range [-33.5, 33.5] but the policy outputs DOF positions
# Action bounds are joint limits
dof_info = d['dofInfo']
action_low = []
action_high = []
for di in dof_info:
    # Find corresponding joint
    for j in d['joints']:
        for ax in j['axes']:
            if ax['name'] == di['axis_name']:
                action_low.append(ax['range'][0] * 1.2)
                action_high.append(ax['range'][1] * 1.2)
                break
        else:
            continue
        break
    else:
        action_low.append(-3.14)
        action_high.append(3.14)

d['action_low'] = action_low
d['action_high'] = action_high

# ---- Step 3: Add motion data ----
print("\n" + "=" * 60)
print("Step 3: Embed motion data")
print("=" * 60)

with open('data/motions/go2/go2_pace.pkl', 'rb') as f:
    motion = pickle.load(f)

d['motion'] = {
    'fps': motion['fps'],
    'loop_mode': motion['loop_mode'],  # 1 = loop
    'frames': [list(float(x) for x in frame) for frame in motion['frames']],
    'frame_format': 'root_pos(3) root_rot_expmap(3) dof_pos(12)',
    'num_frames': len(motion['frames']),
    'duration': len(motion['frames']) / motion['fps'],
}
print(f"Motion: {d['motion']['num_frames']} frames at {d['motion']['fps']}fps = {d['motion']['duration']:.3f}s")

# Normalizer data
d['obs_mean'] = ckpt['_obs_norm._mean'].numpy().tolist()
d['obs_std'] = ckpt['_obs_norm._std'].numpy().tolist()
d['a_mean'] = ckpt['_a_norm._mean'].numpy().tolist()
d['a_std'] = ckpt['_a_norm._std'].numpy().tolist()

with open('web/go2_data.json', 'w') as f:
    json.dump(d, f)

size_kb = os.path.getsize('web/go2_data.json') / 1024
print(f"\nSaved web/go2_data.json ({size_kb:.0f} KB)")
print(f"Bodies: {len(d['bodies'])}, Joints: {len(d['joints'])}, DOFs: {dof_idx}")
print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")
print("\nDone! Files created:")
print(f"  web/go2_actor.onnx ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")
print(f"  web/go2_data.json ({size_kb:.0f} KB)")

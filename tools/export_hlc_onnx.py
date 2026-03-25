"""Export ASE High-Level Controllers (HLCs) to ONNX.

Each HLC takes (normalized_obs_with_task) and outputs a latent vector.
The web demo runs: task_obs → HLC → latent → LLC → action.

Usage:
    python tools/export_hlc_onnx.py
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

ASE_MODELS = os.path.expanduser('/home/selstad/Desktop/ASE/ase/data/models')

HLCS = {
    'heading':  {'path': f'{ASE_MODELS}/ase_hlc_heading_reallusion_sword_shield.pth',  'task_obs_size': 5},
    'location': {'path': f'{ASE_MODELS}/ase_hlc_location_reallusion_sword_shield.pth', 'task_obs_size': 2},
    'reach':    {'path': f'{ASE_MODELS}/ase_hlc_reach_reallusion_sword_shield.pth',    'task_obs_size': 3},
    'strike':   {'path': f'{ASE_MODELS}/ase_hlc_strike_reallusion_sword_shield.pth',   'task_obs_size': 15},
}

class HLCWrapper(nn.Module):
    """Wraps HLC: raw_obs (LLC obs + task_obs) → normalized → MLP → L2-normalized latent."""
    def __init__(self, ckpt):
        super().__init__()
        rms = ckpt['running_mean_std']
        self.register_buffer('obs_mean', rms['running_mean'].float())
        self.register_buffer('obs_var', rms['running_var'].float())
        self.obs_clip = 5.0

        model = ckpt['model']
        # Actor MLP: 2 hidden layers with ReLU
        self.mlp = nn.Sequential(
            nn.Linear(model['a2c_network.actor_mlp.0.weight'].shape[1],
                      model['a2c_network.actor_mlp.0.weight'].shape[0]),
            nn.ReLU(),
            nn.Linear(model['a2c_network.actor_mlp.2.weight'].shape[1],
                      model['a2c_network.actor_mlp.2.weight'].shape[0]),
            nn.ReLU(),
        )
        self.mu = nn.Linear(model['a2c_network.mu.weight'].shape[1],
                            model['a2c_network.mu.weight'].shape[0])

        # Load weights
        self.mlp[0].weight.data = model['a2c_network.actor_mlp.0.weight']
        self.mlp[0].bias.data = model['a2c_network.actor_mlp.0.bias']
        self.mlp[2].weight.data = model['a2c_network.actor_mlp.2.weight']
        self.mlp[2].bias.data = model['a2c_network.actor_mlp.2.bias']
        self.mu.weight.data = model['a2c_network.mu.weight']
        self.mu.bias.data = model['a2c_network.mu.bias']

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Normalize
        norm_obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-5)
        norm_obs = torch.clamp(norm_obs, -self.obs_clip, self.obs_clip)
        # MLP
        h = self.mlp(norm_obs)
        latent = self.mu(h)
        # L2 normalize (HLC outputs are normalized to unit sphere)
        latent = torch.nn.functional.normalize(latent, dim=-1)
        return latent


def export_hlc(name, info, output_dir='web'):
    print(f'\n=== Exporting {name} HLC ===')
    ckpt = torch.load(info['path'], map_location='cpu', weights_only=False)
    wrapper = HLCWrapper(ckpt)
    wrapper.eval()

    obs_dim = wrapper.obs_mean.shape[0]
    latent_dim = wrapper.mu.weight.shape[0]
    task_obs_size = info['task_obs_size']
    llc_obs_size = obs_dim - task_obs_size

    print(f'  obs_dim={obs_dim} (llc_obs={llc_obs_size} + task_obs={task_obs_size})')
    print(f'  latent_dim={latent_dim}')

    # Export
    dummy = torch.randn(1, obs_dim)
    output_path = os.path.join(output_dir, f'hlc_{name}.onnx')

    torch.onnx.export(
        wrapper, dummy, output_path,
        input_names=['obs'], output_names=['latent'],
        dynamic_axes={'obs': {0: 'batch'}, 'latent': {0: 'batch'}},
        opset_version=17,
    )

    # Bake metadata
    model = onnx.load(output_path)
    meta = {
        'hlc_name': name,
        'obs_dim': obs_dim,
        'llc_obs_dim': llc_obs_size,
        'task_obs_size': task_obs_size,
        'latent_dim': latent_dim,
    }
    entry = onnx.StringStringEntryProto(key='mimickit_hlc', value=json.dumps(meta))
    model.metadata_props.append(entry)
    onnx.save(model, output_path, save_as_external_data=False)

    # Verify
    sess = ort.InferenceSession(output_path)
    with torch.no_grad():
        pt_out = wrapper(dummy).numpy()
    ort_out = sess.run(['latent'], {'obs': dummy.numpy()})[0]
    diff = np.max(np.abs(pt_out - ort_out))
    print(f'  Max diff: {diff:.2e}')
    assert diff < 1e-3

    size_kb = os.path.getsize(output_path) / 1024
    print(f'  Exported: {output_path} ({size_kb:.0f} KB)')
    return output_path


if __name__ == '__main__':
    os.makedirs('web', exist_ok=True)
    for name, info in HLCS.items():
        export_hlc(name, info)
    print('\nAll HLCs exported.')

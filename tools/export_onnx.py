"""Export the ASE actor (obs + latent → action) to ONNX and verify equivalence.

The exported model bakes in the observation normalizer and action unnormalizer
so the ONNX graph is:
    raw_obs (158,) + latent (64,)  →  action (31,)

Usage:
    python tools/export_onnx.py \
        --model_file data/models/ase_humanoid_sword_shield_model.pt \
        --output data/models/ase_humanoid_sword_shield_actor.onnx
"""

import argparse
import numpy as np
import torch
import torch.nn as nn


class ASEActorForExport(nn.Module):
    """Wraps the actor layers + normalizers into a single forward pass.

    Input:  raw_obs (B, 158), latent (B, 64)
    Output: action  (B, 31)
    """

    def __init__(self, ckpt):
        super().__init__()

        # --- Observation normalizer ---
        self.register_buffer("obs_mean", ckpt["_obs_norm._mean"])
        self.register_buffer("obs_std",  ckpt["_obs_norm._std"].clamp(min=1e-5))

        # --- Action unnormalizer ---
        self.register_buffer("a_mean", ckpt["_a_norm._mean"])
        self.register_buffer("a_std",  ckpt["_a_norm._std"].clamp(min=1e-5))

        # --- Actor network layers ---
        # Reconstruct the Sequential from state dict keys
        actor_keys = [k for k in ckpt if k.startswith("_model._actor_layers.")]
        # Figure out layer count from keys like _model._actor_layers.0.weight
        layer_indices = sorted(set(int(k.split(".")[2]) for k in actor_keys))

        # Reconstruct Sequential: layer indices 0,2,4 are Linear; 1,3 are ReLU
        # Build a full list inserting ReLU between consecutive Linear layers
        max_idx = max(layer_indices)
        layers = []
        for idx in range(max_idx + 1):
            w_key = f"_model._actor_layers.{idx}.weight"
            b_key = f"_model._actor_layers.{idx}.bias"
            if w_key in ckpt:
                linear = nn.Linear(ckpt[w_key].shape[1], ckpt[w_key].shape[0])
                linear.weight.data = ckpt[w_key]
                linear.bias.data = ckpt[b_key]
                layers.append(linear)
            else:
                layers.append(nn.ReLU())

        self.actor_layers = nn.Sequential(*layers)

        # --- Mean output head ---
        mean_w = ckpt["_model._action_dist._mean_net.weight"]
        mean_b = ckpt["_model._action_dist._mean_net.bias"]
        self.mean_net = nn.Linear(mean_w.shape[1], mean_w.shape[0])
        self.mean_net.weight.data = mean_w
        self.mean_net.bias.data = mean_b

    def forward(self, raw_obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        # Normalize observation
        norm_obs = (raw_obs - self.obs_mean) / self.obs_std

        # Concatenate obs + latent
        x = torch.cat([norm_obs, latent], dim=-1)

        # Actor forward pass
        h = self.actor_layers(x)
        norm_action = self.mean_net(h)  # deterministic (mode) action

        # Unnormalize action
        action = norm_action * self.a_std + self.a_mean
        return action


def export_onnx(model_file: str, output_file: str):
    ckpt = torch.load(model_file, map_location="cpu", weights_only=False)

    model = ASEActorForExport(ckpt)
    model.eval()

    obs_dim = ckpt["_obs_norm._mean"].shape[0]
    latent_dim = ckpt["_model._enc_out.weight"].shape[0]
    act_dim = ckpt["_a_norm._mean"].shape[0]

    print(f"obs_dim={obs_dim}  latent_dim={latent_dim}  act_dim={act_dim}")

    # Dummy inputs
    dummy_obs = torch.randn(1, obs_dim)
    dummy_z = torch.randn(1, latent_dim)

    # PyTorch reference output
    with torch.no_grad():
        pt_action = model(dummy_obs, dummy_z)
    print(f"PyTorch output shape: {pt_action.shape}")
    print(f"PyTorch output sample: {pt_action[0, :5].numpy()}")

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_obs, dummy_z),
        output_file,
        input_names=["obs", "latent"],
        output_names=["action"],
        dynamic_axes={
            "obs": {0: "batch"},
            "latent": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported to {output_file}")

    # Verify with ONNX Runtime
    import onnxruntime as ort

    sess = ort.InferenceSession(output_file)
    ort_inputs = {
        "obs": dummy_obs.numpy(),
        "latent": dummy_z.numpy(),
    }
    ort_action = sess.run(["action"], ort_inputs)[0]

    print(f"ONNX Runtime output shape: {ort_action.shape}")
    print(f"ONNX Runtime output sample: {ort_action[0, :5]}")

    max_diff = np.max(np.abs(pt_action.numpy() - ort_action))
    print(f"Max absolute difference (PyTorch vs ONNX): {max_diff:.2e}")
    rel_diff = max_diff / (np.max(np.abs(pt_action.numpy())) + 1e-8)
    print(f"Relative difference: {rel_diff:.2e}")
    assert max_diff < 1e-3 or rel_diff < 1e-5, f"ONNX output diverges! max_diff={max_diff} rel_diff={rel_diff}"
    print("PASS: PyTorch and ONNX outputs are identical within tolerance.")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", default="data/models/ase_humanoid_sword_shield_model.pt")
    parser.add_argument("--output", default="data/models/ase_humanoid_sword_shield_actor.onnx")
    args = parser.parse_args()
    export_onnx(args.model_file, args.output)

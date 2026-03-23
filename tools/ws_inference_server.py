"""
WebSocket inference server for MimicKit ASE model.

Loads the PyTorch checkpoint and ONNX model, then serves inference requests
over WebSocket on localhost:8765.

Protocol:
  Client sends JSON: {"obs": [158 floats], "latent": [64 floats]}
  Server returns JSON: {
      "pytorch_action": [31 floats],
      "onnx_action": [31 floats],
      "max_diff": float
  }

Usage:
  cd /path/to/MimicKit
  python tools/ws_inference_server.py
"""

import asyncio
import json
import sys
import os

import torch
import numpy as np
import onnxruntime as ort
import websockets

# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = os.path.join(REPO_ROOT, "data/models/ase_humanoid_sword_shield_model.pt")
ONNX_PATH = os.path.join(REPO_ROOT, "data/models/ase_humanoid_sword_shield_actor_internal.onnx")

print(f"Loading checkpoint from {CKPT_PATH} ...")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

# ---------------------------------------------------------------------------
# Reconstruct observation normalizer
# Normalizer.normalize: norm_x = clamp((x - mean) / std, -clip, clip)
# obs_norm uses clip=10.0 (from base_agent.py line 159)
# ---------------------------------------------------------------------------
obs_mean = ckpt["_obs_norm._mean"]                    # shape [158]
obs_std = ckpt["_obs_norm._std"].clamp(min=1e-4)     # shape [158], clamped to match ONNX export
OBS_CLIP = 10.0

# ---------------------------------------------------------------------------
# Reconstruct action normalizer
# Normalizer.unnormalize: x = norm_x * std + mean
# ---------------------------------------------------------------------------
a_mean = ckpt["_a_norm._mean"]                        # shape [31]
a_std = ckpt["_a_norm._std"].clamp(min=1e-4)          # shape [31], clamped to match ONNX export

# ---------------------------------------------------------------------------
# Reconstruct actor model
# Architecture (from checkpoint keys, matching ONNX export):
#   _actor_layers: Sequential(
#       0: Linear(222, 1024)
#       1: ReLU
#       2: Linear(1024, 1024)
#       3: ReLU
#       4: Linear(1024, 512)
#   )
#   _action_dist._mean_net: Linear(512, 31)
# ---------------------------------------------------------------------------
actor_layers = torch.nn.Sequential(
    torch.nn.Linear(222, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 512),
)

# Load actor layer weights
actor_layers[0].weight.data = ckpt["_model._actor_layers.0.weight"]
actor_layers[0].bias.data = ckpt["_model._actor_layers.0.bias"]
actor_layers[2].weight.data = ckpt["_model._actor_layers.2.weight"]
actor_layers[2].bias.data = ckpt["_model._actor_layers.2.bias"]
actor_layers[4].weight.data = ckpt["_model._actor_layers.4.weight"]
actor_layers[4].bias.data = ckpt["_model._actor_layers.4.bias"]

# Mean net (deterministic action = distribution mode = mean)
mean_net = torch.nn.Linear(512, 31)
mean_net.weight.data = ckpt["_model._action_dist._mean_net.weight"]
mean_net.bias.data = ckpt["_model._action_dist._mean_net.bias"]

actor_layers.eval()
mean_net.eval()

print(f"Actor model reconstructed: input=222, output=31")

# ---------------------------------------------------------------------------
# Load ONNX model
# ---------------------------------------------------------------------------
print(f"Loading ONNX model from {ONNX_PATH} ...")
onnx_sess = ort.InferenceSession(ONNX_PATH)
print("ONNX model loaded")

# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------
async def handler(websocket):
    peer = websocket.remote_address
    print(f"Client connected: {peer}")
    try:
        async for message in websocket:
            data = json.loads(message)
            obs = torch.tensor(data["obs"], dtype=torch.float32).unsqueeze(0)    # [1, 158]
            latent = torch.tensor(data["latent"], dtype=torch.float32).unsqueeze(0)  # [1, 64]

            # ---------------------------------------------------------------
            # PyTorch inference path (exact replica of ase_agent._decide_action)
            # ---------------------------------------------------------------
            with torch.no_grad():
                # 1. Normalize observation (Normalizer.normalize with clip=10)
                norm_obs = (obs - obs_mean) / obs_std
                norm_obs = norm_obs.clamp(-OBS_CLIP, OBS_CLIP)

                # 2. Concatenate with latent (ASEModel.eval_actor)
                in_data = torch.cat([norm_obs, latent], dim=-1)  # [1, 222]

                # 3. Forward through actor layers
                h = actor_layers(in_data)

                # 4. Mean net (DistributionGaussianDiag.mode == mean)
                norm_a = mean_net(h)

                # 5. Unnormalize action (Normalizer.unnormalize)
                a = norm_a * a_std + a_mean

            pytorch_action = a[0].numpy()

            # ---------------------------------------------------------------
            # ONNX inference path
            # ---------------------------------------------------------------
            onnx_out = onnx_sess.run(
                ["action"],
                {
                    "obs": obs.numpy(),
                    "latent": latent.numpy(),
                },
            )[0]

            onnx_action = onnx_out[0]

            max_diff = float(np.max(np.abs(pytorch_action - onnx_action)))

            result = {
                "pytorch_action": pytorch_action.tolist(),
                "onnx_action": onnx_action.tolist(),
                "max_diff": max_diff,
            }
            await websocket.send(json.dumps(result))
    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {peer}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    host = "localhost"
    port = 8765
    print(f"Starting WebSocket server on ws://{host}:{port} ...")
    async with websockets.serve(handler, host, port):
        print(f"WebSocket server running on ws://{host}:{port}")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())

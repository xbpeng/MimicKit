# Copyright (c) 2026 Holiday Robotics
# SPDX-License-Identifier: MIT
# Adapted from https://github.com/Holiday-Robot/FlashSAC at 87edc906.
# See LICENSES/FlashSAC-MIT.txt.
"""Models, update functions, and numerical utilities for FlashSAC."""

import math
import os
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


# Numerical and compilation utilities


def safe_tanh_log_det_jacobian(x):
    """Numerically stable log determinant of the tanh Jacobian."""
    return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


def warmup_cosine_decay_scheduler(init_value, peak_value, end_value, warmup_steps, decay_steps):
    """Optax-style linear warmup followed by cosine decay."""
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if decay_steps <= warmup_steps:
        raise ValueError("decay_steps must be greater than warmup_steps")

    def scheduler(step):
        if step < warmup_steps:
            return init_value + (peak_value - init_value) * (step / warmup_steps)
        if step < decay_steps:
            decay_step = step - warmup_steps
            progress = decay_step / (decay_steps - warmup_steps)
            return end_value + (peak_value - end_value) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return end_value

    return scheduler


def resolve_compile_mode(mode):
    if mode != "auto":
        return mode
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    return "max-autotune" if (major, minor) >= (2, 9) else "reduce-overhead"


def compile_method(module, method_name, mode):
    """Compile one module method without wrapping or renaming module parameters."""
    if not hasattr(torch, "compile"):
        raise RuntimeError("FlashSAC use_compile requires PyTorch 2.0 or newer")
    method = getattr(module, method_name)
    setattr(module, method_name, torch.compile(method, mode=mode))


def build_grad_scaler(device, enabled):
    """Build a GradScaler across old and new PyTorch AMP namespaces."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device=device.type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


@contextmanager
def autocast(device, enabled):
    """Use FP16 autocast when enabled while retaining older PyTorch support."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16, enabled=enabled
        ):
            yield
    else:
        with torch.cuda.amp.autocast(enabled=enabled):
            yield


def prepare_module_parameter_normalization(module, compile_mode=None):
    normalizers = getattr(module, "_flashsac_parameter_normalizers", None)
    if normalizers is None:
        normalizers = tuple(
            child.normalize_parameters
            for child in module.modules()
            if hasattr(child, "normalize_parameters")
        )
        # A tuple of bound methods is deliberately not a ModuleList: it is an
        # execution cache and must not alter checkpoint parameter names.
        object.__setattr__(module, "_flashsac_parameter_normalizers", normalizers)

    def normalize_parameters():
        for normalize_parameter in normalizers:
            normalize_parameter()

    if compile_mode is not None:
        if not hasattr(torch, "compile"):
            raise RuntimeError("Compiled parameter normalization requires PyTorch 2.0 or newer")
        normalize_parameters = torch.compile(normalize_parameters, mode=compile_mode)
    object.__setattr__(module, "_flashsac_parameter_normalize_fn", normalize_parameters)


@torch.no_grad()
def normalize_module_parameters(module):
    normalize_parameters = getattr(module, "_flashsac_parameter_normalize_fn", None)
    if normalize_parameters is None:
        prepare_module_parameter_normalization(module)
        normalize_parameters = module._flashsac_parameter_normalize_fn
    normalize_parameters()


def prepare_ema_update(target_module, source_module, tau, compile_mode=None):
    target_params = tuple(target_module.parameters())
    source_params = tuple(source_module.parameters())

    def ema_update():
        if hasattr(torch, "_foreach_lerp_"):
            torch._foreach_lerp_(target_params, source_params, tau)
        else:
            for target, source in zip(target_params, source_params):
                target.lerp_(source, tau)

    if compile_mode is not None:
        if not hasattr(torch, "compile"):
            raise RuntimeError("Compiled EMA requires PyTorch 2.0 or newer")
        ema_update = torch.compile(ema_update, mode=compile_mode)
    cache = (id(source_module), float(tau), ema_update)
    object.__setattr__(target_module, "_flashsac_ema_cache", cache)


@torch.no_grad()
def ema_update_module(target_module, source_module, tau):
    cache = getattr(target_module, "_flashsac_ema_cache", None)
    if cache is None or cache[0] != id(source_module) or cache[1] != float(tau):
        prepare_ema_update(target_module, source_module, tau)
        cache = target_module._flashsac_ema_cache
    cache[2]()


# Reward normalization


class RunningMeanStd:
    """Parallel running mean and variance used by the reward normalizer."""

    def __init__(self, device, epsilon=1e-4, shape=(1,), dtype=torch.float32):
        self.mean = torch.zeros(shape, dtype=dtype, device=device)
        self.var = torch.ones(shape, dtype=dtype, device=device)
        self.count = torch.tensor(0.0, dtype=dtype, device=device)
        self.epsilon = epsilon
        self.device = device

    def update(self, samples):
        sample_mean = torch.mean(samples, dim=0)
        sample_var = torch.var(samples, dim=0, unbiased=False)
        sample_count = float(samples.shape[0])

        delta = sample_mean - self.mean
        total_count = self.count + sample_count
        ratio = sample_count / total_count
        new_mean = self.mean + delta * ratio
        m_a = self.var * (self.count + self.epsilon)
        m_b = sample_var * sample_count
        m2 = m_a + m_b + torch.square(delta) * self.count * ratio

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state):
        self.mean = state["mean"].to(self.device)
        self.var = state["var"].to(self.device)
        self.count = state["count"].to(self.device)


class RewardNormalizer:
    """Normalize rewards using discounted-return variance and maximum magnitude."""

    def __init__(self, gamma, G_max, device, epsilon=1e-8):
        self.gamma = gamma
        self.G_r = torch.zeros(1, dtype=torch.float32, device=device)
        self.G_r_max = torch.zeros(1, dtype=torch.float32, device=device)
        self.G_rms = RunningMeanStd(shape=(1,), device=device, dtype=torch.float32)
        self.G_max = G_max
        self.epsilon = epsilon
        self.device = device

    def update_reward_stats(self, reward, terminated, truncated):
        done = torch.logical_or(terminated.bool(), truncated.bool()).float()
        self.G_r = self.gamma * (1.0 - done) * self.G_r + reward
        self.G_r_max = torch.maximum(self.G_r_max, torch.max(torch.abs(self.G_r)))
        self.G_rms.update(self.G_r.reshape(-1, 1))

    def normalize_rewards(self, rewards):
        var_denominator = torch.sqrt(self.G_rms.var.reshape(()) + self.epsilon)
        min_required_denominator = self.G_r_max.reshape(()) / self.G_max
        denominator = torch.maximum(var_denominator, min_required_denominator)
        return rewards / denominator

    def reset_returns(self):
        """Reset per-environment returns without discarding accumulated statistics."""
        # Return to the reference implementation's lazy shape. The next reward
        # update expands this to the current environment count, which also makes
        # checkpoints resumable with a different number of vector environments.
        self.G_r = torch.zeros(1, dtype=torch.float32, device=self.device)

    def state_dict(self):
        return {
            "G_r": self.G_r,
            "G_r_max": self.G_r_max,
            "G_rms": self.G_rms.state_dict(),
        }

    def load_state_dict(self, state):
        self.G_r = state["G_r"].to(self.device)
        self.G_r_max = state["G_r_max"].to(self.device)
        self.G_rms.load_state_dict(state["G_rms"])

    def save(self, path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))


# Network layers


class UnitLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.w = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.w.weight, gain=1)

    def forward(self, x):
        return self.w(x)

    def normalize_parameters(self):
        """Project each output weight vector onto the unit sphere."""
        self.w.weight.copy_(F.normalize(self.w.weight, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
    def __init__(self, input_dim, momentum=0.01, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x, training):
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def normalize_parameters(self):
        """Project the concatenated scale and bias vector to norm sqrt(d)."""
        scale, bias = self.weight.data, self.bias.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class UnitRMSNorm(nn.Module):
    def __init__(self, input_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.eps = eps

    def forward(self, x):
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, self.weight.shape, self.weight, eps=self.eps)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

    def normalize_parameters(self):
        scale = self.weight.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)


class FlashSACEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.norm = UnitBatchNorm(input_dim)
        self.w = UnitLinear(input_dim, hidden_dim)

    def forward(self, x, training):
        x = self.norm(x, training=training)
        return self.w(x)


class FlashSACBlock(nn.Module):
    def __init__(self, hidden_dim, expansion=4):
        super().__init__()
        self.w1 = UnitLinear(hidden_dim, hidden_dim * expansion)
        self.w2 = UnitLinear(hidden_dim * expansion, hidden_dim)
        self.norm1 = UnitBatchNorm(hidden_dim * expansion)
        self.norm2 = UnitBatchNorm(hidden_dim)

    def forward(self, x, training):
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = F.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        x = F.relu(x)
        return x + residual


class NormalTanhPolicy(nn.Module):
    def __init__(self, hidden_dim, action_dim, log_std_min=-10.0, log_std_max=2.0):
        super().__init__()
        self.mean_w = UnitLinear(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))
        self.std_w = UnitLinear(hidden_dim, action_dim)
        self.std_bias = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def get_mean_and_std(self, x, training):
        del training
        mean = F.linear(x, self.mean_w.w.weight, self.mean_bias)
        raw_log_std = F.linear(x, self.std_w.w.weight, self.std_bias)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1 + torch.tanh(raw_log_std)
        )
        return mean, torch.exp(log_std)

    def forward(self, x, training):
        mean, std = self.get_mean_and_std(x, training)
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action)
        log_prob = log_prob - safe_tanh_log_det_jacobian(raw_action)
        return action, {"log_prob": log_prob.sum(dim=-1)}


class EnsembleUnitLinear(nn.Module):
    def __init__(self, num_ensemble, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_ensemble, output_dim, input_dim))
        for idx in range(num_ensemble):
            nn.init.orthogonal_(self.weight.data[idx], gain=1)

    def forward(self, x):
        return torch.einsum("nbi,noi->nbo", x, self.weight)

    def normalize_parameters(self):
        self.weight.copy_(F.normalize(self.weight, dim=-1, eps=1e-8))


class EnsembleUnitBatchNorm(nn.Module):
    def __init__(self, num_ensemble, input_dim, momentum=0.01, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_mean", torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_var", torch.ones(num_ensemble, input_dim))

    def forward(self, x, training):
        if training:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, unbiased=False, keepdim=True)
            with torch.no_grad():
                batch_size = x.shape[1]
                self.running_mean.lerp_(mean.squeeze(1).float(), self.momentum)
                self.running_var.lerp_(
                    (var.squeeze(1) * (batch_size / (batch_size - 1))).float(),
                    self.momentum,
                )
            x = (x - mean) * torch.rsqrt(var + self.eps)
        else:
            x = (x - self.running_mean.unsqueeze(1)) * torch.rsqrt(
                self.running_var.unsqueeze(1) + self.eps
            )
        return x * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)

    def normalize_parameters(self):
        scale, bias = self.weight.data, self.bias.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class EnsembleUnitRMSNorm(nn.Module):
    def __init__(self, num_ensemble, input_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight.unsqueeze(1)

    def normalize_parameters(self):
        scale = self.weight.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)


class EnsembleFlashSACEmbedder(nn.Module):
    def __init__(self, num_ensemble, input_dim, hidden_dim):
        super().__init__()
        self.norm = EnsembleUnitBatchNorm(num_ensemble, input_dim)
        self.w = EnsembleUnitLinear(num_ensemble, input_dim, hidden_dim)

    def forward(self, x, training):
        x = self.norm(x, training=training)
        return self.w(x)


class EnsembleFlashSACBlock(nn.Module):
    def __init__(self, num_ensemble, hidden_dim, expansion=4):
        super().__init__()
        self.w1 = EnsembleUnitLinear(num_ensemble, hidden_dim, hidden_dim * expansion)
        self.w2 = EnsembleUnitLinear(num_ensemble, hidden_dim * expansion, hidden_dim)
        self.norm1 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim * expansion)
        self.norm2 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim)

    def forward(self, x, training):
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = F.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        x = F.relu(x)
        return x + residual


class EnsembleCategoricalValue(nn.Module):
    def __init__(self, num_ensemble, hidden_dim, num_bins, min_v, max_v):
        super().__init__()
        self.w = EnsembleUnitLinear(num_ensemble, hidden_dim, num_bins)
        self.bias = nn.Parameter(torch.zeros(num_ensemble, num_bins))
        self.register_buffer(
            "bin_values",
            torch.linspace(
                start=min_v,
                end=max_v,
                steps=num_bins,
                dtype=torch.float32,
            ).reshape(1, 1, -1),
        )

    def forward(self, x, training):
        del training
        logits = self.w(x) + self.bias.unsqueeze(1)
        log_prob = F.log_softmax(logits, dim=-1)
        value = torch.sum(torch.exp(log_prob) * self.bin_values, dim=-1)
        return value, {"log_prob": log_prob}


# Actor, critic, and temperature networks


class FlashSACActor(nn.Module):
    def __init__(self, num_blocks, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.embedder = FlashSACEmbedder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.encoder = nn.ModuleList(
            [FlashSACBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = NormalTanhPolicy(
            hidden_dim=hidden_dim, action_dim=action_dim
        )

    def get_mean_and_std(self, observations, training):
        x = self.embedder(observations, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        return self.predictor.get_mean_and_std(x, training)

    def forward(self, observations, training):
        x = self.embedder(observations, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        return self.predictor(x, training)


class FlashSACDoubleCritic(nn.Module):
    """Fused double categorical critic used for clipped double Q-learning."""

    def __init__(
        self,
        num_blocks,
        input_dim,
        hidden_dim,
        num_bins,
        min_v,
        max_v,
        num_qs=2,
    ):
        super().__init__()
        self.num_qs = num_qs
        self.embedder = EnsembleFlashSACEmbedder(
            num_qs, input_dim, hidden_dim
        )
        self.encoder = nn.ModuleList(
            [
                EnsembleFlashSACBlock(num_qs, hidden_dim)
                for _ in range(num_blocks)
            ]
        )
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)
        self.predictor = EnsembleCategoricalValue(
            num_ensemble=num_qs,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
        )

    def forward(self, observations, actions, training):
        x = torch.cat((observations, actions), dim=-1)
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1)
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        return self.predictor(x, training)


class FlashSACTemperature(nn.Module):
    def __init__(self, initial_value=0.01):
        super().__init__()
        self.log_temp = nn.Parameter(
            torch.tensor([math.log(initial_value)], dtype=torch.float32)
        )

    def forward(self):
        return torch.exp(self.log_temp)


# Update functions


def compile_update_helpers():
    """Compile categorical helpers with torch.compile's default mode.

    The public implementation decorates these functions with bare
    ``@torch.compile``. In particular, they must not inherit the network's
    ``reduce-overhead`` mode: doing so creates a separate CUDA Graph that can
    consume an already-reused output buffer from the compiled target critic.
    """
    global _select_min_q_log_probs
    global _compute_categorical_td_target
    if not hasattr(torch, "compile"):
        raise RuntimeError("Compiled FlashSAC updates require PyTorch 2.0 or newer")
    if getattr(_compute_categorical_td_target, "_flashsac_compiled", False):
        return
    _select_min_q_log_probs = torch.compile(_select_min_q_log_probs)
    _compute_categorical_td_target = torch.compile(_compute_categorical_td_target)
    _select_min_q_log_probs._flashsac_compiled = True
    _compute_categorical_td_target._flashsac_compiled = True


def _select_min_q_log_probs(next_qs, next_q_log_probs):
    num_bins = next_q_log_probs.shape[-1]
    min_indices = next_qs.argmin(dim=0)
    return torch.gather(
        next_q_log_probs,
        dim=0,
        index=min_indices[None, :, None].expand(1, -1, num_bins),
    )[0]


def _compute_categorical_td_target(
    target_log_probs,
    reward,
    done,
    actor_entropy,
    gamma,
    num_bins,
    min_v,
    max_v,
):
    batch_size = reward.shape[0]
    reward = reward.reshape(-1, 1)
    done = done.reshape(-1, 1)
    actor_entropy = actor_entropy.reshape(-1, 1)

    bin_width = (max_v - min_v) / (num_bins - 1)
    bin_values = torch.linspace(
        min_v,
        max_v,
        num_bins,
        device=target_log_probs.device,
        dtype=target_log_probs.dtype,
    ).view(1, -1)
    target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
    target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

    b = (target_bin_values - min_v) / bin_width
    lower = torch.floor(b).long()
    upper = torch.clamp(lower + 1, 0, num_bins - 1)
    frac = b - lower.float()

    source_probs = target_log_probs.exp()
    target_probs = torch.zeros(
        batch_size,
        num_bins,
        dtype=source_probs.dtype,
        device=source_probs.device,
    )
    target_probs.scatter_add_(1, lower, source_probs * (1.0 - frac))
    target_probs.scatter_add_(1, upper, source_probs * frac)
    return target_probs


def update_actor(
    actor,
    critic,
    temperature,
    optimizer,
    scheduler,
    batch,
    bc_alpha,
    device,
    use_amp,
    grad_scaler,
):
    with autocast(device=device, enabled=use_amp):
        actor_obs_all = torch.cat(
            [batch["actor_observation"], batch["actor_next_observation"]], dim=0
        )
        actions_all, info = actor(observations=actor_obs_all, training=True)
        actions = torch.chunk(actions_all, 2, dim=0)[0]
        log_probs = torch.chunk(info["log_prob"], 2, dim=0)[0]

        critic.requires_grad_(False)
        qs, _ = critic(
            observations=batch["observation"], actions=actions, training=False
        )
        q = torch.minimum(qs[0], qs[1])
        critic.requires_grad_(True)

        temp_value = temperature().detach()
        actor_loss = (log_probs * temp_value - q).mean()
        if bc_alpha > 0:
            q_abs = torch.abs(q).mean().detach()
            bc_loss = ((actions - batch["action"]) ** 2).mean()
            actor_loss = actor_loss + bc_alpha * q_abs * bc_loss

        entropy = -log_probs.mean()
        mean_action = actions.mean()

    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        grad_scaler.scale(actor_loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        actor_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    normalize_module_parameters(actor)

    return {
        "actor_loss": actor_loss.detach(),
        "actor_entropy": entropy.detach(),
        "actor_mean_action": mean_action.detach(),
    }


def update_critic(
    actor,
    critic,
    target_critic,
    temperature,
    optimizer,
    scheduler,
    batch,
    min_v,
    max_v,
    num_bins,
    gamma,
    n_step,
    device,
    use_amp,
    grad_scaler,
):
    with autocast(device=device, enabled=use_amp):
        with torch.no_grad():
            next_actions, info = actor(
                observations=batch["actor_next_observation"], training=False
            )
            next_actions = next_actions.clone()
            next_actor_log_probs = info["log_prob"].clone()
            next_actor_entropy = temperature() * next_actor_log_probs

            obs_all = torch.cat(
                [batch["observation"], batch["next_observation"]], dim=0
            )
            act_all = torch.cat([batch["action"], next_actions], dim=0)
            qs_all, q_infos_all = target_critic(
                observations=obs_all, actions=act_all, training=True
            )
            next_qs = qs_all.chunk(2, dim=1)[1]
            next_q_log_probs = q_infos_all["log_prob"].chunk(2, dim=1)[1]
            next_q_log_probs = _select_min_q_log_probs(
                next_qs, next_q_log_probs
            )
            target_probs = _compute_categorical_td_target(
                target_log_probs=next_q_log_probs,
                reward=batch["reward"],
                done=batch["terminated"],
                actor_entropy=next_actor_entropy,
                gamma=gamma**n_step,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
            )
            max_entropy_bonus = next_actor_entropy.max()

        _, pred_q_infos = critic(
            observations=obs_all, actions=act_all, training=True
        )
        pred_log_probs = pred_q_infos["log_prob"].chunk(2, dim=1)[0]
        critic_loss = -(
            target_probs.unsqueeze(0) * pred_log_probs
        ).sum(dim=-1).mean()

    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        grad_scaler.scale(critic_loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        critic_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    normalize_module_parameters(critic)

    return {
        "critic_loss": critic_loss.detach(),
        "critic_max_entropy_bonus": max_entropy_bonus.detach(),
    }


def update_temperature(
    temperature,
    optimizer,
    scheduler,
    entropy,
    target_entropy,
):
    temperature_value = temperature().clone()
    temperature_loss = temperature_value * (
        entropy.detach() - target_entropy
    ).mean()
    optimizer.zero_grad(set_to_none=True)
    temperature_loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {
        "temperature_value": temperature_value.detach(),
        "temperature_loss": temperature_loss.detach(),
    }


@torch.no_grad()
def update_target_network(target_critic, critic, tau):
    ema_update_module(target_critic, critic, tau)

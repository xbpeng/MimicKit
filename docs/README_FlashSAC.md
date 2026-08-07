# FlashSAC

[FlashSAC](https://holiday-robot.github.io/FlashSAC/) is a high-throughput,
off-policy actor-critic algorithm for high-dimensional robot control. This
integration follows the authors' public PyTorch implementation at commit
`87edc9061150ae9e962dd84e6544e27a1554b3ab` and adapts it to MimicKit's
vector-environment and checkpoint lifecycle.

The first integration supports a single MimicKit process/device, one-dimensional
Box observations, and finite-bounded one-dimensional Box actions. Multi-process
training is rejected with a clear error instead of running unsynchronized agents.

## Training DeepMimic G1

Install the simulator and MimicKit dependencies, then run:

```bash
python mimickit/run.py \
    --arg_file args/deepmimic_g1_flashsac_args.txt \
    --mode train \
    --visualize false
```

The example is a memory-reduced DeepMimic preset, not a claim that every value
reproduces a particular FlashSAC paper benchmark. It uses two FlashSAC update
steps (critic updates) per 1024 environment transitions, `n_step: 1`, a
1M-transition replay buffer, and a 100K-transition random/replay warmup. With
`actor_update_period: 2`, the actor and temperature update every second critic
step. The official IsaacLab launch script uses a larger 10M buffer and
`n_step: 3`. The example's 50,069,504-sample budget is aligned to a complete
4096-environment, 32-step MimicKit rollout boundary.

Replay memory scales approximately with
`buffer_max_length * (2 * observation_dim + action_dim + 3) * 4` bytes. For
the G1 DeepMimic observation dimension of 849, the 1M example buffer uses about
6.5 GiB before simulator and optimizer memory. On a smaller GPU, reduce
`buffer_max_length` or set `buffer_device: "cpu"`; either choice changes the
memory/throughput tradeoff and should be kept fixed in performance comparisons.
As in the public PyTorch replay buffer, CPU storage is page-locked when CUDA is
available, so it still requires sufficient host memory for the full capacity.

FlashSAC samples uniformly random actions until the replay buffer is ready. The
transition that reaches `buffer_min_length` enables optimizer updates, and the
next environment step uses the policy. This matches the public training loop,
so `buffer_min_length` controls the common warmup boundary.

## MimicKit adapters

- The actor and critic operate on unit actions in `[-1, 1]`; MimicKit's action
  normalizer maps policy actions to each environment's finite Box bounds.
- `FAIL` and `SUCC` are terminal states. `TIME` is a truncation, so the critic
  bootstraps from the terminal observation.
- Raw observations go directly to FlashSAC's own input BatchNorm. MimicKit's
  external observation normalizer is intentionally not applied.
- Optimizer updates are accumulated as `updates_per_env_step * num_envs`, which
  keeps the update-to-data ratio stable when the vector-environment count changes.
- In-training evaluation resets the same MimicKit environment. The integration
  drops only pending n-step transitions and per-episode reward returns at that
  boundary; stored replay data and running reward statistics are retained.
- A simulator row with a non-finite observation/reward, or an observation above
  the configured `observation_max_abs`, is reported as a failed terminal
  transition and reset. Finite pre-step data and rewards are preserved; only
  the invalid output is replaced with a finite fallback. The per-iteration
  `Nonfinite_Env_Count` and `Outlier_Env_Count` metrics make recovery visible;
  both count row events, so repeated recovery of one environment is counted
  repeatedly. Set `observation_max_abs` to a bound appropriate for the
  environment, or omit it to disable finite-value outlier detection.

Because raw observations are part of the reference algorithm, checkpoints from
older local FlashSAC ports that trained with MimicKit observation normalization
are not input-compatible with this implementation and require fresh training.
Versioned MimicKit FlashSAC checkpoints declare this preprocessing contract;
the loader rejects unversioned legacy wrapper checkpoints instead of silently
running them with different inputs.

The example keeps `use_compile: false` as the validated compatibility path.
Setting it to `true` compiles network forwards, weight projection/target EMA,
and categorical target helpers following the public implementation's
performance strategy. Compilation requires PyTorch 2.0 or newer and is
version-sensitive, so validate it over multiple train/evaluation boundaries on
the target PyTorch and simulator stack before relying on it. The official
Python 3.11 IsaacLab environment pins PyTorch 2.9.1. `use_mixed_precision` uses
MimicKit's existing agent configuration key and currently requires CUDA.
`use_tf32: true` enables the CUDA TF32 settings used by the reference training
entry point.

Replay-buffer checkpointing is opt-in because replay files can be large. Set
`save_replay_buffer: true` to write `<model>.replay.pt`, or set
`load_replay_buffer` to a replay sidecar path when resuming.

Resume model and optimizer state by passing `--model_file <model.pt>`. When
`load_optimizer: true`, `max_samples` is a cumulative target that includes the
restored environment-step count, and intermediate checkpoint iteration numbers
continue after the loaded checkpoint instead of starting again at zero. To
resume replay as well, set `load_replay_buffer` to the matching
`<model.pt>.replay.pt` sidecar; simulator state and in-flight n-step transitions
are intentionally not restored. Checkpoints also record whether reward
normalization was enabled. Loading with a different `normalize_reward` setting
is rejected because it would silently change the resumed algorithm; older
version-1 checkpoints without this metadata load with a compatibility warning.

## Tests

The FlashSAC tests use only the standard library plus MimicKit dependencies:

```bash
python -m unittest discover -s tests -p "test_flashsac*.py"
```

## Citation

```bibtex
@article{kim2026flashsac,
  title={FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control},
  author={Kim, Donghu and Lee, Youngdo and Park, Minho and Kim, Kinam and Nahendra, I Made Aswin and Seno, Takuma and Min, Sehee and Palenicek, Daniel and Vogt, Florian and Kragic, Danica and Peters, Jan and Choo, Jaegul and Lee, Hojoon},
  journal={arXiv preprint arXiv:2604.04539},
  year={2026}
}
```

The adapted implementation is distributed under the original MIT terms in
[`LICENSES/FlashSAC-MIT.txt`](../LICENSES/FlashSAC-MIT.txt).

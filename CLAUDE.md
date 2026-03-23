# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MimicKit is a reinforcement learning framework for motion imitation and control. It trains physics-based character controllers to imitate reference motion clips using algorithms like DeepMimic, AMP, AWR, ASE, LCP, and ADD. The main entry point is `mimickit/run.py`.

## Running the Code

All commands are run from the repo root. The `mimickit/` directory must be on the Python path (imports are relative to it, e.g. `import envs.env_builder`).

**Train a model:**
```
python mimickit/run.py --mode train --num_envs 4096 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/deepmimic_humanoid_env.yaml --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml --out_dir output/
```

**Using an arg file (preferred):**
```
python mimickit/run.py --arg_file args/deepmimic_humanoid_ppo_args.txt --visualize true
```

**Test a trained model:**
```
python mimickit/run.py --arg_file args/deepmimic_humanoid_ppo_args.txt --num_envs 4 --visualize true --mode test --model_file data/models/deepmimic_humanoid_spinkick_model.pt
```

**Visualize a motion clip:**
```
python mimickit/run.py --mode test --arg_file args/view_motion_humanoid_args.txt --visualize true
```

**Multi-GPU distributed training:**
```
python mimickit/run.py --arg_file args/deepmimic_humanoid_ppo_args.txt --devices cuda:0 cuda:1
```

**View training logs:**
```
tensorboard --logdir=output/ --port=6006 --samples_per_plugin scalars=999999
```
Or use `tools/plot_log/plot_log.py` for the `log.txt` output file.

## Architecture

### Three-Layer Config System
Everything is driven by three YAML config files:
- **Engine config** (`data/engines/*.yaml`): selects simulator backend (Isaac Gym, Isaac Lab, Newton)
- **Env config** (`data/envs/*.yaml`): selects environment type and parameters (motion file, character asset, rewards)
- **Agent config** (`data/agents/*.yaml`): selects RL algorithm and hyperparameters

The `env_config` can include an `engine` section to override engine parameters for that specific environment.

### Simulator Backends (`mimickit/engines/`)
`Engine` is the abstract base class. Three concrete implementations: `IsaacGymEngine`, `IsaacLabEngine`, `NewtonEngine`. The engine abstraction handles physics simulation, object creation, joint control (position/velocity/torque/PD), and rendering. Engines are not imported at module load — they're imported lazily to avoid conflicts between simulators.

### Environment Hierarchy (`mimickit/envs/`)
- `BaseEnv` → `SimEnv` (adds physics engine) → `CharEnv` (adds character/robot) → `DeepMimicEnv` / `AMPEnv` / `ASEEnv` / `ADDEnv`
- Task environments (`TaskLocationEnv`, `TaskSteeringEnv`) extend the motion imitation envs with goal-directed objectives.
- `env_builder.py` instantiates the correct class based on `env_name` in the env config.

### Learning Algorithms (`mimickit/learning/`)
- `BaseAgent` (extends `torch.nn.Module`) handles training loop, experience buffer, normalization, logging, and model checkpointing.
- Algorithm agents: `PPOAgent`, `AWRAgent`, `AMPAgent`, `ASEAgent`, `ADDAgent`, `LCPAgent`
- Each agent has a corresponding model file (e.g., `ppo_model.py`) and uses `nets/` for neural network architectures.
- `agent_builder.py` instantiates the correct agent based on `agent_name` in the agent config.
- Distributed training uses `mp_util.py` and `mp_optimizer.py` via PyTorch multiprocessing (spawn method).

### Animation System (`mimickit/anim/`)
- `Motion` class (`motion.py`): stores motion clips as `.pkl` files; each frame is `[root_pos(3), root_rot(3), joint_rots...]` using 3D exponential maps (1D for hinge joints).
- `MotionLib` (`motion_lib.py`): loads and manages datasets of multiple motion clips.
- Character models: `MjcfCharModel` (MuJoCo XML), `UrdfCharModel` (URDF), `UsdCharModel` (USD), `KinCharModel` (kinematic).

### Data Layout
- `data/engines/` — engine YAML configs
- `data/envs/` — environment YAML configs
- `data/agents/` — agent YAML configs
- `data/assets/` — character/robot mesh and joint definition files (XML/URDF/USD)
- `data/motions/` — reference motion clips (`.pkl`)
- `data/datasets/` — dataset files listing multiple motion clips
- `data/models/` — pretrained model checkpoints (`.pt`)
- `args/` — pre-composed argument files for common experiment configurations

## Motion Data Format

Each frame: `[root_pos(3D), root_rot(3D), joint_rots...]`
Rotations use 3D exponential maps for ball joints, 1D angles for hinge joints.
Joint order follows depth-first traversal of the kinematic tree as defined in the character's XML file.

## Motion Retargeting Tools
- `tools/gmr_to_mimickit/` — convert GMR retargeting output to MimicKit format
- `tools/smpl_to_mimickit/` — convert AMASS/SMPL motion files to MimicKit format

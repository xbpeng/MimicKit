# MimicKit Codebase Overview

## Project Summary
MimicKit is a reinforcement learning framework for physics-based motion imitation and character control. It provides clean, lightweight implementations of several motion imitation algorithms, designed to train controllers that make virtual characters imitate reference motions.

## Core Algorithms Implemented
1. **DeepMimic** - Example-guided deep RL for physics-based character skills
2. **AMP (Adversarial Motion Priors)** - Adversarial learning approach for stylized character control
3. **ADD (Adversarial Differential Discriminators)** - Physics-based motion imitation with differential discriminators
4. **ASE** - Large-scale reusable adversarial skill embeddings
5. **RL Algorithms**: PPO, AWR

## Directory Structure

```
MimicKit/
├── mimickit/                          # Main package
│   ├── run.py                         # Entry point for training/testing
│   ├── anim/                          # Animation and kinematics
│   │   ├── motion.py                  # Motion representation class
│   │   ├── motion_lib.py              # Motion library for loading/querying motions
│   │   └── kin_char_model.py          # Kinematic character model
│   ├── engines/                       # Physics simulation engines
│   │   ├── engine.py                  # Base engine interface
│   │   └── isaac_gym_engine.py        # Isaac Gym physics engine implementation
│   ├── envs/                          # Environment implementations
│   │   ├── base_env.py                # Base environment interface
│   │   ├── sim_env.py                 # Simulation environment base class
│   │   ├── char_env.py                # Character environment (loads character URDF/XML)
│   │   ├── deepmimic_env.py           # DeepMimic environment
│   │   ├── amp_env.py                 # AMP environment (adds discriminator observations)
│   │   ├── add_env.py                 # ADD environment (extends AMP)
│   │   ├── ase_env.py                 # ASE environment
│   │   ├── view_motion_env.py         # For visualizing reference motions
│   │   ├── env_builder.py             # Factory for building environments
│   │   └── [other specialized envs]
│   ├── learning/                      # Learning algorithms
│   │   ├── base_agent.py              # Base agent interface
│   │   ├── ppo_agent.py               # PPO agent implementation
│   │   ├── amp_agent.py               # AMP agent (extends PPO, adds discriminator)
│   │   ├── add_agent.py               # ADD agent (extends AMP)
│   │   ├── ase_agent.py               # ASE agent
│   │   ├── awr_agent.py               # AWR agent
│   │   ├── base_model.py              # Base neural network model
│   │   ├── ppo_model.py               # PPO model (actor-critic)
│   │   ├── amp_model.py               # AMP model (adds discriminator)
│   │   ├── add_model.py               # ADD model
│   │   ├── agent_builder.py           # Factory for building agents
│   │   ├── experience_buffer.py       # Stores experience for training
│   │   ├── normalizer.py              # Input normalization
│   │   ├── diff_normalizer.py         # Difference-based normalization (for ADD)
│   │   └── nets/                      # Neural network architectures
│   ├── util/                          # Utility functions
│   │   ├── arg_parser.py              # Command-line argument parsing
│   │   ├── torch_util.py              # PyTorch utilities (quaternions, rotations)
│   │   ├── math_util.py               # Math utilities
│   │   ├── logger.py                  # Logging utilities
│   │   └── [other utilities]
│   └── __init__.py
├── data/                              # Data files
│   ├── assets/                        # Character definitions (URDF/XML)
│   │   ├── g1/                        # G1 humanoid robot
│   │   ├── humanoid/                  # Standard humanoid
│   │   └── [other characters]
│   ├── motions/                       # Reference motion clips
│   │   ├── g1/
│   │   │   ├── g1_walk.pkl
│   │   │   ├── g1_spinkick.pkl
│   │   │   ├── g1_cartwheel.pkl
│   │   │   ├── g1_run.pkl
│   │   │   └── g1_speed_vault.pkl
│   │   └── [other motion files]
│   ├── agents/                        # Agent configuration files
│   ├── envs/                          # Environment configuration files
│   ├── models/                        # Pretrained models
│   └── logs/                          # Training logs
├── args/                              # Argument configuration files
│   ├── amp_g1_args.txt                # AMP training args for G1
│   ├── add_g1_args.txt                # ADD training args for G1
│   └── [other arg files]
└── output/                            # Training outputs
```

---

## Key Components

### 1. Physics Simulation Engine (`mimickit/engines/`)

**File**: `/home/ultron/team_files/emma/MimicKit/mimickit/engines/isaac_gym_engine.py`

- Wraps NVIDIA's Isaac Gym physics simulator
- Manages multiple parallel environments
- Key methods:
  - `step()` - Simulate one timestep across all environments
  - `set_apply_forces_callback()` - Register callback for applying external forces
  - `get_root_pos()`, `get_root_rot()`, etc. - Query character state
  - `set_dof_state()` - Set joint positions/velocities
  - Control modes: position control, velocity control, torque control

**Important for random forces**:
- `_apply_forces_callback` (line 50) - Callback mechanism for applying forces
- `_apply_cmd()` (line 79) - Called before physics step
- Called in `step()` method (lines 517-522)

### 2. Environment Hierarchy

#### BaseEnv (`mimickit/envs/base_env.py`)
- Abstract base class defining environment interface
- Methods: `reset()`, `step()`, `get_obs_space()`, `get_action_space()`

#### SimEnv (`mimickit/envs/sim_env.py`)
- Manages physics simulation and environment stepping
- Key methods:
  - `step(action)` → `_pre_physics_step()` → `_physics_step()` → `_post_physics_step()`
  - `_apply_action()` - Apply actions to character (abstract)
  - `_update_reward()` - Compute rewards (abstract)
  - `_update_done()` - Check termination conditions (abstract)

#### CharEnv (`mimickit/envs/char_env.py`)
- Loads character from URDF/XML file
- Initializes kinematic model
- Handles character setup in simulation

#### DeepMimicEnv (`mimickit/envs/deepmimic_env.py`)
- Implements motion imitation task
- Loads motion clips from files
- Computes rewards based on pose/velocity tracking
- Tracks reference character state

#### AMPEnv (`mimickit/envs/amp_env.py`)
- Extends DeepMimicEnv for AMP training
- Maintains historical discriminator observations (circular buffers)
- Key buffers:
  - `_disc_hist_root_pos`, `_disc_hist_root_rot`, `_disc_hist_root_vel`, etc.
  - Stores last N timesteps of state for discriminator

#### ADDEnv (`mimickit/envs/add_env.py`)
- Extends AMPEnv for ADD training
- Adds demo observation buffer: `_disc_obs_demo_buf`
- Computes discriminator observations for both agent and demo

### 3. Learning Agents

#### Base Agent (`mimickit/learning/base_agent.py`)
- Abstract agent interface
- Core training loop: `train_model()`

#### PPO Agent (`mimickit/learning/ppo_agent.py`)
- Proximal Policy Optimization implementation
- Actor-critic architecture
- Key methods:
  - `train_model()` - Main training loop
  - `_record_data_post_step()` - Store experience
  - `_build_train_data()` - Prepare training batch
  - `_compute_rewards()` - Compute task rewards

#### AMP Agent (`mimickit/learning/amp_agent.py`)
- Extends PPO with adversarial discriminator
- Combines task reward + discriminator reward
- Key method:
  - `_compute_disc_loss()` - Compute discriminator loss
  - Rewards: `task_reward_weight` + `disc_reward_weight`

#### ADD Agent (`mimickit/learning/add_agent.py`)
- Extends AMP agent
- Uses differential rewards instead of absolute
- Normalizes observation differences
- Key class: `DiffNormalizer` for handling obs differences

### 4. Neural Network Models

#### BaseModel (`mimickit/learning/base_model.py`)
- Wrapper around actor, critic, and other networks

#### PPOModel (`mimickit/learning/ppo_model.py`)
- Actor network (policy)
- Critic network (value function)

#### AMPModel (`mimickit/learning/amp_model.py`)
- Adds discriminator network
- Binary classifier: real motion vs agent motion

#### ADDModel (`mimickit/learning/add_model.py`)
- Extends AMP model for ADD

**Network architectures** in `mimickit/learning/nets/`:
- `fc_2layers_1024units.py` - 2-layer fully connected (1024 units)
- `fc_3layers_1024units.py` - 3-layer fully connected
- `cnn_3conv_1fc_0.py` - CNN-based architecture

---

## G1 Robot Configuration

### Character Definition
**File**: `/home/ultron/team_files/emma/MimicKit/data/assets/g1/g1.xml` (305 lines, MuJoCo format)

### Upper Body Structure (relevant for random force application)

```
Torso (mass=9.598 kg)
├── Waist Yaw Joint (axis: Z)
│   └── Waist Roll Joint (axis: X)
│       └── Waist Pitch Joint (axis: Y)
│           └── Torso Link
│               ├── Head Link (sphere geom, size=0.06)
│               ├── Left Shoulder Chain
│               │   ├── left_shoulder_pitch_joint (axis: Y, range: -177 to 153 deg)
│               │   ├── left_shoulder_roll_joint (axis: X)
│               │   ├── left_shoulder_yaw_joint (axis: Z)
│               │   ├── left_elbow_joint
│               │   └── left wrist chain
│               └── Right Shoulder Chain (symmetric)
│                   ├── right_shoulder_pitch_joint
│                   ├── right_shoulder_roll_joint
│                   ├── right_shoulder_yaw_joint
│                   ├── right_elbow_joint
│                   └── right wrist chain
```

### Key Upper Body Bodies (for force application)
- `torso_link` - Main torso (9.598 kg, position tracked)
- `head_link` - Head (collision sphere, size 0.06m)
- `left_shoulder_pitch_link` - Left shoulder (mass 0.718 kg)
- `right_shoulder_pitch_link` - Right shoulder (mass 0.718 kg)
- `left_elbow_link` - Left elbow (mass 0.6 kg)
- `right_elbow_link` - Right elbow (mass 0.6 kg)

### G1 Configuration Files

**Environment Config**: `/home/ultron/team_files/emma/MimicKit/data/envs/amp_g1_env.yaml`
```yaml
env:
  char_file: "data/assets/g1/g1.xml"
  episode_length: 10.0 seconds
  key_bodies: ["left_ankle_roll_link", "right_ankle_roll_link", "head_link", 
               "left_wrist_yaw_link", "right_wrist_yaw_link"]
  contact_bodies: ["left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
                   "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link"]
```

**Agent Config**: `/home/ultron/team_files/emma/MimicKit/data/agents/amp_g1_agent.yaml`
```yaml
agent_name: "AMP"
model:
  actor_net: "fc_2layers_1024units"
  critic_net: "fc_2layers_1024units"
  disc_net: "fc_2layers_1024units"
disc_reward_weight: 1.0
task_reward_weight: 0.0
```

### Available Motions for G1
- `g1_walk.pkl` - Walking motion
- `g1_spinkick.pkl` - Spin kick motion
- `g1_cartwheel.pkl` - Cartwheel motion
- `g1_run.pkl` - Running motion
- `g1_speed_vault.pkl` - Speed vault motion

---

## ADD Training Implementation

### ADD Environment (`mimickit/envs/add_env.py`)
Extends AMP environment with:
1. Demo observation buffer: `_disc_obs_demo_buf`
2. Computes discriminator observations for reference motion
3. Key method: `_update_disc_obs_demo()` - updates demo observations

### ADD Agent (`mimickit/learning/add_agent.py`)
- Uses `DiffNormalizer` instead of regular `Normalizer`
- Computes reward from observation differences:
  ```
  obs_diff = disc_obs_demo - disc_obs
  reward = discriminator(normalized(obs_diff))
  ```
- Training loss includes:
  - Positive logits for zero observation difference
  - Negative logits for actual observation differences
  - Gradient penalty on discriminator
  - Weight decay

### ADD Model (`mimickit/learning/add_model.py`)
Discriminator trained on:
- Positive: zero (or near-zero) observation difference
- Negative: actual difference between demo and agent observations

---

## How Forces Are Currently Applied

### Current Force Application Mechanism

1. **Callback-based approach** in Isaac Gym Engine:
   - `set_apply_forces_callback()` registers a Python callback
   - Called during `_apply_cmd()` in physics step loop
   - Location: `/home/ultron/team_files/emma/MimicKit/mimickit/engines/isaac_gym_engine.py`, lines 517-522

2. **No current random force implementation visible** in codebase
   - Search for "random.*force" or "force.*random" returns no results
   - Forces appear to be applied only through action/control

3. **Action Application** in environments:
   - Implemented in subclasses' `_apply_action()` method
   - Currently applies motor torques to joints based on policy output
   - No external perturbation forces implemented by default

---

## Where to Add Random Forces to Upper Body During Training

### Recommended Implementation Locations

#### Option 1: In DeepMimicEnv / AMPEnv / ADDEnv (Recommended)
**File**: `/home/ultron/team_files/emma/MimicKit/mimickit/envs/deepmimic_env.py` or subclasses

Add a method to apply random forces:
```python
def _apply_random_forces(self, env_ids=None):
    # Sample random forces for upper body
    # Apply via engine callback or direct force application
    pass

def _post_physics_step(self):
    # Override from SimEnv
    self._apply_random_forces()
    super()._post_physics_step()
```

**Upper body target bodies**:
- `torso_link` (main mass)
- `head_link` (for balance perturbation)
- `left_shoulder_pitch_link`, `right_shoulder_pitch_link`

#### Option 2: In Isaac Gym Engine
**File**: `/home/ultron/team_files/emma/MimicKit/mimickit/engines/isaac_gym_engine.py`

Enhance force callback mechanism:
```python
def _apply_forces_callback(self):
    # Apply random forces to selected bodies
    # Use self._gym.apply_rigid_body_force_at_pos_tensor()
    pass
```

#### Option 3: Add as Environment Configuration
Create parameters in YAML config files:
```yaml
env:
  enable_random_force: true
  random_force_scale: [0.5, 0.5, 0.5]  # x, y, z components
  random_force_bodies: ["torso_link", "head_link"]
  random_force_probability: 0.5
```

### Integration Points

1. **Configuration**: Add `enable_random_force`, `force_scale`, etc. to `env_config`
2. **Initialization**: In environment `__init__()`, setup force parameters
3. **Execution**: In environment `step()` or `_post_physics_step()`, sample and apply forces
4. **State tracking**: Store force values in buffers for logging/analysis

---

## Training Workflow

### High-level Training Flow

```
run.py (entry point)
  ↓
load_args() - Parse command-line arguments
  ↓
build_env() - Create training environment(s)
  ↓
build_agent() - Create learning agent
  ↓
agent.train_model() - Main training loop
  ├─ For each iteration:
  │  ├─ Collect experience: agent.step() → env.step()
  │  ├─ Compute rewards: env._update_reward()
  │  ├─ Build training batch
  │  ├─ Train networks (actor, critic, discriminator)
  │  └─ Log statistics
  └─ Save model checkpoint
```

### Per-Step Physics Simulation

```
env.step(action)
  ↓
_pre_physics_step(action)
  ├─ _apply_action(action) - Apply motor commands
  └─ [Apply random forces here]
  ↓
_physics_step()
  ├─ engine.step() - Run physics
  │  ├─ _apply_cmd()
  │  ├─ _pre_sim_step()
  │  ├─ _sim_step() (multiple substeps)
  │  └─ _refresh_sim_tensors()
  └─ [Forces applied during physics substeps]
  ↓
_post_physics_step()
  ├─ _update_time()
  ├─ _update_misc()
  ├─ _update_observations()
  ├─ _update_info()
  ├─ _update_reward()
  └─ _update_done()
  ↓
return obs, reward, done, info
```

---

## Key Files Summary

| Path | Purpose |
|------|---------|
| `/home/ultron/team_files/emma/MimicKit/mimickit/run.py` | Entry point, orchestrates training |
| `/home/ultron/team_files/emma/MimicKit/mimickit/engines/isaac_gym_engine.py` | Physics simulation wrapper |
| `/home/ultron/team_files/emma/MimicKit/mimickit/envs/deepmimic_env.py` | Base motion imitation environment |
| `/home/ultron/team_files/emma/MimicKit/mimickit/envs/amp_env.py` | AMP-specific environment |
| `/home/ultron/team_files/emma/MimicKit/mimickit/envs/add_env.py` | ADD-specific environment |
| `/home/ultron/team_files/emma/MimicKit/mimickit/learning/ppo_agent.py` | PPO training agent |
| `/home/ultron/team_files/emma/MimicKit/mimickit/learning/amp_agent.py` | AMP training agent |
| `/home/ultron/team_files/emma/MimicKit/mimickit/learning/add_agent.py` | ADD training agent |
| `/home/ultron/team_files/emma/MimicKit/data/assets/g1/g1.xml` | G1 robot definition |
| `/home/ultron/team_files/emma/MimicKit/data/envs/amp_g1_env.yaml` | G1 AMP config |
| `/home/ultron/team_files/emma/MimicKit/data/agents/amp_g1_agent.yaml` | AMP agent config |

---

## Configuration and Arguments

Training is controlled via YAML config files and command-line arguments.

**Example training command**:
```bash
python mimickit/run.py \
  --mode train \
  --num_envs 4096 \
  --env_config data/envs/amp_g1_env.yaml \
  --agent_config data/agents/amp_g1_agent.yaml \
  --visualize false \
  --out_model_file output/g1_model.pt \
  --log_file output/log.txt
```

**Key argument files**:
- `args/amp_g1_args.txt` - AMP training for G1
- `args/add_g1_args.txt` - ADD training for G1

---

## Summary

**MimicKit** is a well-structured RL framework for motion imitation with:
- Clean separation between physics engine, environments, and learning agents
- Support for multiple imitation learning methods (DeepMimic, AMP, ADD, ASE)
- Modular design allowing easy extension
- Full support for G1 humanoid robot
- Isaac Gym physics simulation for parallelized training

**For adding random forces to the upper body**:
1. Best location: Extend environment classes (DeepMimicEnv or ADDEnv)
2. Apply forces in `step()` or `_post_physics_step()` 
3. Target bodies: `torso_link`, `head_link`, shoulder links
4. Use Isaac Gym's force application API through engine callback
5. Configure via YAML for easy experimentation

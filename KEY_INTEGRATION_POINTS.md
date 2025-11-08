# MimicKit: Key Integration Points for Random Force Addition

## Class Inheritance Hierarchy

```
BaseEnv (abstract)
  ↓
SimEnv (abstract)
  ├─ Core loop: step() → _pre_physics_step() → _physics_step() → _post_physics_step()
  ↓
CharEnv (loads character from XML)
  ↓
DeepMimicEnv (motion imitation)
  ├─ Implements: _apply_action(), _update_reward(), _update_done()
  ↓
AMPEnv (AMP training)
  ├─ Adds: discriminator observation history
  ├─ Circular buffers: _disc_hist_root_pos, _disc_hist_root_rot, _disc_hist_root_vel, etc.
  ↓
ADDEnv (ADD training)
  └─ Adds: demo observation buffer (_disc_obs_demo_buf)
```

## Agent Class Hierarchy

```
BaseAgent (abstract)
  ↓
PPOAgent
  ├─ Core training: train_model()
  ├─ Data collection: _record_data_post_step()
  ├─ Reward computation: _compute_rewards()
  ↓
AMPAgent
  ├─ Extends: adds discriminator loss computation
  ├─ Key method: _compute_disc_loss()
  ↓
ADDAgent
  └─ Extends: uses DiffNormalizer for obs differences
```

## Physics Engine Integration

```
IsaacGymEngine
  ├─ Force callback mechanism:
  │  ├─ set_apply_forces_callback(callback) [line 446]
  │  ├─ _apply_forces_callback (instance var) [line 50]
  │  └─ Called in step() → _apply_cmd() [lines 517-522]
  │
  ├─ State queries:
  │  ├─ get_root_pos(char_id)
  │  ├─ get_root_rot(char_id)
  │  ├─ get_body_pos(char_id)
  │  └─ get_contact_forces(char_id)
  │
  └─ Control methods:
     ├─ set_dof_state()
     ├─ set_root_pos()
     ├─ set_root_vel()
     └─ apply_rigid_body_force_at_pos_tensor() [for force application]
```

## Training Data Flow

```
For each training iteration:

env.step(action)
  ├─ agent.act(obs) → action (policy output)
  │
  ├─ SimEnv.step(action)
  │  ├─ _pre_physics_step(action)
  │  │  ├─ _apply_action(action) [applies motor torques]
  │  │  └─ [INSERT RANDOM FORCE HERE]
  │  │
  │  ├─ _physics_step()
  │  │  └─ engine.step() [Physics simulation]
  │  │
  │  └─ _post_physics_step()
  │     ├─ _update_time()
  │     ├─ _update_misc()
  │     ├─ _update_observations()
  │     ├─ _update_info()
  │     ├─ _update_reward()
  │     └─ _update_done()
  │
  └─ return (obs, reward, done, info)

agent._record_data_post_step(obs, reward, done, info)
  ├─ Store in experience buffer
  ├─ For AMP: record disc_obs, disc_obs_demo
  └─ For ADD: use DiffNormalizer on obs differences
```

## ADD-Specific Data Flow

```
ADDEnv.step()
  └─ _update_observations()
     ├─ _update_disc_obs() [agent observations]
     └─ _update_disc_obs_demo() [reference observations]

ADDAgent.train_model()
  └─ For each training batch:
     ├─ Get disc_obs (agent) and disc_obs_demo (reference)
     ├─ Compute: obs_diff = disc_obs_demo - disc_obs
     ├─ Normalize: norm_obs_diff = _disc_obs_norm.normalize(obs_diff)
     └─ Discriminator loss:
        ├─ Positive: reward for norm_obs_diff ≈ 0 (agent matches demo)
        ├─ Negative: penalty for large norm_obs_diff
        └─ Combined with policy loss via:
           reward = task_reward_weight * task_r + disc_reward_weight * disc_r
```

## File Organization - Training Pipeline

```
Entry Point
  └─ mimickit/run.py [lines 1-166]
     ├─ load_args() → parse command-line & config files
     ├─ build_env() → env_builder.build_env()
     │  └─ Creates environment from YAML config
     ├─ build_agent() → agent_builder.build_agent()
     │  └─ Creates agent from YAML config
     └─ agent.train_model()

Environment Creation
  └─ mimickit/envs/env_builder.py
     └─ build_env(env_file) → returns appropriate env class
        ├─ "deepmimic" → DeepMimicEnv
        ├─ "amp" → AMPEnv
        ├─ "add" → ADDEnv
        └─ etc.

Agent Creation
  └─ mimickit/learning/agent_builder.py
     └─ build_agent(agent_file) → returns appropriate agent class
        ├─ "ppo" → PPOAgent
        ├─ "amp" → AMPAgent
        ├─ "add" → ADDAgent
        └─ etc.

Config Files
  ├─ mimickit/data/envs/*.yaml [environment configs]
  └─ mimickit/data/agents/*.yaml [agent configs]
```

---

## Implementation Strategy for Random Forces

### Option 1: Add to DeepMimicEnv (affects all subclasses)

**File**: `mimickit/envs/deepmimic_env.py`

```python
class DeepMimicEnv(char_env.CharEnv):
    def __init__(self, config, num_envs, device, visualize):
        # ... existing code ...
        
        # NEW: Parse random force config
        env_config = config["env"]
        self._enable_random_force = env_config.get("enable_random_force", False)
        self._random_force_scale = env_config.get("random_force_scale", [0.0, 0.0, 0.0])
        self._random_force_bodies = env_config.get("random_force_bodies", [])
        self._random_force_prob = env_config.get("random_force_probability", 0.5)
        
        # Build body ID mapping for force application
        self._random_force_body_ids = []
        for body_name in self._random_force_bodies:
            body_id = self._engine.get_body_id(char_id, body_name)
            self._random_force_body_ids.append(body_id)
        
        # Initialize force buffer
        num_envs = self.get_num_envs()
        self._random_forces = torch.zeros([num_envs, len(self._random_force_body_ids), 3], 
                                          device=device, dtype=torch.float32)
    
    def _apply_random_forces(self):
        """Sample and apply random forces to upper body."""
        if not self._enable_random_force:
            return
        
        num_envs = self.get_num_envs()
        
        # Sample random forces with given probability
        for i, body_id in enumerate(self._random_force_body_ids):
            # Random force probability
            force_mask = torch.rand(num_envs, device=self._device) < self._random_force_prob
            
            # Sample random forces (uniform distribution)
            force_x = torch.randn(num_envs, device=self._device) * self._random_force_scale[0]
            force_y = torch.randn(num_envs, device=self._device) * self._random_force_scale[1]
            force_z = torch.randn(num_envs, device=self._device) * self._random_force_scale[2]
            
            # Apply mask
            force_x = force_x * force_mask
            force_y = force_y * force_mask
            force_z = force_z * force_mask
            
            forces = torch.stack([force_x, force_y, force_z], dim=-1)  # [num_envs, 3]
            self._random_forces[:, i, :] = forces
            
            # Apply to physics engine
            self._engine.apply_body_forces(self._get_char_id(), body_id, forces)

    def _step_sim(self):
        """Override to apply random forces before/during physics step."""
        self._apply_random_forces()
        super()._step_sim()
```

### Option 2: Add to ADDEnv (specific to ADD training)

**File**: `mimickit/envs/add_env.py`

```python
class ADDEnv(amp_env.AMPEnv):
    def __init__(self, config, num_envs, device, visualize):
        # ... existing code ...
        
        # NEW: Parse random force config
        env_config = config["env"]
        self._enable_random_force = env_config.get("enable_random_force", False)
        # ... rest of initialization ...

    def _apply_random_forces(self):
        """Apply random forces to upper body during ADD training."""
        # Implementation similar to Option 1
        pass
```

### Option 3: Configuration-Only (preferred for flexibility)

**Config file** `data/envs/amp_g1_env.yaml`:

```yaml
env:
  char_file: "data/assets/g1/g1.xml"
  episode_length: 10.0
  
  # NEW: Random force configuration
  enable_random_force: true
  random_force_scale: [5.0, 5.0, 5.0]  # Newtons in x, y, z
  random_force_bodies: ["torso_link", "head_link"]
  random_force_probability: 0.5  # Apply force 50% of steps
  
  # ... rest of config ...
```

---

## Target Upper Body Bodies for G1

From `data/assets/g1/g1.xml`:

```xml
<!-- Torso (main upper body mass) -->
<body name="torso_link" pos="0 0 0.019">
  <inertial mass="9.598" ... />  <!-- Heaviest upper body part -->
  
  <!-- Head (good for balance perturbation) -->
  <body name="head_link" pos="0.015 0 0.43">
    <geom name="head_collision" type="sphere" size="0.06" />
  </body>
  
  <!-- Left shoulder chain -->
  <body name="left_shoulder_pitch_link" pos="0.0039563 0.10022 0.23778">
    <inertial mass="0.718" ... />
    
    <!-- Left shoulder roll -->
    <body name="left_shoulder_roll_link" pos="0 0.038 -0.013831">
      <inertial mass="0.643" ... />
      
      <!-- Left shoulder yaw -->
      <body name="left_shoulder_yaw_link" pos="0 0.00624 -0.1032">
        <inertial mass="0.734" ... />
        
        <!-- Left elbow -->
        <body name="left_elbow_link" pos="0.015783 0 -0.080518">
          <inertial mass="0.6" ... />
```

**Recommended target bodies** (with decreasing priority):
1. `torso_link` - Maximum mass effect on stability (9.598 kg)
2. `head_link` - Balance perturbation (minimal mass)
3. `left_shoulder_pitch_link`, `right_shoulder_pitch_link` - Arm perturbation

---

## Integration Checklist

- [ ] Modify environment config YAML to include random force parameters
- [ ] Add force config parsing to `DeepMimicEnv.__init__()`
- [ ] Create body ID mapping for force target bodies
- [ ] Implement `_apply_random_forces()` method
- [ ] Call force application in appropriate step location
- [ ] Ensure force tensor shapes match Isaac Gym API
- [ ] Add logging for applied forces (optional but useful)
- [ ] Test with simple visualization
- [ ] Train and compare results with/without random forces

---

## Expected Benefits

1. **Robustness**: Agent learns to handle external perturbations
2. **Stability**: More stable gaits that can recover from pushes
3. **Generalization**: Better transfer to real-world scenarios
4. **ADD Training**: Can improve discriminator signal by increasing behavior diversity


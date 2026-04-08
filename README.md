# MimicKit

<p align="center">
  <img src="images/teaser_humanoid.gif" width="32%" style="margin-right: 1px;">
  <img src="images/teaser_g1.gif" width="32%" style="margin-right: 1px;">
  <img src="images/teaser_go2.gif" width="32%">
</p>

This framework provides a suite of motion imitation methods for training motion controllers. This codebase is designed to be clean and lightweight, with minimal dependencies. A more detailed overview of MimicKit is available in the [Starter Guide](https://arxiv.org/abs/2510.13794). For a more feature-rich and modular motion imitation framework, checkout [ProtoMotions](https://github.com/NVlabs/ProtoMotions/). 

Instructions for each method are available here:
- [DeepMimic](docs/README_DeepMimic.md)
- [AMP - Adversarial Motion Priors](docs/README_AMP.md)
- [AWR - Advantage-Weighted Regression](docs/README_AWR.md)
- [ASE - Adversarial Skill Embeddings](docs/README_ASE.md)
- [LCP - Lipschitz-Constrained Policies](docs/README_LCP.md)
- [ADD - Adversarial Differential Discriminator](docs/README_ADD.md)

---

## Installation

This framework supports different simulator backends (referred to as `Engines`). We highly recommend using a package manager, like [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html), to create dedicated Python environments for each simulator.

1. Install the simulator of your choice. 
<details>
<summary>Isaac Gym</summary>

Install [Isaac Gym](https://developer.nvidia.com/isaac-gym).

To use Isaac Gym, specify the argument `--engine_config data/engines/isaac_gym_engine.yaml` when running the code.
</details>

<details>
<summary>Isaac Lab</summary>

Install [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). This framework has been tested with `2ed331acfcbb1b96c47b190564476511836c3754`.

To use Isaac Lab, specify the argument `--engine_config data/engines/isaac_lab_engine.yaml` when running the code.
</details>

<details>
<summary>Newton</summary>

Install [Newton](https://newton-physics.github.io/newton/guide/installation.html). This framework has been tested with `v1.0.0`.

To use Newton, specify the argument `--engine_config data/engines/newton_engine.yaml` when running the code.
</details>

2. Install the requirements.
```
pip install -r requirements.txt
```

3. Download assets and motion data from [here](https://1sfu-my.sharepoint.com/:u:/g/personal/xbpeng_sfu_ca/EclKq9pwdOBAl-17SogfMW0Bved4sodZBQ_5eZCiz9O--w?e=bqXBaa), then extract the contents into [`data/`](data/).


---

## Training

To train a model, run the following command:
```
python mimickit/run.py --mode train --num_envs 4096 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/deepmimic_humanoid_env.yaml --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml --visualize true --out_dir output/
```
- `--mode` selects either `train` or `test` mode.
- `--num_envs` the number of parallel environments used for simulation. Not all environments support parallel envs, this is mainly used for Isaac Gym envs and other environments, like DeepMind Control Suite does not support this feature and should therefore use 1 for the number of envs.
- `--engine_config` configuration file for the engine to select between different simulator backends.
- `--env_config` configuration file for the environment.
- `--agent_config` configuration file for the agent.
- `--visualize` enables visualization. Rendering should be disabled for faster training.
- `--out_dir` the output directory where the models and logs will be saved.
- `--logger` the logger used to record training stats. The options are text `txt`, TensorBoard `tb`, or `wandb`.
- `--video` either `true` or `false` to enable headless video recording, which are then recorded by the logger. Only Isaac Gym and Isaac Lab currently support video logging.

Instead of specifying all arguments through the command line, arguments can also be loaded from an `arg_file`:
```
python mimickit/run.py --arg_file args/deepmimic_humanoid_ppo_args.txt --visualize true
```
The arguments in `arg_file` are treated the same as command line arguments. Arguments for all algorithms are provided in [`args/`](args/).


## Testing

To test a model, run the following command:
```
python mimickit/run.py --arg_file args/deepmimic_humanoid_ppo_args.txt --num_envs 4 --visualize true --mode test --model_file data/models/deepmimic_humanoid_spinkick_model.pt
```
- `--model_file` specifies the `.pt` file that contains the parameters of the trained model. Pretrained models are available in [`data/models/`](data/models/), and the corresponding training log files are available in [`data/logs/`](data/logs/).


## Distributed Training

To use distributed training with multi-CPU or multi-GPU:
```
python mimickit/run.py --arg_file args/deepmimic_humanoid_ppo_args.txt --devices cuda:0 cuda:1
``` 
- `--devices` specifies the devices used for training, which can be `cpu` or `cuda:{i}`. Multiple devices can be provided to parallelize training across multiple processes.


## Visualizer UI

- **Camera control:** Hold `Alt` key and drag with the left mouse button to pan the camera. Scroll with the mouse wheel to zoom in/out.
- **Pause Simulation:** `Enter` key can be used to pause/unpause the simulation
- **Step Simulation:** `Space` key can be used to step the simulator one step at a time.


## Visualizing Training Logs

When using the TensorBoard logger during training, a TensorBoard `events` file will be saved in the same output directory as the log file. The log can be viewed with:
```
tensorboard --logdir=output/ --port=6006 --samples_per_plugin scalars=999999
```
The output `log.txt` file can also be plotted using the plotting script [`plot_log.py`](tools/plot_log/plot_log.py).

---

## Motion Data
Motion data is stored in [`data/motions/`](data/motions/). The `motion_file` field in the environment configuration file can be used to specify the reference motion clip. In addition to imitating individual motion clips, `motion_file` can also specify a dataset file, located in [`data/datasets/`](data/datasets/), which will train a model to imitate a dataset containing multiple motion clips.

The `view_motion` environment can be used to visualize motion clips:
```
python mimickit/run.py --mode test --arg_file args/view_motion_humanoid_args.txt --visualize true
```

Motion clips are represented by the `Motion` class implemented in [`motion.py`](mimickit/anim/motion.py). Each motion clip is stored in a `.pkl` file. Each frame in a motion specifies the pose of the character according to
```
[root position (3D), root rotation (3D), joint rotations]
```
where 3D rotations are specified using 3D exponential maps. Joint rotations are recorded in the order that the joints are specified in the `.xml` file (i.e. depth-first traversal of the kinematic tree). For example, in the case of [`humanoid.xml`](data/assets/humanoid/humanoid.xml), each frame is represented as
```
[root position (3D), root rotation (3D), abdomen (3D), neck (3D), right_shoulder (3D), right_elbow (1D), left_shoulder (3D), left_elbow (1D), right_hip (3D), right_knee (1D), right_ankle (3D), left_hip (3D), left_knee (1D), left_ankle (3D)]
```
The rotations of 3D joints are represented using 3D exponential maps, and the rotations of 1D joints are represented using 1D rotation angles.


## Motion Retargeting
Motion retargeting can be done using [GMR](https://github.com/YanjieZe/GMR). A script to convert GMR files to the MimicKit format is available in [`tools/gmr_to_mimickit/`](tools/gmr_to_mimickit/).

A script to convert SMPL motion files from [AMASS](https://amass.is.tue.mpg.de/) to the MimicKit format is available in [`tools/smpl_to_mimickit/`](tools/smpl_to_mimickit/).

---

## Citation
If you find this codebase helpful, please cite:
```
@article{
      MimicKitPeng2025,
      title={MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control}, 
      author={Peng, Xue Bin},
      year={2025},
      eprint={2510.13794},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2510.13794}, 
}

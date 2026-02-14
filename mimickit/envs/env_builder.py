import yaml

# this is needed to ensure correct import order for some simulators
import engines.engine_builder as engine_builder

from util.logger import Logger

def build_env(env_file, engine_file, num_envs, device, visualize, record_video=False):
    env_config, engine_config = load_configs(env_file, engine_file)

    env_name = env_config["env_name"]
    Logger.print("Building {} env".format(env_name))
    
    if (env_name == "char"):
        import envs.char_env as char_env
        env = char_env.CharEnv(env_config=env_config, engine_config=engine_config, num_envs=num_envs, device=device, visualize=visualize, record_video=record_video)
    elif (env_name == "deepmimic"):
        import envs.deepmimic_env as deepmimic_env
        env = deepmimic_env.DeepMimicEnv(env_config=env_config, engine_config=engine_config, num_envs=num_envs, device=device, visualize=visualize, record_video=record_video)
    elif (env_name == "amp"):
        import envs.amp_env as amp_env
        env = amp_env.AMPEnv(env_config=env_config, engine_config=engine_config, num_envs=num_envs, device=device, visualize=visualize, record_video=record_video)
    elif (env_name == "ase"):
        import envs.ase_env as ase_env
        env = ase_env.ASEEnv(env_config=env_config, engine_config=engine_config, num_envs=num_envs, device=device, visualize=visualize, record_video=record_video)
    elif (env_name == "add"):
        import envs.add_env as add_env
        env = add_env.ADDEnv(env_config=env_config, engine_config=engine_config, num_envs=num_envs, device=device, visualize=visualize, record_video=record_video)
    elif (env_name == "char_dof_test"):
        import envs.char_dof_test_env as char_dof_test_env
        env = char_dof_test_env.CharDofTestEnv(env_config=env_config, engine_config=engine_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "view_motion"):
        import envs.view_motion_env as view_motion_env
        env = view_motion_env.ViewMotionEnv(env_config=env_config, engine_config=engine_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "task_location"):
        import envs.task_location_env as task_location_env
        env = task_location_env.TaskLocationEnv(env_config=env_config, engine_config=engine_config, num_envs=num_envs, device=device, visualize=visualize, record_video=record_video)
    elif (env_name == "task_steering"):
        import envs.task_steering_env as task_steering_env
        env = task_steering_env.TaskSteeringEnv(env_config=env_config, engine_config=engine_config, num_envs=num_envs, device=device, visualize=visualize, record_video=record_video)
    elif (env_name == "static_objects"):
        import envs.static_objects_env as static_objects_env
        env = static_objects_env.StaticObjectsEnv(env_config=env_config, engine_config=engine_config, num_envs=num_envs, device=device, visualize=visualize, record_video=record_video)
    else:
        assert(False), "Unsupported env: {}".format(env_name)

    return env

def load_config(file):
    if (file is not None and file != ""):
        with open(file, "r") as stream:
            config = yaml.safe_load(stream)
    else:
        config = None
    return config

def load_configs(env_file, engine_file):
    env_config = load_config(env_file)
    engine_config = load_config(engine_file)

    if ("engine" in env_config):
        env_engine_config = env_config["engine"]
        engine_config = override_engine_config(env_engine_config, engine_config)

    return env_config, engine_config

def override_engine_config(env_engine_config, engine_config):
    Logger.print("Overriding Engine configs with parameters from the Environment:")
    
    if (engine_config is None):
        engine_config = env_engine_config
    else:
        engine_config = engine_config.copy()
        for key, val in env_engine_config.items():
            engine_config[key] = val
            Logger.print("\t{}: {}".format(key, val))

    Logger.print("")
    return engine_config
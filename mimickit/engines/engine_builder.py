try:
    import isaacgym.gymapi as gymapi
except ImportError:
    pass

def build_engine(config, num_envs, device, visualize, record_video=False):
    eng_name = config["engine_name"]

    if (eng_name == "isaac_gym"):
        import engines.isaac_gym_engine as isaac_gym_engine
        engine = isaac_gym_engine.IsaacGymEngine(config, num_envs, device, visualize, record_video=record_video)
    elif (eng_name == "isaac_lab"):
        import engines.isaac_lab_engine as isaac_lab_engine
        engine = isaac_lab_engine.IsaacLabEngine(config, num_envs, device, visualize, record_video=record_video)
    elif (eng_name == "newton"):
        import engines.newton_engine as newton_engine
        engine = newton_engine.NewtonEngine(config, num_envs, device, visualize, record_video=record_video)
    elif (eng_name == "warp"):
        import engines.warp_engine as warp_engine
        engine = warp_engine.WarpEngine(config, num_envs, device, visualize, record_video=record_video)
    elif (eng_name == "ovphysx"):
        import engines.ovphysx_engine as ovphysx_engine
        engine = ovphysx_engine.OvPhysXEngine(config, num_envs, device, visualize, record_video=record_video)
    else:
        assert False, print("Unsupported engine: {:s}".format(eng_name))

    return engine
import envs.deepmimic_env as deepmimic_env
import engines.engine as engine
import numpy as np

class StaticObjectsEnv(deepmimic_env.DeepMimicEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return
    
    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)
        self._build_static_object(env_id, config)
        return
    
    def _build_static_object(self, env_id, env_config):
        color = np.array([0.3, 0.3, 0.3])
        objs_config = env_config["objects"]

        for i, obj_config in enumerate(objs_config):
            asset_file = obj_config["file"]
            pos = obj_config["pos"]
            rot = obj_config.get("rot", [0.0, 0.0, 0.0, 1.0])

            pos = np.array(pos)
            rot = np.array(rot)

            obj_name = "static_object{:d}".format(i)
            self._engine.create_obj(env_id=env_id,
                                    obj_type=engine.ObjType.rigid,
                                    asset_file=asset_file,
                                    name=obj_name,
                                    start_pos=pos,
                                    start_rot=rot,
                                    fix_root=True,
                                    color=color)
        return
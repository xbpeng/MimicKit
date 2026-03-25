import yaml

from util.logger import Logger

def build_agent(agent_file, env, device):
    if (agent_file is None or agent_file == ""):
        agent_name = "Dummy"
    else:
        agent_config = load_agent_file(agent_file)
        agent_name = agent_config["agent_name"]

    Logger.print("Building {} agent".format(agent_name))

    if (agent_name == "Dummy"):
        import learning.dummy_agent as dummy_agent
        agent = dummy_agent.DummyAgent(env=env, device=device)
    elif (agent_name == "PPO"):
        import learning.ppo_agent as ppo_agent
        agent = ppo_agent.PPOAgent(config=agent_config, env=env, device=device)
    elif (agent_name == "AWR"):
        import learning.awr_agent as awr_agent
        agent = awr_agent.AWRAgent(config=agent_config, env=env, device=device)
    elif (agent_name == "AMP"):
        import learning.amp_agent as amp_agent
        agent = amp_agent.AMPAgent(config=agent_config, env=env, device=device)
    elif (agent_name == "ASE"):
        import learning.ase_agent as ase_agent
        agent = ase_agent.ASEAgent(config=agent_config, env=env, device=device)
    elif (agent_name == "ADD"):
        import learning.add_agent as add_agent
        agent = add_agent.ADDAgent(config=agent_config, env=env, device=device)
    elif (agent_name == "LCP"):
        import learning.lcp_agent as lcp_agent
        agent = lcp_agent.LCPAgent(config=agent_config, env=env, device=device)
    else:
        assert(False), "Unsupported agent: {}".format(agent_name)

    num_params = agent.calc_num_params()
    Logger.print("Total parameter count: {}".format(num_params))

    return agent

def load_agent_file(file):
    with open(file, "r") as stream:
        agent_config = yaml.safe_load(stream)
    return agent_config

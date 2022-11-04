import numpy as np
import pandas as pd
import json
import configparser


def test():
    config_ini = configparser.ConfigParser()
    config_ini.optionxform = str
    config_ini.read('./config/config.ini', encoding='utf-8')
    N_AGENTS = int(config_ini.get("ENV1", "N_AGENTS"))
    trajs = []
    start_goal_position = []
    for i in range(N_AGENTS):
        agent_info = json.loads(config_ini.get("ENV1","AGENT_START_GOAL_EXPERT"+str(i+1)))
        start_goal_position += [agent_info[0]]
        trajs += [agent_info[1]]
    print(start_goal_position)
    print(trajs)
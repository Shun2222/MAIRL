import numpy as np
import copy
from colorama import Fore, Back, Style
from .libs.traj_util import *

class Archive():
    def __init__(self, N_AGENTS, config_ini):
        ARCHIVE = "ARCHIVE"

        self.N_AGENTS = N_AGENTS
        self.ARCHIVE_ADDITIONAL_STEP = int(config_ini.get(ARCHIVE, "ARCHIVE_ADDITIONAL_STEP"))

        self.opt_traj_archive = [[] for _ in range(self.N_AGENTS)]
        self.traj_archive = [[] for _ in range(self.N_AGENTS)]
        self.count_memory = [[{} for _ in range(self.N_AGENTS)] for _ in range(self.N_AGENTS)]
    
    def clear_memory(self):
        self.opt_traj_archive = [[] for _ in range(self.N_AGENTS)]
        self.traj_archive = [[] for _ in range(self.N_AGENTS)]
        self.count_memory = [[{} for _ in range(self.N_AGENTS)] for _ in range(self.N_AGENTS)]

    def print_traj_archive(self):
        for i in range(self.N_AGENTS):
            print(self.traj_archive[i])

    def print_count_memory(self):
        for i in range(self.N_AGENTS):
            print(self.count_memory[i])

    def archive(self, trajs, experts):
        dose_optimal = [False]*self.N_AGENTS
        for i in range(self.N_AGENTS):
            if(has_duplicates(trajs[i])) or len(trajs[i]) > len(experts[i][0]):
                continue
            else:
                dose_optimal[i] = True
                self.update_execute_count(i, trajs[i])
            if not trajs[i] in self.traj_archive[i]:
                self.traj_archive[i] += [copy.deepcopy(trajs[i])]
                if trajs[i] not in self.opt_traj_archive[i]:
                    self.opt_traj_archive[i] += [copy.deepcopy(trajs[i])]
        #print(f"Is archived? {self.traj_archive}")
        self.count()
        self.traj_archive = [[] for _ in range(self.N_AGENTS)]

    def count(self):
        for i in range(self.N_AGENTS):
            if not self.traj_archive[i]:
                continue
            for traj1 in self.traj_archive[i]:
                for j in range(self.N_AGENTS):
                    if i==j or not(self.traj_archive[j]):
                        continue
                    for traj2 in self.traj_archive[j]:
                        if is_collision(traj1, traj2):
                            self.update_collision_count(i, j, traj1)
                        else:
                            self.update_not_collision_count(i, j, traj1)

    def update_execute_count(self, i, traj):
        str_traj = array_to_str(traj)
        for j in range(self.N_AGENTS):
            if str_traj in self.count_memory[i][j]:
                #print("count exec")
                self.count_memory[i][j][str_traj][2] += 1
            else:
                self.count_memory[i][j][str_traj] = [0,0,1]

    def update_collision_count(self, i, j, traj):
        str_traj = array_to_str(traj)
        #print(f"col count {self.count_memory[i][j][str_traj][0]}")
        if str_traj in self.count_memory[i][j]:
            self.count_memory[i][j][str_traj][0] += 1
            #print("Enter col")
        #print(f"col count {self.count_memory[i][j][str_traj][0]}")

    def update_not_collision_count(self, i, j, traj):
        str_traj = array_to_str(traj)
        #print(f"non col count {self.count_memory[i][j][str_traj][1]}")
        if str_traj in self.count_memory[i][j]:
            self.count_memory[i][j][str_traj][1] += 1
            #print("Enter non col")
        #print(f"non col count {self.count_memory[i][j][str_traj][1]}")
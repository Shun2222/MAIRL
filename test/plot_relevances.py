from MAIRL import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as patches
import seaborn as sns
import datetime



def run():
    dir=[
    "Min-col-rate-uprate02",
    ]

    seed = 20
    N_seed = 1
    agent = 8
    #str_trajs = ["19,20,21,22,23,17,16","19,13,7,8,9,10,16"]
    window = [100, 500]
    y_sum = []
    file=r"D:\graduation\graduation_datas\large-cycle\SumExpert/"
    for d in dir:
        relevances = []
        for i in range(seed, seed+N_seed):
            fileDir = file+d+"/Seed_No"+str(i)+"/logs.pickle"
            logs = pickle_load(fileDir)
            tg = logs["traj_gif"].datas
            count_memories = logs["agent_memory"]
            col_count = []
            for j in range(window[0], window[1]):
                mat = is_collision_matrix(tg[j])
                if col_count==[]:
                    col_count = np.zeros(len(tg[j]))
                for k in range(len(col_count)):
                    if mat[agent][k]: col_count[k] += 1  
                rel = col_count/np.sum(col_count)
                if relevances==[]:
                    relevances = [[] for _ in range(len(rel))]
                for k in range(len(rel)):
                    relevances[k].append(rel[k])
                
                count_memory = count_memories[j]
                agent_num = agent
                #for s in range(len(str_trajs)):
                    #str_traj = str_trajs[s]
                count = 0
                for str_traj in count_memory[agent_num][agent_num]:
                    print(f"#####{str_traj}#####")
                    t = str_to_array(str_traj)
                    col = 0
                    non_col = 0
                    sum_non_col_rate = 0.0
                    for a in range(len(count_memory)):
                        if a==agent_num:
                            continue
                        if count_memory[agent_num][a][str_traj]:
                            col = count_memory[agent_num][a][str_traj][0]
                            non_col = count_memory[agent_num][a][str_traj][1]
                            r = non_col/(col+non_col) if col+non_col!=0 else 1.0
                            sum_non_col_rate += r
                    sum_non_col_rate /= len(count_memory)
                    if y_sum==[]:
                        y_sum = [[] for _ in range(len(count_memory[agent_num][agent_num]))]
                    y_sum[count].append(sum_non_col_rate)
                    count += 1

        plt.figure()
        plt.xlabel("Iteration")
        plt.ylabel("Collision rate")
        for i in range(len(relevances)):
            plt.plot(np.arange(len(relevances[i])), relevances[i], label=str(i))
        plt.legend()
        plt.show()
        print(rel)

        plt.figure()
        plt.xlabel("Iteration")
        plt.ylabel("sum value")
        count = 0
        for str_traj in count_memory[agent_num][agent_num]:
            plt.plot(np.arange(window[0], window[1]), np.array(y_sum[count]), label=str_traj)
            print(str_traj)
            print(y_sum[count][-1])
            count += 1
        plt.legend()
        plt.show()
        print(sum_non_col_rate)

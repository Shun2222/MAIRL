from MAIRL import *


def print_best_traj_by_rank(agent_num, rank, count_memory):
    if agent_num==rank[0]:
        print("Agent rank is 0.")
        return 0


    i = 0
    lower_rank = []
    while True:
        if agent_num==rank[i]:
            break
        lower_rank.append(rank[i])
        i += 1

    max_col = None
    traj = None
    for str_traj in count_memory[agent_num][agent_num]:
        t = str_to_array(str_traj)
        col = 0
        non_col = 0
        non_col_rate = 1.0
        for i in lower_rank:
            if count_memory[agent_num][i][str_traj]:
                col = count_memory[agent_num][i][str_traj][0]
                non_col = count_memory[agent_num][i][str_traj][1]
                non_col_rate *= non_col/(col+non_col) if col+non_col!=0 else 1.0
                #print(f'Agent{agent_num}{i}: col non_col col_rate traj {col} : {non_col} : {non_col_rate} : {str_traj}')
        if not max_col:
            max_col = non_col_rate
            traj = str_to_array(str_traj)
        else:
            if max_col < non_col_rate:
                max_col = non_col_rate
                traj = str_to_array(str_traj)
    print(f'Agent{agent_num}, rank{rank[agent_num]}, max_non_col{max_col}, best_traj [{traj}]')
    return traj

def run():
    dir=[
    "100step",
    ]

    seed = 12
    N_seed = 10

    file="logs/MYENV/"
    """plt.figure()
    plt.xlabel("iteration")
    plt.ylabel("step")
    plt.ylim(0, 10)"""
    
    for d in dir:
        data = []
        for i in range(seed, seed+N_seed):
            fileDir = file+d+"/Seed_No"+str(i)+"/logs.pickle"
            logs = pickle_load(fileDir)
            rank = logs["rank"]
            j = 19
            for i in rank:
                j+=100
                memories = logs['agent_memory'][j]
                print(f"Agent{i}, greedy_act [{logs['agents'][i].greedy_act}]")
                best = print_best_traj_by_rank(i, rank, memories)
                if best==logs['agents'][i].greedy_act or best==0:
                    print("OK")
                else:
                    print("Faild")




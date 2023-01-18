from MAIRL import *
from tqdm import tqdm

def run():
    dir=[
    "Min-col-rate",
    ]

    seed = 22 24kara
    N_seed = 1
    state_size = [6, 6]
    s = 70
    g = 90

    file=r"C:/Users/messi/graduation_datas/large-cycle/exist/"
    
    for d in dir:
        data = []
        for i in tqdm(range(seed, seed+N_seed)):
            fileDir = file+d+"/Seed_No"+str(i)+"/logs.pickle"
            logs = pickle_load(fileDir)
            for j in tqdm(range(len(logs["expert_gifs"]))):
            	logs["expert_gifs"][j].datas = logs["expert_gifs"][j].datas[s:g]
            	logs["expert_gifs"][j].make(state_size=state_size, folder=file+d+"/Seed_No"+str(i), file_name=f"expert{j}_seed-{s}-{g}")



def best_traj(agent_num, count_memory):
    max_col = None
    traj = None
    for str_traj in count_memory[agent_num][agent_num]:
        t = str_to_array(str_traj)
        col = 0
        non_col = 0
        non_col_rate = 1.0
        x = []
        y = []
        plt.xlabel("Agent index")
        plt.ylabel("Non collition rate")
        for i in range(len(count_memory)):
            if count_memory[agent_num][i][str_traj]:
                col = count_memory[agent_num][i][str_traj][0]
                non_col = count_memory[agent_num][i][str_traj][1]
                r = non_col/(col+non_col) if col+non_col!=0 else 1.0
                non_col_rate *= r
                x.append(f"{i}")
                y.append(r)
        plt.title(f"Agent{agent_num}, {str_traj}, "+str(non_col_rate))
        if not max_col:
            max_col = non_col_rate
            traj = str_to_array(str_traj)
        else:
            if max_col < non_col_rate:
                max_col = non_col_rate
                traj = str_to_array(str_traj)
        plt.bar(x, y)
        plt.show()
    return traj


def best_traj2(agent_num, count_memory, trajs):
    max_col = None
    traj = None
    for str_traj in count_memory[agent_num][agent_num]:
        t = str_to_array(str_traj)
        if not t in trajs:
            continue
        col = 0
        non_col = 0
        non_col_rate = 1.0
        plt.xlabel("Agent index")
        plt.ylabel("Non collition rate")
        for i in range(len(count_memory)):
            if count_memory[agent_num][i][str_traj]:
                col += count_memory[agent_num][i][str_traj][0]
                non_col += count_memory[agent_num][i][str_traj][1]
        r = non_col/(col+non_col) if col+non_col!=0 else 1.0
        non_col_rate = r
        print(f"{str_traj}, {non_col_rate}")
        if not max_col:
            max_col = non_col_rate
            traj = str_to_array(str_traj)
        else:
            if max_col < non_col_rate:
                max_col = non_col_rate
                traj = str_to_array(str_traj)
    return traj
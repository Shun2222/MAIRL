from MAIRL import *

def print_traj(agent_num, count_memory, trajs=None, bar=True):

    max_col = None
    traj = None
    if not trajs:
        trajs = count_memory[agent_num][agent_num]
    
    y_sum = []
    y_pi = []
    for str_traj in trajs:
        print(f"#####{str_traj}#####")
        t = str_to_array(str_traj)
        col = 0
        non_col = 0
        sum_non_col_rate = 0.0
        pi_non_col_rate = 1.0
        x1 = []
        y1 = []
        for i in range(len(count_memory)):
            if i==agent_num:
                continue
            if count_memory[agent_num][i][str_traj]:
                col = count_memory[agent_num][i][str_traj][0]
                non_col = count_memory[agent_num][i][str_traj][1]
                r = non_col/(col+non_col) if col+non_col!=0 else 1.0
                sum_non_col_rate += r
                pi_non_col_rate *= r
                x1.append(i)
                y1.append(r)
                print(f"Agent{i}:{r} ({col}/{non_col})")
        sum_non_col_rate /= len(count_memory)
        y_sum.append(sum_non_col_rate)
        y_pi.append(pi_non_col_rate)
    print(f'Mean non col rate: {sum_non_col_rate}')
    print(f'Probability non col rate: {pi_non_col_rate}\n')
    plt.figure()
    plt.xlabel("Trajs")
    plt.ylabel("Non collision rate(sum)")
    #plt.xticks(x)
    plt.title(f"Non collition rate(sum)")
    plt.bar(np.arange(len(y_pi)), y_sum)
    plt.show()

    plt.figure()
    plt.xlabel("Trajs")
    plt.ylabel("Non collision rate(pi)")
    #plt.xticks(x)
    plt.title(f"Non collition rate(pi)")
    plt.bar(np.arange(len(y_pi)), y_pi)
    plt.show()


def run():
    dir=[
    "Min-col-rate",
    ]

    seed = 34
    N_seed = 1
    agent = 1
    trajs = []

    file=r"D:\graduation\graduation_datas\large-cycle\SumExpert/"
    
    for d in dir:
        data = []
        for i in range(seed, seed+N_seed):
            fileDir = file+d+"/Seed_No"+str(i)+"/logs.pickle"
            logs = pickle_load(fileDir)
            m = logs["agent_memory"][-1]
            print(f"##########Agent{agent}##########")
            print_traj(agent, m, trajs)
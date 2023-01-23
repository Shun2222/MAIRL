from MAIRL import *


def print_traj1(agent_num, count_memory, relevance, state_size=None, n_state=None, str_trajs=None, bar=True):
    max_col = None
    traj = None
    if not str_trajs:
        str_trajs = count_memory[agent_num][agent_num]
    
    y_sum = []
    y_pi = []
    y_prop = []
    y = []
    for str_traj in str_trajs:
        print(f"#####{str_traj}#####")
        t = str_to_array(str_traj)
        col = 0
        non_col = 0
        sum_non_col_rate = 0.0
        prop_non_col_rate = 0.0
        pi_non_col_rate = 1.0
        for i in range(len(count_memory)):
            if i==agent_num:
                continue
            if count_memory[agent_num][i][str_traj]:
                col = count_memory[agent_num][i][str_traj][0]
                non_col = count_memory[agent_num][i][str_traj][1]
                r = non_col/(col+non_col) if col+non_col!=0 else 1.0
                sum_non_col_rate += r
                prop_non_col_rate += relevance[agent_num][i]*r
                pi_non_col_rate *= r
                print(f"Agent{i}:{r} ({relevance[agent_num][i]}*{col}/{non_col})")
        sum_non_col_rate /= len(count_memory)
        prop_non_col_rate = prop_non_col_rate/np.sum(relevance[agent_num]) if np.sum(relevance[agent_num])!=0 else 1.0
        y_sum.append(sum_non_col_rate)
        y_pi.append(pi_non_col_rate)
        y_prop.append(prop_non_col_rate)
        n = count_memory[agent_num][agent_num][str_traj][2]
        y.append(n)
    print(f'Mean non col rate: {sum_non_col_rate}')
    print(f'Probability non col rate: {pi_non_col_rate}\n')

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.set_xlabel("Trajs")
    ax.set_ylabel("Non collision rate(sum)")
    ax.set_xticks(np.arange(len(y_sum)))
    ax.set_title(f"Non collition rate(sum) for each traj")
    ax.bar(np.arange(len(y_pi)), y_sum)

    ax = fig.add_subplot(2, 2, 2)
    ax.set_xlabel("Trajs")
    ax.set_ylabel("Non collision rate(pi)")
    ax.set_xticks(np.arange(len(y_pi)))
    ax.set_title(f"Non collition rate(pi) for each traj")
    ax.bar(np.arange(len(y_pi)), y_pi)

    ax = fig.add_subplot(2, 2, 3)
    ax.set_xlabel("Trajs")
    ax.set_ylabel("Non collision rate(prop)")
    ax.set_xticks(np.arange(len(y_prop)))
    ax.set_title(f"Non collition rate(prop) for each traj")
    ax.bar(np.arange(len(y_prop)), y_prop)

    ax = fig.add_subplot(2, 2, 4)
    ax.set_xlabel("Trajs")
    ax.set_ylabel("Execution count")
    ax.set_xticks(np.arange(len(y)))
    ax.set_title(f"Execution count for each traj")
    ax.bar(np.arange(len(y)), y)
    plt.show()

def print_traj0(agent_num, count_memory, relevance, state_size=None, n_state=None, sum_rate=-0.1, pi_rate=-0.1, str_trajs=None, bar=True):
    max_col = None
    traj = None
    if not str_trajs:
        str_trajs = count_memory[agent_num][agent_num]

    print(f"num of imgs:{len(str_trajs)}<=n*m, ")
    s = input("n,m = ").split(",")

    x1 = [[] for _ in range(len(str_trajs))]
    y1 = [[] for _ in range(len(str_trajs))]
    y = []
    y_sum = []
    y_prop = []
    f_trajs = []
    count = 0
    for k, str_traj in enumerate(str_trajs):
        print(f"#####{str_traj}#####")
        t = str_to_array(str_traj)
        col = 0
        non_col = 0
        sum_non_col_rate = 0.0
        prop_non_col_rate = 0.0
        pi_non_col_rate = 1.0
        for i in range(len(count_memory)):
            if i==agent_num:
                continue
            if count_memory[agent_num][i][str_traj]:
                col = count_memory[agent_num][i][str_traj][0]
                non_col = count_memory[agent_num][i][str_traj][1]
                r = non_col/(col+non_col) if col+non_col!=0 else 1.0
                sum_non_col_rate += r
                prop_non_col_rate += relevance[agent_num][i]*r
                pi_non_col_rate *= r
                x1[k].append(i)
                y1[k].append(r)
        sum_non_col_rate /= len(count_memory)
        prop_non_col_rate = prop_non_col_rate/np.sum(relevance[agent_num]) if np.sum(relevance[agent_num])!=0 else 1.0
        y_sum.append(np.round(sum_non_col_rate, decimals=3))
        y_prop.append(np.round(prop_non_col_rate, decimals=3))
        n = count_memory[agent_num][agent_num][str_traj][2]
        y.append(n)
        f_traj = calc_state_visition_count(n_state, [t])
        f_trajs.append(f_traj)
        count += 1
    
    fig = plt.figure()
    for i in range(len(f_trajs)):
        ax = fig.add_subplot(int(s[0]), int(s[1]), i+1)
        ax_heatmap(f_trajs[i], state_size, ax, set_annot=True, cbar=False, square=False, label_size=8, title=f"{i}, prop{y_prop[i]}")
    plt.show()

    fig = plt.figure()
    for i in range(len(str_trajs)):
        ax = fig.add_subplot(int(s[0]), int(s[1]), i+1)
        #ax.set_xlabel("Agent index")
        #ax.set_ylabel("Non collision rate")
        ax.set_xticks(x1[i])
        ax.set_title(f"{i}, prop{y_prop[i]}")
        ax.tick_params(axis='x', labelsize=6)
        ax.bar(x1[i], y1[i])
        ax.plot(np.arange(len(x1[i])+1), np.ones(len(x1[i])+1)*y_sum[i], color='forestgreen')
        ax.plot(np.arange(len(x1[i])+1), np.ones(len(x1[i])+1)*y_prop[i], color='red')
    plt.show()

def create_relevance(col_count):
        relevance = []
        for i in range(len(col_count)):
            c = np.array(col_count[i])
            c = c/np.sum(c) if np.sum(c)!=0 else c
            relevance.append(c)
        return relevance
def run():
    dir=[
    "Min-col-rate",
    ]

    seed = 12
    N_seed = 1

    file=r"D:\graduation\graduation_datas\large-cycle\PropExpert/"

    state_size = [6, 6]
    n_state = state_size[0]*state_size[1]
    agent = 8
    str_trajs = []#["19,13,7,8,9,10,16"] 

    
    for d in dir:
        data = []
        for i in range(seed, seed+N_seed):
            fileDir = file+d+"/Seed_No"+str(i)+"/logs.pickle"
            logs = pickle_load(fileDir)
            m = logs["agent_memory"][-1]
            relevance = create_relevance(logs["col_count"])
            print(f"##########Agent{agent}##########")
            print_traj0(agent, m, relevance ,state_size=state_size, n_state=n_state, str_trajs=str_trajs)
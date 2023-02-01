from MAIRL import *


def print_traj1(agent_num, count_memory, state_size=None, n_state=None, str_trajs=None, bar=True):
    max_col = None
    traj = None
    if not str_trajs:
        str_trajs = count_memory[agent_num][agent_num]
    
    y_sum = []
    y_pi = []
    y = []
    for str_traj in str_trajs:
        print(f"#####{str_traj}#####")
        t = str_to_array(str_traj)
        col = 0
        non_col = 0
        sum_non_col_rate = 0.0
        pi_non_col_rate = 1.0
        for i in range(len(count_memory)):
            if i==agent_num:
                continue
            if count_memory[agent_num][i][str_traj]:
                col = count_memory[agent_num][i][str_traj][0]
                non_col = count_memory[agent_num][i][str_traj][1]
                r = non_col/(col+non_col) if col+non_col!=0 else 1.0
                sum_non_col_rate += r
                pi_non_col_rate *= r
                print(f"Agent{i}:{r} ({col}/{non_col})")
        sum_non_col_rate /= len(count_memory)
        y_sum.append(sum_non_col_rate)
        y_pi.append(pi_non_col_rate)
        n = count_memory[agent_num][agent_num][str_traj][2]
        y.append(n)
    print(f'Mean non col rate: {sum_non_col_rate}')
    print(f'Probability non col rate: {pi_non_col_rate}\n')

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.set_xlabel("Trajs")
    ax.set_ylabel("Non collision rate(sum)")
    ax.set_xticks(np.arange(len(y_sum)))
    ax.set_title(f"Non collition rate(sum) for each traj")
    ax.bar(np.arange(len(y_pi)), y_sum)

    ax = fig.add_subplot(1, 3, 2)
    ax.set_xlabel("Trajs")
    ax.set_ylabel("Non collision rate(pi)")
    ax.set_xticks(np.arange(len(y_pi)))
    ax.set_title(f"Non collition rate(pi) for each traj")
    ax.bar(np.arange(len(y_pi)), y_pi)

    ax = fig.add_subplot(1, 3, 3)
    ax.set_xlabel("Trajs")
    ax.set_ylabel("Execution count")
    ax.set_xticks(np.arange(len(y_pi)))
    ax.set_title(f"Execution count for each traj")
    ax.bar(np.arange(len(y)), y)
    plt.show()

def print_traj0(agent_num, count_memory, state_size=None, n_state=None, sum_rate=-0.1, pi_rate=-0.1, str_trajs=None, bar=True):
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
    f_trajs = []
    count = 0
    for k, str_traj in enumerate(str_trajs):
        print(f"#####{str_traj}#####")
        t = str_to_array(str_traj)
        col = 0
        non_col = 0
        sum_non_col_rate = 0.0
        pi_non_col_rate = 1.0
        for i in range(len(count_memory)):
            if i==agent_num:
                continue
            if count_memory[agent_num][i][str_traj]:
                col = count_memory[agent_num][i][str_traj][0]
                non_col = count_memory[agent_num][i][str_traj][1]
                r = non_col/(col+non_col) if col+non_col!=0 else 1.0
                sum_non_col_rate += r
                pi_non_col_rate *= r
                x1[k].append(i)
                y1[k].append(r)
        sum_non_col_rate /= len(count_memory)
        y_sum.append(np.round(sum_non_col_rate, decimals=3))
        n = count_memory[agent_num][agent_num][str_traj][2]
        y.append(n)
        f_traj = calc_state_visition_count(n_state, [t])
        f_trajs.append(f_traj)
        count += 1
    
    fig = plt.figure()
    for i in range(len(f_trajs)):
        ax = fig.add_subplot(int(s[0]), int(s[1]), i+1)
        ax_heatmap(f_trajs[i], state_size, ax, set_annot=True, cbar=False, square=False, label_size=8, title=f"{i}, sum{y_sum[i]}")
    plt.show()

    fig = plt.figure()
    for i in range(len(str_trajs)):
        ax = fig.add_subplot(int(s[0]), int(s[1]), i+1)
        #ax.set_xlabel("Agent index")
        #ax.set_ylabel("Non collision rate")
        ax.set_xticks(x1[i])
        ax.set_title(f"{i}, mean{y_sum[i]}")
        ax.tick_params(axis='x', labelsize=6)
        ax.bar(x1[i], y1[i])
        ax.plot(np.arange(len(x1[i])+1), np.ones(len(x1[i])+1)*y_sum[i], color='forestgreen')
    plt.show()


def run():
    file=r"C:\Users\messi\graduation/"
    #dirs = [r"min-col-sum", r"min-col-pi", "free-no-order-sum", r"free-no-order-pi", r"free-no-order-prop"]#, r"min-col-prop"]#r"free-no-order-sum", r"free-no-order-pi", r"free-no-order-prop"]#r"min-col-sum", r"min-col-pi", r"min-col-prop"]#, r"free-no-order-sum", r"free-no-order-pi", r"free-no-order-prop"]
    dirs = ["free-no-order-sum"]
    labels = ["TA-MAIRL(sum)", "TA-MAIRL(prod)", "TFA-MAIRL(sum)", "TFA-MAIRL(prod)", "TRA-MAIRL",]
    seed_file = "/MAIRL/logs/seed"
    seed = 12
    N_seed = 1
    state_size = [6, 6]
    n_state = state_size[0]*state_size[1]
    agent = 8
    str_trajs = []#["19,13,7,8,9,10,16"] 

    for d in dirs:
        data = []
        for i in range(seed, seed+N_seed):
            fileDir = file+d+seed_file+str(i)+"/logs499.pickle"
            logs = pickle_load(fileDir)
            m = logs["agent_memory"][-1]
            print(f"##########Agent{agent}##########")
            print_traj0(agent, m, state_size=state_size, n_state=n_state, str_trajs=str_trajs)
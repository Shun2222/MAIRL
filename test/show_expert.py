from MAIRL import *
def run():
    file=r"C:\Users\messi\graduation/"
    #dirs = [r"min-col-sum", r"min-col-pi", "free-no-order-sum", r"free-no-order-pi", r"free-no-order-prop"]#, r"min-col-prop"]#r"free-no-order-sum", r"free-no-order-pi", r"free-no-order-prop"]#r"min-col-sum", r"min-col-pi", r"min-col-prop"]#, r"free-no-order-sum", r"free-no-order-pi", r"free-no-order-prop"]
    dirs = ["min-col-pi"]
    labels = ["TA-MAIRL(sum)", "TA-MAIRL(prod)", "TFA-MAIRL(sum)", "TFA-MAIRL(prod)", "TRA-MAIRL",]
    seed_file = "/MAIRL/logs/seed"
    seed = 12
    N_seed = 15
    state_size = [6, 6]
    n_state = state_size[0]*state_size[1]
    str_trajs = []#["19,13,7,8,9,10,16"] 

    for d in dirs:
        data = []
        for i in range(seed, seed+N_seed):
            print(f"seed{i}")
            fileDir = file+d+seed_file+str(i)+"/logs1999.pickle"
            logs = pickle_load(fileDir)
            agents = logs["agents"]
            rewards = logs["rewards"]
            fig = plt.figure()
            n_agents = len(agents)
            #print(f"num of imgs:{n_agents}<=n*m, ")
            #s = input("n,m = ").split(",")
            s = ["5","4"]
            for i in range(n_agents):
                ax = fig.add_subplot(int(s[0]), int(s[1]), i+1)
                f_traj = agents[i].feature_expert
                #f_traj = rewards[i]
                ax_heatmap(f_traj, state_size, ax, set_annot=True, cbar=False, square=False, label_size=8, title=f"agent{i}")
            plt.show()
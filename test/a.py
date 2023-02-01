from MAIRL import *

def run():
    file=r"C:\Users\messi\graduation/"
    #dirs = [r"min-col-sum", r"min-col-pi", "free-no-order-sum", r"free-no-order-pi", r"free-no-order-prop"]#, r"min-col-prop"]#r"free-no-order-sum", r"free-no-order-pi", r"free-no-order-prop"]#r"min-col-sum", r"min-col-pi", r"min-col-prop"]#, r"free-no-order-sum", r"free-no-order-pi", r"free-no-order-prop"]
    dirs = ["min-col-sum", "min-col-pi"]
    labels = ["TA-MAIRL(sum)", "TA-MAIRL(prod)", "TFA-MAIRL(sum)", "TFA-MAIRL(prod)", "TRA-MAIRL",]
    seed_file = "/MAIRL/logs/seed"
    seed = 12
    n_seed = 3
    state_size = [6, 6]
    n_state = state_size[0]*state_size[1]
    str_trajs = []#["19,13,7,8,9,10,16"] 


    plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("Average number of collisions")
    al=0
    for d in range(len(dirs)):
        data = []
        print(f"Loading data in {d} now")
        for i in range(seed, seed+n_seed):
            print(f"seed{i}")
            fileDir = file+dirs[d]+seed_file+str(i)+"/logs399.pickle"
            log = pickle_load(fileDir)
            step = log["col_greedy"]
            m_step = np.mean(step, axis=0)[1:1999]
            ave_step = []
            for ite in np.arange(len(m_step)):
                pre = ite-5 if ite-5>=0 else 0
                nex = ite+5 if ite+5<len(m_step) else len(m_step)-1
                ave_step += [np.mean(m_step[pre:nex])]
            data.append(ave_step)
        data = np.array(data)
        m = data.mean(axis=0)
        std = data.std(axis=0)  
        print(f"std:{np.mean(std)}")
        plt.fill_between(np.arange(len(ave_step)), m+std, m-std, alpha=0.2)
        plt.plot(np.arange(len(ave_step)), m, label=labels[d])
        logs = {"mean_col": m, 
                "std": std}
        pickle_dump(logs, file+dirs[d]+seed_file+str(seed)+"/mean_col.pickle")
        al+=1

    fileName = file+dirs[0]+"/exp-con"+'.png'
    plt.legend()
    plt.show()
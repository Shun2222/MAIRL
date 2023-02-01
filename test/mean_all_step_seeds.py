from MAIRL import *
def run():
    n_iter = 2000
    file=r"C:\Users\messi\graduation/"
    dirs = [r"min-col-pi"]#, "free-no-order-sum", r"free-no-order-pi", r"free-no-order-prop"]#, r"min-col-prop"]#r"free-no-order-sum", r"free-no-order-pi", r"free-no-order-prop"]#r"min-col-sum", r"min-col-pi", r"min-col-prop"]#, r"free-no-order-sum", r"free-no-order-pi", r"free-no-order-prop"]
    labels = ["TA-MAIRL(sum)", "TA-MAIRL(prod)"]#, "TFA-MAIRL(sum)", "TFA-MAIRL(prod)", "TRA-MAIRL",]
    seed_file = "/MAIRL/logs/seed"
    seed = 12
    n_seed = 15

    for d in dirs:
        plt.figure()
        plt.ylim(0, 35)
        plt.xlabel("Seed number")
        plt.ylabel("Average steps")
        x = []
        y = []
        data = []
        for i in range(seed, seed+n_seed):
            fileDir = file+d+seed_file+str(i)+"/logs.pickle"
            logs = pickle_load(fileDir)
            steps = logs["step_in_multi_hist"]
            m_step = np.mean(steps, axis=0)[1:500]
            m_all_step = np.mean(m_step)
            x.append(i)
            y.append(m_all_step-1)
            print(f"seed{i}:{m_all_step}")
        plt.xticks(np.arange(seed, seed+n_seed, step=1))
        plt.bar(x, y)
        plt.show()
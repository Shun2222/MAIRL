from MAIRL import *
def run():
    dir=[
    "Min-col-rate",
    ]

    seed = 12
    N_seed = 25

    file=r"D:\graduation\graduation_datas\large-cycle\SumExpert/"
    plt.figure()
    plt.xlabel("Seed number")
    plt.ylabel("Mean step")
    x = []
    y = []
    for d in dir:
        data = []
        for i in range(seed, seed+N_seed):
            fileDir = file+d+"/Seed_No"+str(i)+"/logs.pickle"
            logs = pickle_load(fileDir)
            steps = logs["step_in_multi_hist"]
            m_step = np.mean(steps, axis=0)[1:]
            m_all_step = np.mean(m_step)

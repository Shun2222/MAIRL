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
    "Min-col-rate",
    ]

    seed = 12
    N_seed = 30

    file=r"D:\graduation\graduation_datas\large-cycle\PropExpert/"
    plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("Collision num")
    datas = []
    for d in dir:
        data = []
        for i in range(seed, seed+N_seed):
            fileDir = file+d+"/Seed_No"+str(i)+"/logs.pickle"
            logs = pickle_load(fileDir)
            col = logs["col_greedy"]
            if not datas:
                datas = [[] for _ in range(len(col))]
            for i in range(len(col)):
                ave_col = pre_nex(col[i])
                datas[i].append(copy.deepcopy(ave_col))
    for i in range(len(datas)):
        data = np.array(datas[i]) 
        print(f"data{data}")
        m = data.mean(axis=0)
        print(f"m{m}")
        std = data.std(axis=0) 

        plt.fill_between(np.arange(len(m)), m+std, m-std, alpha=0.2)
        plt.plot(np.arange(len(m)), m, label="Agent"+str(i))

    fileName = "col_seeds"+'.png'
    plt.legend()
    #plt.savefig(save_dirs[0]+"/"+fileName) 
    #plt.close()
    plt.show()
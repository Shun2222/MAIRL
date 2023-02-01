from MAIRL import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as patches
import seaborn as sns
import datetime

def run():

    n_iter = 2000
    file=r"C:\Users\messi\graduation/"
    #dirs = [r"min-col-sum", r"min-col-pi", "free-no-order-sum", r"free-no-order-pi", r"free-no-order-prop"]#, r"min-col-prop"]#r"free-no-order-sum", r"free-no-order-pi", r"free-no-order-prop"]#r"min-col-sum", r"min-col-pi", r"min-col-prop"]#, r"free-no-order-sum", r"free-no-order-pi", r"free-no-order-prop"]
    dirs = ["free-no-order-sum"]
    labels = ["TA-MAIRL(sum)", "TA-MAIRL(prod)", "TFA-MAIRL(sum)", "TFA-MAIRL(prod)", "TRA-MAIRL",]
    seed_file = "/MAIRL/logs/seed"
    seed = 12
    N_seed = 15

    plt.xlabel('Seed number')
    plt.ylabel('Collision number')
    x = []
    y = []
    for d in dirs:
        data = []
        for i in range(seed, seed+N_seed):
            fileDir = file+d+seed_file+str(i)+"/logs.pickle"
            logs = pickle_load(fileDir)
            col = logs['col_greedy']
            min_col = np.min(np.max(np.array(col), axis=0))
            print(f'min_col:{min_col}')
            x.append(i)
            y.append(min_col)
    plt.xticks(np.arange(seed, seed+N_seed, step=1))
    plt.bar(x, y)
    plt.show()
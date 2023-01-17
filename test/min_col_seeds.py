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
    "16agents",
    ]

    seed = 12
    N_seed = 20

    file="logs/MYENV/"
    x = []
    y = []
    for d in dir:
        data = []
        for i in range(seed, seed+N_seed):
            fileDir = file+d+"/Seed_No"+str(i)+"/logs.pickle"
            logs = pickle_load(fileDir)
            col = logs['col_greedy']
            min_col = np.min(np.max(np.array(col), axis=0))
            print(f'min_col:{min_col}')
            x.append("seed"+str(i))
            y.append(min_col)
    plt.bar(x, y)
    plt.show()

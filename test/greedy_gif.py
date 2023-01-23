from MAIRL import *

def run():
    dir=[
    "Min-col-rate-uprate02",
    ]

    state_size = [6, 6]
    n_state = state_size[0]*state_size[1]
    seed = 20
    N_seed = 1
    window = [350, 450]
    str_trajs = []#["19,13,7,8,9,10,16"] 

    file=r"D:\graduation\graduation_datas\large-cycle\SumExpert/"
    
    for d in dir:
        data = []
        for i in range(seed, seed+N_seed):
            fileDir = file+d+"/Seed_No"+str(i)+"/logs.pickle"
            logs = pickle_load(fileDir)
            datas = logs["traj_gif"].datas
            labels = [f"{i}" for i in range(window[0], window[1])]
            heatmap_gif(datas[window[0]:window[1]], state_size, labels=labels, folder=file, file_name=f"greedy_trajs{window[0]}-{window[1]}")
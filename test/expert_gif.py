from MAIRL import *

def run():
    dir=[
    "Min-col-rate-uprate02",
    ]

    state_size = [6, 6]
    n_state = state_size[0]*state_size[1]
    seed = 20
    N_seed = 1
    window = [250, 350]
    str_trajs = []#["19,13,7,8,9,10,16"] 

    file=r"D:\graduation\graduation_datas\large-cycle\SumExpert/"
    
    for d in dir:
        data = []
        for i in range(seed, seed+N_seed):
            fileDir = file+d+"/Seed_No"+str(i)+"/logs.pickle"
            logs = pickle_load(fileDir)
            egs = logs["expert_gifs"]
            datas = [eg.datas for eg in egs]
            reshaped_datas = []
            for j in range(window[0], window[1]):            
                d = []
                for i in range(len(datas)):
                    d.append(datas[i][j])
                reshaped_datas.append(copy.deepcopy(d))
            datas = np.array(reshaped_datas)
            print(datas.shape)
            labels = [f"{i}" for i in range(window[0], window[1])]
            heatmap_gif(datas, state_size, feature=True, labels=labels, folder=file, file_name=f"expert_trajs{window[0]}-{window[1]}")
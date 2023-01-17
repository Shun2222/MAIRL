from MAIRL import *
def run():
    dir=[
    "16agents-30step",
    ]

    seed = 12
    N_seed = 30

    file="logs/MYENV/"
    
    for d in dir:
        data = []
        for i in range(seed, seed+N_seed):
            fileDir = file+d+"/Seed_No"+str(i)+"/logs.pickle"
            ave_step = mean_pre_nex(log_file=fileDir, key="step_in_multi_hist")
            print(f'seed:{i}, min step:{np.min(ave_step)}')

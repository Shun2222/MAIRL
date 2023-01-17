from MAIRL import *
def run():
    file="logs/MYENV/100step"
    seed = 12
    N_seed = 20

    dirs=[file+"/Seed_No"+str(i) for i in range(seed, seed+N_seed)]
    plot_steps_seeds(dirs)



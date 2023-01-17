from MAIRL import *
def run():
    dir=[
    "100step",
    ]

    seed = 12
    N_seed = 10

    file="logs/MYENV/"
    """plt.figure()
    plt.xlabel("iteration")
    plt.ylabel("step")
    plt.ylim(0, 10)"""
    
    for d in dir:
        data = []
        for i in range(seed, seed+N_seed):
            fileDir = file+d+"/Seed_No"+str(i)+"/logs.pickle"
            logs = pickle_load(fileDir)
            for agent in logs['agents']:
                best_traj = agent.best_traj
                greedy_act = agent.greedy_act
                if all(best_trajs==greedy_act):
                    print("Completely matched")
                else:
                    print("Donot matched")
                print(f"best_traj: {best_traj}")
                print(f"greedy_act: {greedy_act}")
from MAIRL import *

n_iter = 2000
file=r"C:\Users\messi\graduation/"
dirs = [r"min-col-sum", r"min-col-pi", r"min-col-prop-2"]#, "free-no-order-sum", "min-col-prop-2"]
#dirs = [r"convention/id/MAIRL-Conventional", r"convention/np/MAIRL-Conventional"]
seed_file = r"/MAIRL/logs/seed"
labels = ["TC-MAIRL(sum)", "TC-MAIRL(prod)","WTC-MAIRL"]
#labels = ["A-MAIRL(id)", "A-MAIRL(np)"]
seed = 12
n_seed = 15

n_states = 36
def run():
    plt.figure()
    plt.xlabel("iteration")
    plt.ylabel("expert_step")
    for d in range(len(dirs)):
        print(f"Loading data in {d} now")
        data = []
        for s in range(seed, seed+n_seed):
            m_steps = []
            fileDir = file+dirs[d]+seed_file+str(s)+"/logs.pickle"
            log = pickle_load(fileDir)
            n_agents = len(log["agents"])
            for iteration in range(n_iter):
                experts = []
                for i in range(n_agents):
                    expert = log["expert_gifs"][i].datas[iteration]
                    print(expert)
                    experts.append(copy.deepcopy(expert))
                is_col = is_collision_matrix(experts)
                steps = 0.0
                for  i in range(n_agents):
                    if any(is_col[i]):
                        steps+=n_states-1
                    else:
                        steps+=len(experts[i])-1
                steps/= n_agents
                m_steps.append(steps)
            ave_step = []
            for ite in np.arange(len(m_steps)):
                pre = ite-5 if ite-5>=0 else 0
                nex = ite+5 if ite+5<len(m_step) else len(m_step)-1
                ave_step += [np.mean(m_step[pre:nex])]
            data.append(ave_step)
        data = np.array(data)
        m = data.mean(axis=0)
        std = data.std(axis=0)  
        print(f"std:{np.mean(std)}")
        plt.fill_between(np.arange(len(ave_step)), m+std, m-std, alpha=0.2)
        plt.plot(np.arange(len(ave_step)), m, label=labels[d])
        logs = {"mean_step": m, 
                "std": std}
        pickle_dump(logs, file+dirs[d]+seed_file+str(seed)+"/mean_expert_step.pickle")

    fileName = file+dirs[0]+"/exp-con"+'.png'
    plt.legend()
    plt.show()




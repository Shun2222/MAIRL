from MAIRL import *

n_iter = 2000
file=r"C:\Users\messi\graduation/"
dirs = [r"min-col-sum", r"min-col-pi"]#, "free-no-order-sum", "min-col-prop-2"]
#dirs = [r"convention/id/MAIRL-Conventional", r"convention/np/MAIRL-Conventional"]
seed_file = r"/MAIRL/logs/env6/seed"
labels = ["TC-MAIRL(sum)", "TC-MAIRL(prod)", "FTC-MAIRL", "WTC-MAIRL"]
#labels = ["A-MAIRL(id)", "A-MAIRL(np)"]
seed = 12
n_seed = 15

def run():
    plt.figure()
    plt.xlabel("iteration")
    plt.ylabel("step")
    al=0
    for d in range(len(dirs)):
        data = []
        print(f"Loading data in {d} now")
        for i in range(seed, seed+n_seed):
            #print(f"seed{i}")
            fileDir = file+dirs[d]+seed_file+str(i)+"/logs49.pickle"
            log = pickle_load(fileDir)
            step = log["step_in_multi_hist"]
            m_step = np.mean(step, axis=0)[1:1999]
            ave_step = []
            for ite in np.arange(len(m_step)):
                pre = ite-5 if ite-5>=0 else 0
                nex = ite+5 if ite+5<len(m_step) else len(m_step)-1
                ave_step += [np.mean(m_step[pre:nex])]
            data.append(ave_step)
        data = np.array(data)
        m = data.mean(axis=0)-1
        std = data.std(axis=0)  
        print(f"std:{np.mean(std)}")
        plt.fill_between(np.arange(len(ave_step)), m+std, m-std, alpha=0.2)
        plt.plot(np.arange(len(ave_step)), m, label=labels[d])
        logs = {"mean_step": m, 
                "std": std}
        pickle_dump(logs, file+dirs[d]+seed_file+str(seed)+"/mean_step.pickle")
        al+=1

    fileName = file+dirs[0]+"/exp-con"+'.png'
    plt.legend()
    plt.savefig(fileName) 
    plt.close()
    print(f'Saved {fileName}')
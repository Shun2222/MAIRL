from MAIRL import *
def run():
    dir=[
    "alpha02",
    "alpha04",
    "alpha06",
    "alpha08",
    "alpha10",
    ]

    seed = 12
    N_seed = 10

    file=r"D:\graduation\graduation_datas\cycle_env/Min-col-rate/"
    plt.figure()
    plt.xlabel("iteration")
    plt.ylabel("step")
    al=0
    for d in dir:
        data = []
        for i in range(seed, seed+N_seed):
            fileDir = file+d+"/Seed_No"+str(i)+"/logs.pickle"
            log = pickle_load(fileDir)
            step = log["step_in_multi_hist"]
            m_step = np.mean(step, axis=0)[1:]
            ave_step = []
            for ite in np.arange(len(m_step)):
                pre = ite-5 if ite-5>=0 else 0
                nex = ite+5 if ite+5<len(m_step) else len(m_step)-1
                ave_step += [np.mean(m_step[pre:nex])]
            data.append(ave_step)
        data = np.array(data)
        m = data.mean(axis=0)
        std = data.std(axis=0)  
        plt.fill_between(np.arange(len(ave_step)), m+std, m-std, alpha=0.2)
        plt.plot(np.arange(len(ave_step)), m, label=d)
        al+=1

    fileName = file+"step_in_multi_mean"+'.png'
    plt.legend()
    plt.savefig(fileName) 
    plt.close()
    print('Saved image')
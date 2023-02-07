from MAIRL import *

n_iter = 2000
file=r"C:\Users\messi\graduation/"
dirs = [r"min-col-sum", "min-col-prop-2"]
#dirs = [r"convention/id/MAIRL-Conventional", r"convention/np/MAIRL-Conventional"]
seed_file = r"/MAIRL/logs/seed"
labels = ["Unweighted", "Weighted"]
#labels = ["A-MAIRL(id)", "A-MAIRL(np)"]
seed = 12
n_seed = 15

def run():
    plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("Average steps")
    al=0
    for d in range(len(dirs)):
        print(f"Loading data in {d} now")
        mean_step = pickle_load(file+dirs[d]+seed_file+str(seed)+"/mean_step.pickle")
        m = mean_step["mean_step"]
        std = mean_step["std"]
        print(f"std:{np.mean(std)}")
        plt.fill_between(np.arange(len(m)), m+std, m-std, alpha=0.2)
        plt.plot(np.arange(len(m)), m, label=labels[d])

    print(f"Is this steps  really correct? Check now!!!!!!!")
    #ave = 10*1 + 2*4 + 6*12
    #ave /= 17
    fileName = file+dirs[0]+"/exp1-average-steps"+'.png'
    #plt.plot(np.arange(len(m)), np.ones(len(m))*ave, label="Min average steps")
    plt.legend()
    plt.show()

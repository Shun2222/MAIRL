from .MaxEntIRL import *
from .libs import *
import pandas as pd
import json
import configparser
import pickle
from colorama import Fore, Back, Style

def save(logs, Seed_No, N_ITERS, STATE_SIZE, N_AGENTS, ENV, experts, save_dir):
    print("Saving datas now.")
    folder = "logs/"+ENV
    if not os.path.exists(folder):
        os.mkdir(folder)
    if save_dir=="":
        folder += "/"+str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    else:
        folder += "/"+save_dir
    if not os.path.exists(folder):
        os.mkdir(folder)
    folderName = folder+"/Seed_No" + str(Seed_No)
    if not os.path.exists(folderName):
        os.mkdir(folderName)
    imageFolder = folderName + "/images/"
    if not os.path.exists(imageFolder):
        os.mkdir(imageFolder)

    experts = [e[0] for e in experts]
    trajs_plot(experts, STATE_SIZE, 'expert'+'_'+str(Seed_No), imageFolder)
    for i in range(N_AGENTS):
        #step
        plt.figure()
        plt.plot(np.arange(len(logs["step_hist"][i])), logs["step_hist"][i], label="")
        plt.xlabel("iteration")
        plt.ylabel("step")
        fileName = "step"+"_agent"+str(i)+str(Seed_No)+'.png'
        plt.savefig(os.path.join(imageFolder, fileName)) 
        plt.close()
        #archive count
        plot_on_grid(logs["rewards"][i], STATE_SIZE, 'reward_agent'+str(i)+'_'+str(Seed_No), imageFolder)
        plot_on_grid(logs["feat_experts"][i], STATE_SIZE, 'expert_feature_agent'+str(i)+'_'+str(Seed_No), imageFolder)

        """df = pd.DataFrame(data=rewards)
        df.to_csv(folderName+'/reward_agent'+'_'+str(Seed_No)+'.csv', index = False)
        for i in range(N_AGENTS):
            df = pd.DataFrame(data=policy[i])
            df.to_csv(folderName+'/policy_agent'+str(i)+'_'+str(Seed_No)+'.csv', index = False)
            df = pd.DataFrame(data=Qtables[i])
            df.to_csv(folderName+'/Qtables_agent'+str(i)+'_'+str(Seed_No)+'.csv', index = False)
            df = pd.DataFrame(data=feat_experts[i])
            df.to_csv(folderName+'/feat_expert_agent'+str(i)+'_'+str(Seed_No)+'.csv', index = False)
            policy = np.array(policy)
            Qtables = np.array(Qtables)
            act = []
            for j in range(STATE_SIZE[0]*STATE_SIZE[1]):
                act.append(np.argmax(policy[i][j]))
            act = np.array(act)
            #arrow_plot(act, STATE_SIZE, title="trajectory", folder=imageFolder, fileName="act"+str(i)+str(j))
            plot_on_grid(act, STATE_SIZE, "act_agent"+str(i)+"_"+str(Seed_No), imageFolder)
            for a in range(N_ACTIONS):
                plot_on_grid(policy[i].T[a], STATE_SIZE, 'policy_agent'+str(i)+"_"+str(action_set[a])+'_'+str(Seed_No), imageFolder)
                plot_on_grid(Qtables[i].T[a], STATE_SIZE, 'Qtables_agent'+str(i)+"_"+str(action_set[a])+"_"+str(Seed_No), imageFolder)"""
    #step in multi env
    plt.figure()
    for i in range(N_AGENTS):
        plt.plot(np.arange(len(logs["step_in_multi_hist"][i])), logs["step_in_multi_hist"][i], label="Agent"+str(i))
    plt.xlabel("iteration")
    plt.ylabel("step")
    plt.legend(loc='upper right') 
    fileName = "steps_in_multi"+"_agent"+str(Seed_No)+'.png'
    plt.savefig(os.path.join(imageFolder, fileName)) 
    plt.close()
    #mean step in multi env
    m_step = np.mean(logs["step_in_multi_hist"], axis=0)[1:]
    ave_step = []
    for ite in np.arange(len(m_step)):
        pre = ite-5 if ite-5>=0 else 0
        nex = ite+5 if ite+5<len(m_step) else len(m_step)-1
        ave_step += [np.mean(m_step[pre:nex])]
    plt.figure()
    plt.plot(np.arange(len(ave_step)), ave_step)
    plt.xlabel("iteration")
    plt.ylabel("step")
    fileName = "step_in_multi_mean"+"_agent"+str(Seed_No)+'.png'
    plt.savefig(os.path.join(imageFolder, fileName)) 
    plt.close()
    plt.figure()
    for i in range(N_AGENTS):
        plt.plot(np.arange(len(logs["col_greedy"][i])), logs["col_greedy"][i], label="Agent"+str(i))
    plt.xlabel("iteration")
    plt.ylabel("collision num (act greedy)")
    plt.legend(loc='upper right') 
    fileName = "col_greedy"+"_agent"+str(Seed_No)+'.png'
    plt.savefig(os.path.join(imageFolder, fileName)) 
    plt.close()
    plot_on_grid(logs["col_count"], [N_AGENTS, N_AGENTS], 'col_count'+'_'+str(Seed_No), imageFolder, set_annot=False)
    """
    print("making gif now")
    for i in range(N_AGENTS):
        logs["expert_gifs"][i].make(state_size=STATE_SIZE, folder=imageFolder, file_name="expert"+str(i))
    #logs["traj_gif"].make_test(state_size=STATE_SIZE, folder=imageFolder, file_name="traj")"""
    with open(os.path.join(folderName, "logs.pickle"), mode='wb') as f:
        pickle.dump(logs, f)
    print("saved in {}".format(folderName))
    print("Finished saving datas.")
    return folderName

if __name__ == "__main__":
    print(Fore.BLUE+"\n\
         __  __    _    ___ ____  _\n\
        |  \\/  |  / \\  |_ _|  _ \\| |\n\
        | |\\/| | / _ \\  | || |_) | |\n\
        | |  | |/ ___ \\ | ||  _ <| |___\n\
        |_|  |_/_/   \\_\\___|_| \\_\\_____|\n")

    print(Style.RESET_ALL)
    ACTION = "ACTION"
    MAIRL_PARAM = "MAIRL_PARAM"

    config_ini = configparser.ConfigParser()
    config_ini.optionxform = str
    config_ini.read('./config/config.ini', encoding='utf-8')
    N_ACTIONS = int(config_ini.get(ACTION, "N_ACTIONS"))
    action_set = json.loads(config_ini.get(ACTION, "ACTION_SET"))
    ENV = json.loads(config_ini.get("ENV", "ENV_INFO"))
    """ENV"""
    """
    ENV = "ENV10"

        """

    """Random Env"""
    if ENV=="RANDOM":
        N_AGENTS = int(config_ini.get("ENV", "N_AGENTS"))
        STATE_SIZE = json.loads(config_ini.get("ENV", "STATE_SIZE")) 
        N_OBSTACLES = json.loads(config_ini.get("ENV", "OBSTACLE")) 
        experts = [[[]] for _ in range(N_AGENTS)]
        env = create_environment(N_AGENTS, STATE_SIZE, N_OBSTACLES=N_OBSTACLES)
        for i in range(N_AGENTS):
            print("#################Agent{}#################".format(i))
            env[i].print_env()
            experts[i][0] += env[i].create_expert()
            print("create expert{}:{}\n".format(i, experts[i][0]))
       #print(env)

    elif ENV=="LOG":
        env_file = json.loads(config_ini.get("LOG", "LOG_FILE")) 
        env = pickle_load(env_file)
        N_AGENTS = len(env)
        STATE_SIZE = env[0].grid.shape
        N_OBSTACLES = np.sum(env[0].grid=='-1')
        experts = [[[]] for _ in range(N_AGENTS)]
        for i in range(N_AGENTS):
            print("#################Agent{}#################".format(i))
            env[i].print_env()
            experts[i][0] += env[i].create_expert()
            print("create expert{}:{}\n".format(i, experts[i][0]))   
    elif ENV=='MYENV':
        STATE_SIZE = json.loads(config_ini.get("ENV", "STATE_SIZE")) 
        agents = create_my_env(STATE_SIZE)
        env = create_env(STATE_SIZE, agents)
        N_AGENTS = len(env)
        experts = []
        for i in range(len(env)):
            print("#################Agent{}#################".format(i))
            env[i].print_env()
            print("expert{}:{}\n".format(i, agents[i][2]))  
            experts.append([agents[i][2]])
    else:
        experts = []
        start_goal_position = []
        N_AGENTS = int(config_ini.get(ENV, "N_AGENTS"))
        STATE_SIZE = json.loads(config_ini.get(ENV, "STATE_SIZE")) 
        obstacle = json.loads(config_ini.get(ENV, "OBSTACLE")) 
        for i in range(N_AGENTS):
            agent_info = json.loads(config_ini.get(ENV,"AGENT_START_GOAL_EXPERT"+str(i+1)))
            start_goal_position += [agent_info[0]]
            experts += [agent_info[1]]

        env = [[] for i in range(N_AGENTS)]
        for i in range(N_AGENTS):
            e = [[0]*STATE_SIZE[1] for i in range(STATE_SIZE[0])]
            e[start_goal_position[i][0][0]][start_goal_position[i][0][1]] = 'S'
            e[start_goal_position[i][1][0]][start_goal_position[i][1][1]] = 'G'
            for o in obstacle:
                if o:
                    e[o[0]][o[1]] = '-1'
            env[i] = GridWorldEnv(grid=e)
            print("#################Agent{}#################".format(i))
            env[i].print_env()
            if not experts[i][0]:
                experts[i][0] += env[i].create_expert()
                print("create expert{}:{}\n".format(i, experts[i][0]))
        
    rewards = [[] for i in range(N_AGENTS)]
    irl = MaxEntIRL(env, N_AGENTS, config_ini)

    GAMMA = float(config_ini.get(MAIRL_PARAM, "GAMMA"))
    LEARNING_RATE = float(config_ini.get(MAIRL_PARAM, "LEARNING_RATE"))
    N_ITERS = int(config_ini.get(MAIRL_PARAM, "N_ITERS"))
    Seed_No = int(config_ini.get(MAIRL_PARAM, "Seed_No"))
    N_Seeds = int(config_ini.get(MAIRL_PARAM, "N_Seeds"))
    state = [str(i) for i in range(len(env[0].states))]
    
    """学習"""
    save_dirs = []
    for count in range(N_Seeds):
        seed = Seed_No+count
        print("###### Now " + str(count/N_Seeds) + "% (Seed_No = "+ str(seed)+") ######")
        np.random.seed(seed)
        feat_map = np.eye(irl.N_STATES)    
        logs= irl.maxent_irl(irl.N_STATES,irl.N_STATES,feat_map, experts, LEARNING_RATE, GAMMA, N_ITERS)
        save_dir = json.loads(config_ini.get("LOG", "SAVE_DIR"))
        save_dir = save(logs, seed, N_ITERS, STATE_SIZE, N_AGENTS, ENV, experts, save_dir)
        save_dirs.append(save_dir)

    with open(os.path.join(save_dir, "env.pickle"), mode='wb') as f:
        pickle.dump(env, f)
    if N_Seeds!=1:
        plot_steps_seeds(save_dirs, label="")
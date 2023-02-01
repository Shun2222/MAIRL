from MAIRL import *
import gc

n_iter = 2000
file=r"C:\Users\messi\graduation/"
dirs = [r"min-col-prop-2"]#r"free-no-order-prop"r"min-col-sum", r"min-col-pi", r"min-col-prop"]
#dirs = [r"convention/id/MAIRL-Conventional", r"convention/np/MAIRL-Conventional"]
seed_file = r"/MAIRL/logs\seed"
labels = ["TA-MAIRL(sum)", "TA-MAIRL(prod)", "TFA-MAIRL(sum)", "TFA-MAIRL(prod)", "TRA-MAIRL"]
#labels = ["A-MAIRL(id)", "A-MAIRL(np)"]
seed = 12
n_seed = 15

def run():
	datas = []
	for d in dirs:
		for s in range(seed, seed+n_seed):
			logs = {
				"rewards" : [],
				"feat_experts" : [],
				"step_in_multi_hist" : [],
				"step_hist" : [],
				"expert_gifs" : [],
				"agent_memory" : [],
				"col_count" : [],
				"col_greedy" :[],
				"traj_gif" : [], 
				"rank" : [], 
				"agents" : []
			}
			for iteration in range(n_iter):
				if (iteration+1)%100==0:
					file_name = file+d+seed_file+str(s)+f"/logs{iteration}.pickle"
					l = pickle_load(file_name)
					if logs["step_in_multi_hist"]==[]:
						n_agent = len(l["step_in_multi_hist"])
						logs["step_in_multi_hist"] = [[] for _ in range(n_agent)]
						logs["step_hist"] = [[] for _ in range(n_agent)]
						logs["expert_gifs"] = [make_gif() for _ in range(n_agent)]
						logs["col_greedy"] = [[] for _ in range(n_agent)]			
					for i in range(len(l["step_in_multi_hist"])):
						logs["step_in_multi_hist"][i] += l["step_in_multi_hist"][i]
						logs["step_hist"][i] += l["step_hist"][i]
						logs["expert_gifs"][i].datas += l["expert_gifs"][i].datas
						logs["col_greedy"][i] += l["col_greedy"][i]
					logs["rank"] += l["rank"]
					logs["agent_memory"] += l["agent_memory"]
			logs["traj_gif"] = l["traj_gif"]
			logs["rewards"]  = l["rewards"]
			logs["col_count"]  = l["col_count"]
			logs["feat_experts"] = l["feat_experts"]
			logs["agents"] = l["agents"]
			pickle_dump(logs, file+d+seed_file+str(s)+f"/logs.pickle")
			print(f"Saved seed{s} logs.")
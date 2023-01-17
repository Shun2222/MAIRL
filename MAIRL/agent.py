class Agent():
	def __init__(self, id=None):
		self.id = id
		self.original_expert = None
		self.feature_experts = None
		self.status = None
		self.greedy_act = None
		self.best_traj = None
		self.policy = None
		self.Qtable = None

	def print_info(self):
		print(f'\n\
			--------------------\n\
			Agent id: {self.id}\n\
			original_expert: {self.original_expert}\n\
			feature_expert: {self.feature_expert}\n\
			status: {self.status}\n\
			greedy_act: {self.greedy_act}\n\
			best_traj: {self.best_traj}\n\
			--------------------')
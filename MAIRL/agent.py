class Agent():
	def __init__(self, id=None):
		self.id = id
		self.original_expert = None
		self.feature_experts = None
		self.status = None
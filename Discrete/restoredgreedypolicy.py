from policy import Policy


class RestoredGreedyPolicy(Policy):
	def __init__(self, states:int):
		self.state_action_dict = {};
		for state in range(0,states):
			self.state_action_dict[state] = 1;

	def action_prob(self,state:int,action:int):
		if(self.state_action_dict[state] == action):
			return 1.0;
		return 0.0;

	def action(self,state:int):
		return self.state_action_dict[state];

	def print_all(self):
		n = self.env_columns
		for state in self.state_action_dict:
			if self.state_action_dict[state] == 0:
				print("state " , int(state/n) , int(state%n), " : " , self.state_action_dict[state], "N")

			elif self.state_action_dict[state] == 1:
				print("state " , int(state/n) , int(state%n), " : " , self.state_action_dict[state], "W")

			elif self.state_action_dict[state] == 2:
				print("state " , int(state/n) , int(state%n), " : " , self.state_action_dict[state], "S")

			else:
				print("state " , int(state/n) , int(state%n), " : " , self.state_action_dict[state], "E")

	def restore_fromfile(self, filename):
		with open(filename, 'r') as f:
			lines = f.readlines()
			for line in lines:
				state_action = line.split(':')
				self.state_action_dict[int(state_action[0])] = int(state_action[1])


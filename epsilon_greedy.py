import numpy as np

class AdversarialEpsilonGreedy:
    def __init__(self, num_machines=4, epsilon=0.1):
        self.num_machines = num_machines
        self.epsilon = epsilon
        self.rewards = np.zeros(num_machines)  # Estimated rewards
        self.counts = np.zeros(num_machines)  
        self.probabilities = np.full(num_machines, 1 / num_machines) 

    def update(self, machine, reward):
        self.counts[machine] += 1
        self.rewards[machine] += (reward - self.rewards[machine]) / self.counts[machine]
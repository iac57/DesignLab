

from collections import defaultdict

class MachineProbabilityPredictor:
    def __init__(self, num_machines=4):
        self.machine_counts = [0]*4
        self.probabilities = [0]*4
        self.total_count = 0
        self.num_machines = num_machines

    def update(self, machine_played):
        self.machine_counts[machine_played] += 1
        self.total_count += 1

    def predict(self):
        if self.total_count == 0:
            for m in range(self.num_machines):
                self.probabilities[m]=1/self.num_machines
            return self.probabilities
        for m in range(self.num_machines):
            self.probabilities[m]=self.machine_counts[m] / self.total_count
        return self.probabilities


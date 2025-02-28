class Casino:
    def __init__(self, bandits):
        self.bandits = bandits
        
class Bandit:
    def __init__(self, p):
        self.p = p
        self.N = 0 #Number of times the bandit has been pulled

    def pull(self):
        """Simulate pulling the bandit arm."""
        return np.random.random() < self.p

    def setPayout(self, x):
        """Update the payout of the bandit."""
        self.p = x
import numpy as np
#Sample usage:
"""
M = .2
B = 1
bandit1 = Bandit(B/4)
bandit2 = Bandit(B/4)
bandit3 = Bandit(B/4)
bandit4 = Bandit(B/4)
casino = Casino([bandit1, bandit2, bandit3, bandit4], B, M)
#(after player detected leaving foyer:)
casino.setPayoutsRandom()
#after player detected playing slot machine 1:
reward = bandit1.pull
print(reward)
"""
class Casino:
    def __init__(self, bandits, B, M):
        self.bandits = bandits
        self.B = B
        self.M = M
    
    def setPayoutsRandom(self):
        """Set the payouts of the bandits randomly while maintaining constraints."""
        current_payouts = [bandit.p for bandit in self.bandits]
        new_payouts = []
        
        # Generate random changes within M constraint
        for p in current_payouts:
            change = np.random.uniform(-self.M, self.M)
            new_p = np.clip(p + change, 0, 1)
            new_payouts.append(new_p)
        
        # Scale to maintain sum = B
        total = sum(new_payouts)
        if total > 0:
            new_payouts = [p * (self.B / total) for p in new_payouts]
        
        # Update bandits
        for bandit, payout in zip(self.bandits, new_payouts):
            bandit.setPayout(payout)
        
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
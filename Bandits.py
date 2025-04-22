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
        self.B = B #total win probability 
        self.M = M #max you can change any one machine per encounter

    def setPayoutsBehavioral(self, machine_id, accuracy):
        weight = accuracy
        max_adjustment = weight * self.M
        pred_bandit = self.bandits[machine_id-1]
        
        # Calculate the maximum possible decrease for predicted machine
        max_decrease = min(max_adjustment, pred_bandit.p)
        actual_decrease = max_decrease
        
        # Initially decrease the predicted machine's payout
        new_payout = pred_bandit.p - actual_decrease
        
        # Get other bandits and their current payouts
        other_bandits = [b for b in self.bandits if b is not pred_bandit]
        other_payouts = [b.p for b in other_bandits]
        
        # Calculate how much we can increase other machines without exceeding 1.0
        available_space = [1.0 - p for p in other_payouts]
        total_available = sum(available_space)
        
        # Calculate proportional increases for other bandits
        increases = [actual_decrease * (space / total_available) for space in available_space]
       
        # Apply the new payouts
        pred_bandit.setPayout(new_payout)
        for bandit, increase in zip(other_bandits, increases):
            bandit.setPayout(bandit.p + increase)

    def setPayoutsMoCap(self, machine_id):
        # Same logic as above but with fixed adjustment
        max_adjustment = self.M
        pred_bandit = self.bandits[machine_id-1]
        
        # Calculate the maximum possible decrease for predicted machine
        max_decrease = min(max_adjustment, pred_bandit.p)
        actual_decrease = max_decrease
        
        # Initially decrease the predicted machine's payout
        new_payout = pred_bandit.p - actual_decrease
        
        # Get other bandits and their current payouts
        other_bandits = [b for b in self.bandits if b is not pred_bandit]
        other_payouts = [b.p for b in other_bandits]
        
        # Calculate how much we can increase other machines without exceeding 1.0
        available_space = [1.0 - p for p in other_payouts]
        total_available = sum(available_space)
        
        # If we can't distribute all the decrease, reduce the actual decrease
        # Calculate proportional increases for other bandits
       
        increases = [actual_decrease * (space / total_available) for space in available_space]
        
        # Apply the new payouts
        pred_bandit.setPayout(new_payout)
        for bandit, increase in zip(other_bandits, increases):
            bandit.setPayout(bandit.p + increase)
        
        
class Bandit:
    def __init__(self, p):
        self.p = p
        self.N = 0 #Number of times the bandit has been pulled

    def pull(self):
        """Returns 1 is the random number is less than p. Simulates
        Generating 1 with probability p and 0 otherwise."""
        return np.random.random() < self.p

    def setPayout(self, x):
        """Update the payout of the bandit."""
        self.p = x
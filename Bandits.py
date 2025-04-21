import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point

class Casino:
    def __init__(self, bandits, B, M):
        self.bandits = bandits
        self.B = B #total win probability 
        self.M = M #max you can change any one machine per encounter
    
    def set_payouts_random(self):
        """Set the payouts of the bandits randomly while maintaining constraints."""
        current_payouts = [bandit.p for bandit in self.bandits]
        new_payouts = [np.random.random() for _ in range(4)]
        
        # Generate random changes within M constraint
        #for p in current_payouts:
        #    change = np.random.uniform(-self.M, self.M)
        #    new_p = np.clip(p + change, 0, 1)
        #    new_payouts.append(new_p)
        
        # Scale to maintain sum = B
        total = sum(new_payouts)
        #if total > 0:
        #scales it so that the sum is B
        new_payouts = [p * (self.B / total) for p in new_payouts]
        
        # Update bandits
        for bandit, payout in zip(self.bandits, new_payouts):
            bandit.setPayout(payout)

    def setPayoutsBehavioral(self, machine_id, accuracy):
        weight=accuracy
        adjustment= weight*self.M
        pred_bandit=self.bandits[machine_id-1]
        payout=pred_bandit.p + adjustment
        pred_bandit.setPayout(payout)
        for bandit in self.bandits:
            if bandit is not pred_bandit:
                new_payout = bandit.p - adjustment/3
                bandit.setPayout(new_payout)

    
    def setPayoutsMoCap(self, machine_id):
        adjustment= self.M
        pred_bandit=self.bandits[machine_id-1]
        payout=pred_bandit.p + adjustment
        pred_bandit.setPayout(payout)
        for bandit in self.bandits:
            if bandit is not pred_bandit:
                new_payout = bandit.p - adjustment/3
                bandit.setPayout(new_payout)
        
class Bandit:
    def __init__(self, p, points):
        self.p = p
        self.N = 0 #Number of times the bandit has been pulled

    def pull(self):
        """Returns 1 is the random number is less than p. Simulates
        Generating 1 with probability p and 0 otherwise."""
        return np.random.random() < self.p

    def set_payout(self, x):
        """Update the payout of the bandit."""
        self.p = x
    
    def is_played(self, body_cm):
        """Check if the player is playing this machine."""
        return self.polygon.contains(body_cm)
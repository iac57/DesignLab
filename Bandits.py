import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point

class Casino:
    def __init__(self, bandits, B, M):
        self.bandits = bandits
        self.B = B
        self.M = M
    
    def set_payouts_random(self):
        """Set the payouts of the bandits randomly while maintaining constraints."""
        current_payouts = [bandit.get_payout() for bandit in self.bandits]
        new_payouts = []
        
        # Generate random changes within M constraint
        for p in current_payouts:
            change = np.random.uniform(-self.M, self.M)
            new_p = np.clip(p + change, 0, 1) #limit payout to [0,1]
            new_payouts.append(new_p)
        
        # Scale to maintain sum = B
        total = sum(new_payouts)
        if total > 0:
            new_payouts = [p * (self.B / total) for p in new_payouts]
        
        # Update bandits
        for bandit, payout in zip(self.bandits, new_payouts):
            bandit.set_payout(payout)
    
    def check_for_play(self, body_cm):
        """Check if the player is playing any of the bandits."""
        for i, bandit in enumerate(self.bandits):
            if bandit.is_played(body_cm):
                return i+1
        return 0
        
class Bandit:
    def __init__(self, p, points):
        self.p = p
        self.N = 0 #Number of times the bandit has been pulled
        self.polygon = Polygon(points) #define the polygon for the bandit

    def get_payout(self):
        """Get the payout of the bandit."""
        return self.p
    
    def play(self):
        """Simulate pulling the bandit arm."""
        self.N += 1
        return np.random.random() < self.p

    def set_payout(self, x):
        """Update the payout of the bandit."""
        self.p = x
    
    def is_played(self, body_cm):
        """Check if the player is playing this machine."""
        return self.polygon.contains(body_cm)
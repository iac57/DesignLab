import random
import numpy as np
from Bandits import Casino, Bandit

"""
This script simulates a player playing slot machines in a casino.

The player uses a learning algorithm to update their perception of the payout probabilities of each machine. 
The player chooses the machine with the highest perceived payout.

The casino changes the payout probabilities of the machines randomly within a certain range after each play.

The player's goal is to maximize their total money over a fixed number of plays.

The simulation tracks the player's total money, the number of plays per machine, and the player's perceived payout probabilities.

Learning rate parameter controls how much a win/loss affects their perception of a machine's probability.
"""
class SlotMachinePlayer:
    def __init__(self, num_machines, learning_rate=0.1):
        self.num_machines = num_machines
        self.total_money = 0
        #self.body_cm = body_cm
        self.perceived_payouts = [0.5] * num_machines  # Initial perception of 0.5 for all machines
        self.plays_per_machine = [0] * num_machines
        self.learning_rate = learning_rate

    def update_perception(self, machine: Bandit, won):
        # Update perception based on win/loss
        if won:
            self.perceived_payouts[machine] += self.learning_rate * (1 - self.perceived_payouts[machine])
        else:
            self.perceived_payouts[machine] -= self.learning_rate * self.perceived_payouts[machine]

    def play(self):
        # Return index of the machine with the highest perceived payout
        machine_index = self.perceived_payouts.index(max(self.perceived_payouts))
        machine: Bandit = self.machines[machine_index]
        # Play that machine
        won = machine.pull()
        
        # Update money and perceptions
        self.total_money += 1 if won else -1
        self.update_perception(machine, won)
        self.plays_per_machine[machine] += 1
        
        return machine, won


def run_simulation(b=1, m=0.1, num_machines=4, num_plays=300):
    # Set up initial win probabilities for each machine
    casino = Casino([Bandit(b/4) for _ in range(4)], b, m)    
    # Create player
    player = SlotMachinePlayer(num_machines)

    # Run simulation
    results = []

    for _ in range(num_plays):
        # Update probabilities before each play
        casino.setPayoutsRandom()
        true_probabilities = [bandit.p for bandit in casino.bandits]
        
        machine, won = player.play()
        results.append({
            'machine': machine,
            'won': won,
            'total_money': player.total_money,
            'perceptions': player.perceived_payouts.copy(),
            'true_probs': true_probabilities.copy()
        })
    
    return player, results

if __name__ == "__main__":
    final_player, simulation_results = run_simulation()
    print(f"Final money: ${final_player.total_money}")
    print(f"Final perceptions: {final_player.perceived_payouts}")
    print(f"Plays per machine: {final_player.plays_per_machine}")

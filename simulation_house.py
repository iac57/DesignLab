import csv
import random

def compute_action_probabilities(Q, epsilon):
    """
    Compute the probability distribution over actions given the epsilon-greedy policy.
    If there is a unique best machine, its probability is:
      (1 - epsilon) + epsilon/4,
    and each of the others gets epsilon/4.
    In case of ties for the best, the exploit probability is divided equally among them.
    """
    n = len(Q) # Total number of machines
    best_value = max(Q) # Find the highest estimated reward among the machines
    best_indices = [i for i, q in enumerate(Q) if q == best_value]
    num_best = len(best_indices)
    
    probs = []
    for i in range(n):
        if i in best_indices:
            p = (1 - epsilon) / num_best + epsilon / n
        else:
            p = epsilon / n
        probs.append(p)
    return probs

def epsilon_greedy_choice(Q, epsilon):
    """
    Choose an action (machine index) using an epsilon-greedy strategy.
    """
    if random.random() < epsilon:
        return random.randint(0, len(Q) - 1)
    else:
        max_value = max(Q)
        candidates = [i for i, q in enumerate(Q) if q == max_value]
        return random.choice(candidates)

def update_Q(Q, counts, action, reward):
    """
    Update the estimated Q-value for the chosen action using the incremental update rule.
    """
    counts[action] += 1
    Q[action] = Q[action] + (reward - Q[action]) / counts[action]
    return Q

def run_simulation_house(num_trials=100, epsilon=0.1):
    machines = ['m1', 'm2', 'm3', 'm4']
    Q = [0.0 for _ in range(len(machines))]
    counts = [0 for _ in range(len(machines))]
    
    results = []
    for trial in range(1, num_trials + 1):
        # Compute the predicted probabilities for each machine based on the current Q-values.
        probs = compute_action_probabilities(Q, epsilon)
        # The house chooses the winning machine as the one with the lowest probability.
        min_prob = min(probs)
        candidates = [i for i, p in enumerate(probs) if p == min_prob]
        winning_machine_index = random.choice(candidates)
        
        # Simulated person selects a machine using epsilon-greedy strategy.
        action = epsilon_greedy_choice(Q, epsilon)
        
        # Determine if the person wins
        win = 1 if action == winning_machine_index else 0
        
        # Update the Q-value for the chosen machine.
        Q = update_Q(Q, counts, action, win)
        
        # Record trial result.
        results.append([trial, machines[action], "win" if win == 1 else "lose"])
    
    # Write the simulation results to a CSV file.
    with open('simulation_house.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "machine", "result"])
        writer.writerows(results)

if __name__ == "__main__":
    run_simulation_house()

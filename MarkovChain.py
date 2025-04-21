import numpy as np

def build_transition_matrices(sequences, num_machines=4):
    """
    Build two transition matrices conditioned on the outcome (win or loss).
    
    Each sequence is a list of tuples: (machine, outcome) where
      - machine is an integer in 0-3
      - outcome is 1 (win) or 0 (loss)
      
    This function computes:
      - win_matrix: P(next_machine | current_machine, win)
      - loss_matrix: P(next_machine | current_machine, loss)
    
    Parameters:
      sequences: list of sequences (each a list of (machine, outcome) tuples)
      num_machines: number of slot machines
      
    Returns:
      win_matrix, loss_matrix as numpy arrays (shape: num_machines x num_machines)
    """
    # Initialize counts for transitions based on outcome
    win_counts = np.zeros((num_machines, num_machines))
    loss_counts = np.zeros((num_machines, num_machines))
    
    for seq in sequences:
        # For each consecutive trial pair in the sequence
        for i in range(len(seq) - 1):
            current_machine, current_outcome = seq[i]
            next_machine, _ = seq[i+1]  # We only need the next machine choice.
            if current_outcome == 1:  # Win in the current trial
                win_counts[current_machine, next_machine] += 1
            else:  # Loss in the current trial
                loss_counts[current_machine, next_machine] += 1
                
    # Convert counts to probabilities for each machine row
    win_matrix = np.zeros_like(win_counts)
    loss_matrix = np.zeros_like(loss_counts)
    
    for i in range(num_machines):
        if win_counts[i].sum() > 0:
            win_matrix[i] = win_counts[i] / win_counts[i].sum() #Index each row of win_matrix to get all transition probabilities for machine i
        else:
            win_matrix[i] = np.ones(num_machines) / num_machines  # Uniform if no wins recorded

        if loss_counts[i].sum() > 0:
            loss_matrix[i] = loss_counts[i] / loss_counts[i].sum()
        else:
            loss_matrix[i] = np.ones(num_machines) / num_machines  # Uniform if no losses recorded
            
    return win_matrix, loss_matrix

def predict_next_machine(win_matrix, loss_matrix, current_machine, last_outcome):
    """
    Given the current machine and the outcome of that trial,
    predict the next machine as the one with the highest transition probability.
    
    Parameters:
      win_matrix: transition matrix for wins
      loss_matrix: transition matrix for losses
      current_machine: integer for the current machine (0 to num_machines-1)
      last_outcome: outcome of current trial (1 for win, 0 for loss)
      
    Returns:
      predicted next machine (integer)
    """
    if last_outcome == 1:
        probs = win_matrix[current_machine]
    else:
        probs = loss_matrix[current_machine]
    return np.argmax(probs)

# --- Example Usage ---

# Let's simulate some dummy data for 6 subjects with 100 trials each.
num_subjects = 6
trials_per_subject = 100
num_machines = 4

# Generate random sequences of (machine, outcome) pairs:
# For simplicity, we generate machines uniformly at random and outcomes randomly (0 or 1).
sequences = []
for _ in range(num_subjects):
    seq = []
    for _ in range(trials_per_subject):
        machine = np.random.randint(0, num_machines)
        outcome = np.random.randint(0, 2)  # 0: loss, 1: win
        seq.append((machine, outcome))
    sequences.append(seq)

# Build the transition matrices from all sequences
win_matrix, loss_matrix = build_transition_matrices(sequences, num_machines=num_machines)
print("Win Transition Matrix:")
print(win_matrix)
print("\nLoss Transition Matrix:")
print(loss_matrix)

# Suppose for a given subject, the last trial was on machine 2 and was a win.
current_machine = 2
last_outcome = 1
predicted_machine = predict_next_machine(win_matrix, loss_matrix, current_machine, last_outcome)
print(f"\nFor current machine {current_machine} with a win outcome, the predicted next machine is {predicted_machine}.")

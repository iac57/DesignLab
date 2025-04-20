import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

# Load data
data = pd.read_csv("slot_machine_data.csv")

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.1  # Exploration factor

def default_q_values():
    return np.zeros(4)

Q = defaultdict(default_q_values)

# Train Q-learning model
last_machine = None
last_win = 0

# Sort data to ensure order (just in case)
data = data.sort_values(by=["experiment_id", "trial"])

# Group by each person's experiment
for _, group in data.groupby("experiment_id"):
    group = group.reset_index(drop=True)
    last_machine = None
    last_win = 0

    for i in range(len(group) - 1):  # Exclude last trial
        current = group.loc[i]
        next_row = group.loc[i + 1]

        trial = current["trial"]
        machine = current["machine"]
        win = current["win"]

        state = (trial, last_machine, last_win)
        prediction = np.argmax(Q[state])

        # Determine reward based on next machine
        actual_next_machine = next_row["machine"]
        reward = 1 if prediction == actual_next_machine else -1

        next_state = (trial + 1, machine, win)

        # Q-learning update
        Q[state][prediction] = (1 - alpha) * Q[state][prediction] + alpha * (
            reward + gamma * np.max(Q[next_state])
        )

        # Update for next iteration
        last_machine = machine
        last_win = win

# Save trained model
with open("q_learning_model.pkl", "wb") as f:
    pickle.dump(Q, f)

print("Reinforcement learning model trained and saved.")

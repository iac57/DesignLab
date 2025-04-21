import os
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

def default_q_values():
    return np.zeros(4)

# Load trained Q-learning model
with open("q_learning_model.pkl", "rb") as f:
    Q = pickle.load(f)

# Folder with new test CSVs
folder = "testing_data"
csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]

overall_correct = 0
overall_total = 0

for file in csv_files:
    path = os.path.join(folder, file)
    data = pd.read_csv(path)

    correct = 0
    total = 0

    last_machine = None
    last_win = 0

    print(f"\nEvaluating: {file}")
    
    for _, row in data.iterrows():
        trial, machine, win = row["trial"], row["machine"], row["win"]
        state = (trial, last_machine, last_win)

        if state in Q:
            prediction = np.argmax(Q[state])
        else:
            prediction = np.random.choice(4)

        if prediction == machine:
            correct += 1
        total += 1

        last_machine = machine
        last_win = win

    accuracy = correct / total
    overall_correct += correct
    overall_total += total

    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")

# Print overall accuracy
overall_accuracy = overall_correct / overall_total
print(f"\nOverall accuracy across all files: {overall_accuracy:.2%} ({overall_correct}/{overall_total})")

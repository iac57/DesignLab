import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def process_last_trial_rows(input_file, machine_file, output_file):
    # Load CSV files
    df = pd.read_csv(input_file)
    machine_df = pd.read_csv(machine_file)

    # Get the last row for each trial number
    last_rows = df.groupby("Trial Number").last().reset_index()

    # Get the last recorded machine ID per trial
    last_machine_rows = machine_df.groupby("Trial Number").last().reset_index()

    # Merge the last rows of both datasets on Trial Number
    merged_df = pd.merge(last_rows, last_machine_rows[['Trial Number', 'Machine ID']], on="Trial Number", how="inner")

    # Compute torso angle and classify
    correct_predictions = 0
    total_trials = len(merged_df)
    results = []
    trial_correctness = []  

    for i, row in merged_df.iterrows():
        quaternion = [row["Orientation W"], row["Orientation X"], row["Orientation Y"], row["Orientation Z"]]
        euler_angles = R.from_quat(quaternion).as_euler('zyx', degrees=True)
        torso_angle = euler_angles[1]

        classification = 12 if torso_angle > 0 else 34
        correct = (row["Machine ID"] in [1, 2] and classification == 12) or (row["Machine ID"] in [3, 4] and classification == 34)

        if correct:
            correct_predictions += 1

        results.append({"Trial Number": row["Trial Number"], "Torso Classification": classification, "Machine ID": row["Machine ID"], "Correct": correct})
        trial_correctness.append((row["Trial Number"], int(correct)))

    # Compute accuracy
    accuracy = (correct_predictions / total_trials) * 100 if total_trials > 0 else 0
    

    # Save results to a new CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)

    trial_numbers, correctness = zip(*trial_correctness)
    plt.figure(figsize=(8, 5))
    plt.scatter(trial_numbers, correctness, marker='o', color='b', label="Correct (1) / Incorrect (0)")
    plt.xlabel("Trial Number")
    plt.ylabel("Correct Classification (1 = Yes, 0 = No)")
    plt.title("Classification Accuracy per Trial")
    plt.yticks([0, 1])  # Ensure only 0 and 1 are on y-axis
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()

    return accuracy, result_df

# Example usage:
input_path = "rigid_body_dataG2.csv"  # Update with actual file path
machine_path = "machine_play_log_G2.csv"  # Update with actual file path
output_path = "torso_classification_results.csv"

accuracy, processed_results = process_last_trial_rows(input_path, machine_path, output_path)
print(f"Accuracy: {accuracy:.2f}%")
# evaluate_predictions.py

import os
import csv
from machine_probability_predictor import MachineProbabilityPredictor

DATA_DIR = 'data'
N_LAST_TRIALS = 2  # <--- CHANGE THIS for different 'n'

def evaluate_prediction(file_path):
    predictor = MachineProbabilityPredictor()
    predictions = []  # Store (actual, predicted) tuples

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            actual_machine = int(row['machine'])
            prediction = predictor.predict()
            predicted_machine = max(prediction)
            predicted_machine=prediction.index(predicted_machine)

            predictions.append((actual_machine, predicted_machine))
            predictor.update(actual_machine)

    # Evaluate accuracy over the last N trials
    last_n = predictions[-N_LAST_TRIALS:]
    correct_predictions = sum(1 for actual, pred in last_n if actual == pred)
    accuracy = correct_predictions / len(last_n)

    return accuracy

def main():
    accuracies = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.csv') and filename.startswith('slot_machine_data_'):
            file_path = os.path.join(DATA_DIR, filename)
            acc = evaluate_prediction(file_path)
            print(f'{filename}: Last {N_LAST_TRIALS} Trials Accuracy = {acc:.2f}')
            accuracies.append(acc)

    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f'\nAverage Accuracy Over Last {N_LAST_TRIALS} Trials Across All Files: {avg_accuracy:.2f}')

if __name__ == '__main__':
    main()

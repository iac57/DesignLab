# evaluate_predictions.py

import os
import csv
from machine_probability_predictor import MachineProbabilityPredictor

DATA_DIR = 'data'
TOP_N = 2  # <--- Change this value (1 to 4)

def evaluate_prediction(file_path):
    predictor = MachineProbabilityPredictor()
    correct_predictions = 0
    total_predictions = 0

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            actual_machine = int(row['machine'])
            prediction = predictor.predict()

            # Get top-N predicted machines
            top_n_machine_ids = sorted(range(len(prediction)), key=lambda i: prediction[i], reverse=True)[:TOP_N]


            if actual_machine in top_n_machine_ids:
                correct_predictions += 1
            total_predictions += 1

            predictor.update(actual_machine)

    accuracy = correct_predictions / total_predictions
    return accuracy

def main():
    accuracies = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.csv') and filename.startswith('slot_machine_data_'):
            file_path = os.path.join(DATA_DIR, filename)
            acc = evaluate_prediction(file_path)
            print(f'{filename}: Top-{TOP_N} Accuracy = {acc:.2f}')
            accuracies.append(acc)

    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f'\nAverage Top-{TOP_N} Accuracy Across All Files: {avg_accuracy:.2f}')

if __name__ == '__main__':
    main()

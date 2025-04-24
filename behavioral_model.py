import pickle
import numpy as np
from machine_probability_predictor import MachineProbabilityPredictor
from repeater_check import RepeaterCheck

#when to run predict and when to run update? 
# run both immediately after a machine has been played

class BehavioralModel:
    def default_q_values():
        return np.zeros(4)

    def __init__(self, q_table_path="q_learning_model2.pkl", n=5):
        with open(q_table_path, "rb") as f:
            self.Q = pickle.load(f)

        self.predictor = MachineProbabilityPredictor()
        self.repeater = RepeaterCheck()

        self.P_correct_predictions = 0
        self.QL_correct_predictions = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        
        # Add tracking for recent predictions
        self.n = n  # Number of recent trials to consider
        self.recent_results = []  # List to store True/False for correct/incorrect predictions
        
        self.P_prediction = 5
        self.QL_prediction = 5
        self.check = 0

    def predict(self, trial, last_machine, last_win):
        state = (trial, last_machine, last_win)

        if state in self.Q:
            self.QL_prediction = np.argmax(self.Q[state])
        else:
            self.QL_prediction = np.random.choice(4)

        self.P_prediction = np.argmax(self.predictor.predict())
        check = self.repeater.check(last_machine)

        P_accuracy = self.P_correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
        QL_accuracy = self.QL_correct_predictions / self.total_predictions if self.total_predictions > 0 else 0

        if check == 1:
            return self.repeater.prev_machine
        elif trial < 5:
            return self.QL_prediction + 1
        elif P_accuracy > QL_accuracy:
            return self.P_prediction + 1
        else:
            return self.QL_prediction + 1

    def update(self, actual_machine, predicted_machine, trial_number):
        self.predictor.update(actual_machine)
        self.repeater.update(actual_machine)
        if trial_number == 1:
            return
            
        # Track if prediction was correct
        is_correct = (actual_machine == predicted_machine)
        
        # Update total counters
        if is_correct:
            self.correct_predictions += 1
        if actual_machine == self.P_prediction:
            self.P_correct_predictions += 1
        if actual_machine == self.QL_prediction:
            self.QL_correct_predictions += 1
        self.total_predictions += 1
        
        # Track recent results (for the last n trials)
        self.recent_results.append(is_correct)
        if len(self.recent_results) > self.n:
            self.recent_results.pop(0)  # Remove oldest result when exceeding n

    def get_accuracy(self):
        if self.check == 1:
            return 1
            
        # If we have recent results, calculate accuracy based on those
        if self.recent_results:
            recent_correct = sum(self.recent_results)
            return recent_correct / len(self.recent_results)
        
        # Fallback to overall accuracy if no recent results
        return self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0

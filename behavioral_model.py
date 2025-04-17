import pickle
import numpy as np
from machine_probability_predictor import MachineProbabilityPredictor
from repeater_check import RepeaterCheck

#when to run predict and when to run update? 
# run both immediately after a machine has been played

class BehavioralModel:
    def __init__(self, q_table_path="q_learning_model.pkl"):
        with open(q_table_path, "rb") as f:
            self.Q = pickle.load(f)

        self.predictor = MachineProbabilityPredictor()
        self.repeater = RepeaterCheck()

        self.P_correct_predictions = 0
        self.QL_correct_predictions = 0
        self.correct_predictions = 0
        self.total_predictions = 0

        self.P_prediction = 5
        self.QL_prediction = 5

    def predict(self, trial, last_machine, last_win):
        state = (trial, last_machine, last_win)

        if state in self.Q:
            self.QL_prediction = np.argmax(self.Q[state])
        else:
            self.QL_prediction = np.random.choice(4)

        self.P_prediction = self.predictor.predict()
        check = self.repeater.check()

        P_accuracy = self.P_correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
        QL_accuracy = self.QL_correct_predictions / self.total_predictions if self.total_predictions > 0 else 0

        if check == 1:
            return self.repeater.last_machine
        elif trial < 5:
            return self.QL_prediction
        elif P_accuracy > QL_accuracy:
            return self.P_prediction
        else:
            return self.QL_prediction

    def update(self, actual_machine, predicted_machine):
        if actual_machine == predicted_machine:
            self.correct_predictions += 1
        if actual_machine == self.P_prediction:
            self.P_correct_predictions += 1
        if actual_machine == self.QL_prediction:
            self.QL_correct_predictions += 1
        self.total_predictions += 1

        self.predictor.update(actual_machine)
        self.repeater.update(actual_machine)

    def get_accuracy(self):
        return self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0

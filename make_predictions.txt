#JUST IZZYS NOTES, NOT CODE TO BE RUN

# I want to make a class with two functions. One will return a prediction according to the strategy outlined
#below and the other will update the necessary values after each time 

import pickle
from machine_probability_predictor import MachineProbabilityPredictor
from repeater_check import RepeaterCheck
import numpy as np

# Load the trained Q-table
with open("q_learning_model.pkl", "rb") as f:
    Q = pickle.load(f)


# Q learning model stuff
if state in Q:
    QL_prediction = np.argmax(Q[state])
else:
    QL_prediction = np.random.choice(4)  # If state not in Q, fall back to random choice

print(f"Predicted machine: {prediction}")

#Probabilistic model stuff
predictor = MachineProbabilityPredictor()

#use these variables to track probabilistic data
correct_predictions = 0
total_predictions = 0
P_prediction = predictor.predict()
#do this after every trial result comes in 
predictor.update(actual_machine)
if actual_machine == P_prediction:
    correct_predictions += 1
total_predictions += 1

P_accuracy = correct_predictions/total_predictions

repeater = RepeaterCheck()
check = repeater.check()
#do this after every trial result comes in 
repeater.update(actual_machine)


# Weighting strategy: 1. If check is one, prediction = actual machine 
# 2. If trial number is less than 5, use QL_prediction.
# 3. If trial number greater than 5 and P_accuracy > 0.8, use P_prediction. 
# 4. Otherwise use QL_prediction



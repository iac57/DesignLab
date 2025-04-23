import joblib
import numpy as np
import pandas as pd
from motion_rnn import weighted_average, last_n_samples

class MotionPredictor:
    def __init__(self, model_path='torso_only_M3_T2_I3_classifier.pkl'):
        # Load the saved model and configuration
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.feature_strategy = model_data['strategy']
        self.feature_params = model_data['params']
        
    def process_frame(self, frame_data):
        """
        Process motion data and return a prediction.
        
        Args:
            frame_data: DataFrame containing the current frame's motion data
                       with columns matching the original feature columns
        """
        feature_columns = ['Position X', 'Position Y', 'Position Z', 
                   'Orientation W', 'Orientation X', 'Orientation Y', 'Orientation Z']
        # Convert frame data to DataFrame if it's not already
        if not isinstance(frame_data, pd.DataFrame):
            frame_data = pd.DataFrame([frame_data])
            
        # Extract features using the same strategy as training
        if self.feature_strategy == 'weighted_avg':
            features = weighted_average(frame_data, feature_columns)
        elif self.feature_strategy == 'last_n':
            n = self.feature_params.get('n', 20)
            features = last_n_samples(frame_data, feature_columns, n)
            # self.feature_columns does not match the feature columns format that last_n_samples expects
            #self.feature_columns has position and orientation columns for each sample so its 140 columns for last_n = 20
            #
        # Reshape features for prediction
        features = features.reshape(1, -1) #what does 1, -1 do? 1 row, -1 indicates the number of column = the number of elements
        
        # Make prediction -- np.argmax(probabilities)
        prediction = self.model.predict(features)[0] #essentially np.argmax(probabilities)
        probabilities = self.model.predict_proba(features)[0] #2D array of probabilities for each class (machine 1-4) per frame
        
        return {
            'prediction': prediction,
            'probabilities': dict(zip(self.model.classes_, probabilities)) #dictionary structure so that each probability is labeled with class (machine #)
        }

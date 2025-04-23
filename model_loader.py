import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import os

class MotionPredictor:
    def __init__(self, model_path="torso_only_M3_T2_I3_classifier.pkl"):
        """
        Initialize the motion predictor with a trained model.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model pickle file
        """
        # Load the model
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Prediction will not work.")
            self.model_data = None
        else:
            self.model_data = joblib.load(model_path)
            print(f"Loaded model from {model_path}")
            print(f"Model trained on subjects: {self.model_data.get('subjects', 'unknown')}")
        
        # Number of samples to use for feature extraction
        self.n_samples = 5
        if self.model_data and 'n_samples' in self.model_data:
            self.n_samples = self.model_data['n_samples']
            print(f"Using {self.n_samples} samples for prediction as specified in model")
        
        # Feature columns to use - matching motion_rnn.py
        self.feature_columns = ['Position X', 'Position Z', 
                              'Orientation W', 'Orientation X', 'Orientation Y', 'Orientation Z']
    
    def extract_torso_only(self, trial_df, n=5, torso_rigid_body_id=1):
        """
        Extract last n samples from torso rigid body (ID 1).
        
        Parameters:
        -----------
        trial_df : DataFrame
            DataFrame containing motion capture data
        n : int
            Number of samples to extract
        torso_rigid_body_id : int
            ID for the torso rigid body
            
        Returns:
        --------
        numpy.ndarray
            Flattened feature vector
        """
        torso_data = trial_df[trial_df['Rigid Body ID'] == torso_rigid_body_id]
        
        # Check if we have enough samples
        if len(torso_data) < n:
            print(f"Warning: Only {len(torso_data)} torso samples available, need {n}")
            # Padding with zeros if not enough samples
            if len(torso_data) == 0:
                return np.zeros(len(self.feature_columns) * n)
            
            # Use what we have and pad with zeros
            last_n = torso_data[self.feature_columns].values
            padding_needed = n - len(last_n)
            padded_array = np.vstack([np.zeros((padding_needed, len(self.feature_columns))), last_n])
            return padded_array.flatten()
        
        # Get the last n samples
        last_n = torso_data[self.feature_columns].tail(n).values
        return last_n.flatten()
    
    def process_frame(self, data_df):
        """
        Process the current frame data to make a prediction.
        
        Parameters:
        -----------
        data_df : DataFrame
            DataFrame containing motion capture data for the current frame
            
        Returns:
        --------
        dict
            Dictionary containing prediction and probabilities
        """
        if self.model_data is None:
            print("Error: No model loaded, cannot make prediction")
            return {"prediction": 1, "probabilities": {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}}
        
        try:
            # Get the latest trial number
            trial_number = data_df['Trial Number'].max()
            trial_data = data_df[data_df['Trial Number'] == trial_number]
            
            # Extract features using the torso_only method (consistent with motion_rnn.py)
            features = self.extract_torso_only(trial_data, n=self.n_samples)
            
            # Reshape for prediction
            features = features.reshape(1, -1)
            
            # Check for NaN values and replace with zeros
            if np.isnan(features).any():
                print(f"Warning: NaN values detected in feature vector, replacing with zeros")
                features = np.nan_to_num(features, nan=0.0)
            
            # Get the model
            model = self.model_data['model']
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Get probabilities
            probabilities = model.predict_proba(features)[0]
            
            # Build result dictionary
            result = {
                "prediction": int(prediction),
                "probabilities": {int(model.classes_[i]): float(prob) for i, prob in enumerate(probabilities)}
            }
            
            return result
        
        except Exception as e:
            print(f"Error making prediction: {e}")
            # Return a default prediction if something goes wrong
            return {"prediction": 1, "probabilities": {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}}

def get_predictor():
    """
    Factory function to create and return a motion predictor.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model file
        
    Returns:
    --------
    MotionPredictor
        Initialized predictor object
    """
    return MotionPredictor()
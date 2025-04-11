import joblib
import numpy as np
import pandas as pd
from motion_rnn import weighted_average, last_n_samples

class MotionPredictor:
    def __init__(self, model_path='motion_classifier.pkl'):
        # Load the saved model and configuration
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.feature_strategy = model_data['feature_strategy'] #last n, weighted average
        self.feature_params = model_data['feature_params']
        
    def process_frame(self, frame_data):
        """
        Process a single frame of motion data and return a prediction.
        
        Args:
            frame_data: DataFrame containing the current frame's motion data
                       with columns matching the original feature columns
        """
        # Convert frame data to DataFrame if it's not already
        if not isinstance(frame_data, pd.DataFrame):
            frame_data = pd.DataFrame([frame_data])
            
        # Extract features using the same strategy as training
        if self.feature_strategy == 'weighted_avg':
            features = weighted_average(frame_data, self.feature_columns)
        elif self.feature_strategy == 'last_n':
            n = self.feature_params.get('n', 2)
            features = last_n_samples(frame_data, self.feature_columns, n)
            
        # Reshape features for prediction
        features = features.reshape(1, -1)
        
        # Make prediction -- np.argmax(probabilities)
        prediction = self.model.predict(features)[0] #essentially np.argmax(probabilities)
        probabilities = self.model.predict_proba(features)[0] #2D array of probabilities for each class (machine 1-4) per frame
        
        return {
            'prediction': prediction,
            'probabilities': dict(zip(self.model.classes_, probabilities)) #dictionary structure so that each probability is labeled with class (machine #)
        }

# Example usage:
if __name__ == "__main__": #for testing the code here
    # Initialize the predictor
    predictor = MotionPredictor()
    
    # Example frame data (you would replace this with your actual motion data)
    example_frame = {
        'Position X': 0.1,
        'Position Y': 0.2,
        'Position Z': 0.3,
        'Orientation W': 0.4,
        'Orientation X': 0.5,
        'Orientation Y': 0.6,
        'Orientation Z': 0.7
    }
    
    # Get prediction
    result = predictor.process_frame(example_frame)
    print(f"Predicted class: {result['prediction']}")
    print("Class probabilities:", result['probabilities']) 
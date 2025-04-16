import joblib
from realtime_predictor import MotionPredictor

# Load model once when this module is imported
predictor = MotionPredictor()

#access the predictor
def get_predictor():
    return predictor
from realtime_predictor import MotionPredictor

# Load model once when this module is imported
predictor = MotionPredictor()
#print(predictor.feature_columns)
#access the predictor
def get_predictor():
    return predictor
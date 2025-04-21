import pandas as pd
from model_loader import get_predictor

predictor = get_predictor()
trial_file = 'rigid_body_data.csv' #this is the file that is being continuously updated by the motion capture system

data_df = pd.read_csv(trial_file) #since rigid_body_data is being continuously updated, i need to continuously read it into a dataframe
                        #transform data_df into the format that the model expects:
                        # for last_n, that'll be a 140-length feature vector 
result = predictor.process_frame(data_df) #this should work as it will just take the last n samples as a feature vector
                        # Debugging why the prediction is not working:
                        # Does the csv need to be in pandas dataframe format? Is it in this format? What does pd.read_csv do?
                        
print(result['prediction'])
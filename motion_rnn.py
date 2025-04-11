import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import re
import glob
import joblib

# Define feature columns globally
feature_columns = ['Position X', 'Position Y', 'Position Z', 
                  'Orientation W', 'Orientation X', 'Orientation Y', 'Orientation Z']

def weighted_average(trial_df, cols): #To-Do: Add parameter to control distribution of weights
    """
    Compute a weighted average over the specified columns.
    Later rows are given higher weight using a linear weighting.
    """
    n = len(trial_df)
    weights = np.linspace(1, n, n)
    weights /= weights.sum()
    return (trial_df[cols].values.T * weights).sum(axis=1)

def last_n_samples(trial_df, cols, n=2):
    """
    Extract the last n samples from the trial.
    Returns a flattened array of the features.
    """
    last_n = trial_df[cols].tail(n).values
    return last_n.flatten()

def process_subject_data(trial_file, label_file, feature_extractor='weighted_avg', **kwargs): #**kwargs (keyword args) makes function flexible. ** packs additional kwargs into dictionary. 
    """
    Process trial and label CSV files for a single subject.
    
    Parameters:
    -----------
    trial_file : str
        Path to the trial data CSV file
    label_file : str
        Path to the label data CSV file
    feature_extractor : str
        Strategy to extract features. Options:
        - 'weighted_avg': weighted average of all samples
        - 'last_n': last n samples (specify n in kwargs)
    kwargs : dict
        Additional arguments for the feature extraction method
    """
    # Load trial data and label data
    data_df = pd.read_csv(trial_file)
    labels_df = pd.read_csv(label_file)
    
    features_list = []
    trial_numbers = []
    
    # Group data by trial and compute features for each trial
    for trial_num, group in data_df.groupby('Trial Number'):
        group = group.sort_values('Frame Number')
        
        #Add more if-else statements for other feature extractors
        if feature_extractor == 'weighted_avg':
            features = weighted_average(group, feature_columns)
        elif feature_extractor == 'last_n':
            n = kwargs.get('n', 2)  # default to last 2 samples
            features = last_n_samples(group, feature_columns, n)
        else:
            raise ValueError(f"Unknown feature extractor: {feature_extractor}")
            
        features_list.append(features)
        trial_numbers.append(trial_num)
    
    # Create a DataFrame for the trial features. Expand for new feature extractors
    if feature_extractor == 'weighted_avg':
        feature_names = feature_columns
    elif feature_extractor == 'last_n':
        n = kwargs.get('n', 2)
        feature_names = [f"{col}_sample_{i}"
                        for i in range(n-1, -1, -1) #start, stop, step
                        for col in feature_columns]
    
    features_df = pd.DataFrame(features_list, columns=feature_names)
    features_df['Trial Number'] = trial_numbers
    
    # Merge with the labels
    merged_df = pd.merge(features_df, labels_df[['Trial Number', 'Machine ID']], on='Trial Number')
    return merged_df

# List to store processed DataFrames for all subjects
subject_dfs = []

# Get all trial CSV files
trial_files = glob.glob("rigid_body_data_*.csv")

# Specify the feature extraction strategy here
feature_strategy = 'last_n'  # or 'weighted_avg'
feature_params = {'n': 2}    # parameters for the chosen strategy

for trial_file in trial_files:
    match = re.search(r'rigid_body_data_([A-Z]\d+)\.csv', trial_file)
    if match:
        subject_id = match.group(1)
        label_file = f"machine_play_log_{subject_id}.csv"
        if os.path.exists(label_file):
            print(f"Processing subject {subject_id}...")
            subject_df = process_subject_data(trial_file, label_file, 
                                           feature_extractor=feature_strategy,
                                           **feature_params)
            subject_df['Subject ID'] = subject_id
            subject_dfs.append(subject_df)
        else:
            print(f"Label file {label_file} not found for subject {subject_id}.")
    else:
        print(f"File name {trial_file} does not match expected pattern.")

# Concatenate all subject dataframes
if subject_dfs:
    combined_df = pd.concat(subject_dfs, ignore_index=True)
    combined_df.to_csv("combined_dataset.csv", index=False)
    print("Combined dataset saved to combined_dataset.csv")
else:
    print("No subject files were processed.")

# Get the feature columns based on the chosen strategy
if feature_strategy == 'weighted_avg':
    X_columns = feature_columns
elif feature_strategy == 'last_n':
    n = feature_params['n']
    X_columns = [f"{col}_sample_{i}" 
                 for i in range(n-1, -1, -1) 
                 for col in feature_columns]

# Prepare data for training
X = combined_df[X_columns].values
y = combined_df['Machine ID'].values

# Split the data and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save classification report to disk
with open('classification_report.txt', 'w') as f:
    f.write(f"Classification Report for {feature_strategy} strategy:\n")
    f.write(f"Parameters: {feature_params}\n\n")
    f.write(classification_report(y_test, y_pred))
    
# Save the trained model and configuration
model_data = {
    'model': clf,
    'feature_columns': X_columns,
    'feature_strategy': feature_strategy,
    'feature_params': feature_params
}

joblib.dump(model_data, 'motion_classifier.pkl')
print("Model saved to motion_classifier.pkl")
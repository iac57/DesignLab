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

def weighted_average(trial_df, cols):
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

def extract_head_only(trial_df, n=5):
    """Extract last n samples from head rigid body (ID 1)"""
    head_data = trial_df[trial_df['Rigid Body ID'] == 2]
    return last_n_samples(head_data, feature_columns, n)

def extract_torso_only(trial_df, n=5):
    """Extract last n samples from torso rigid body (ID 2)"""
    torso_data = trial_df[trial_df['Rigid Body ID'] == 1]
    return last_n_samples(torso_data, feature_columns, n)

def extract_combined(trial_df, n=5):
    """Extract last n samples from both rigid bodies"""
    head_data = trial_df[trial_df['Rigid Body ID'] == 2]
    torso_data = trial_df[trial_df['Rigid Body ID'] == 1]
    
    head_features = last_n_samples(head_data, feature_columns, n)
    torso_features = last_n_samples(torso_data, feature_columns, n)
    
    return np.concatenate([head_features, torso_features])

def process_subject_data(trial_file, label_file, feature_extractor='weighted_avg', **kwargs):
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
        - 'head_only': last n samples from head only
        - 'torso_only': last n samples from torso only
        - 'combined': last n samples from both rigid bodies. will be the same as last_n most likely
    kwargs : dict
        Additional arguments for the feature extraction method
    """
    # Load trial data and label data
    data_df = pd.read_csv(trial_file)
    labels_df = pd.read_csv(label_file)
    
    # Filter data: only keep trials that have a label
    valid_trials = set(labels_df['Trial Number'])
    data_df = data_df[data_df['Trial Number'].isin(valid_trials)]
    
    features_list = []
    trial_numbers = []
    
    # Group data by trial and compute features for each trial
    for trial_num, group in data_df.groupby('Trial Number'):
        group = group.sort_values('Frame Number')
        
        if feature_extractor == 'weighted_avg':
            features = weighted_average(group, feature_columns)
        elif feature_extractor == 'last_n':
            n = kwargs.get('n', 20)
            features = last_n_samples(group, feature_columns, n)
        elif feature_extractor == 'head_only':
            n = kwargs.get('n', 10)
            features = extract_head_only(group, n)
        elif feature_extractor == 'torso_only':
            n = kwargs.get('n', 10)
            features = extract_torso_only(group, n)
        elif feature_extractor == 'combined':
            n = kwargs.get('n', 10)
            features = extract_combined(group, n)
        else:
            raise ValueError(f"Unknown feature extractor: {feature_extractor}")
            
        features_list.append(features)
        trial_numbers.append(trial_num)
    
    # Create a DataFrame for the trial features
    if feature_extractor == 'weighted_avg':
        feature_names = feature_columns
    elif feature_extractor == 'last_n':
        n = kwargs.get('n', 20)
        feature_names = [f"{col}_sample_{i}"
                         for i in range(n-1, -1, -1)
                         for col in feature_columns]
    elif feature_extractor in ['head_only', 'torso_only']:
        n = kwargs.get('n', 10)
        feature_names = [f"{col}_sample_{i}"
                         for i in range(n-1, -1, -1)
                         for col in feature_columns]
    elif feature_extractor == 'combined':
        n = kwargs.get('n', 10)
        feature_names = [f"head_{col}_sample_{i}"
                         for i in range(n-1, -1, -1)
                         for col in feature_columns]
        feature_names.extend([f"torso_{col}_sample_{i}"
                              for i in range(n-1, -1, -1)
                              for col in feature_columns])
    
    features_df = pd.DataFrame(features_list, columns=feature_names)
    features_df['Trial Number'] = trial_numbers
    
    # Merge with the labels (only keeping the relevant label columns)
    merged_df = pd.merge(features_df, labels_df[['Trial Number', 'Machine ID']], on='Trial Number')
    return merged_df

def train_and_evaluate(X, y, strategy_name):
    """Train and evaluate MLP classifier"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train classifier
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"\nResults for {strategy_name}:")
    print(classification_report(y_test, y_pred))
    
    return clf, report, report['accuracy'], y_test, y_pred

def main():
    # Define target subjects and feature extraction strategies
    target_subjects = ['M2', 'I2']
    strategies = [
        ('last_n', {'n': 20}),
        ('head_only', {'n': 10}),
        ('torso_only', {'n': 10}),
        ('combined', {'n': 10})
    ]
    
    best_accuracy = 0
    best_model_data = None
    best_model_info = None

    # Process each strategy individually
    for strategy, params in strategies:
        print(f"\n=== Processing strategy: {strategy} ===")
        subject_dfs = []
        trial_files = glob.glob("rigid_body_data_*.csv")
        
        # Process each target subject's data for the current strategy
        for trial_file in trial_files:
            match = re.search(r'rigid_body_data_([A-Z]\d+)\.csv', trial_file)
            if match:
                subject_id = match.group(1)
                if subject_id not in target_subjects:
                    continue
                    
                label_file = f"machine_play_log_{subject_id}.csv"
                if os.path.exists(label_file):
                    print(f"Processing subject {subject_id} for strategy {strategy}")
                    subject_df = process_subject_data(trial_file, label_file,
                                                      feature_extractor=strategy,
                                                      **params)
                    # Tag the DataFrame with the subject ID (optional)
                    subject_df['Subject ID'] = subject_id
                    subject_dfs.append(subject_df)
                else:
                    print(f"Label file {label_file} not found for subject {subject_id}.")
            else:
                print(f"File name {trial_file} does not match expected pattern.")
        
        # Combine data from all subjects for the current strategy, then train/evaluate
        if subject_dfs:
            combined_df = pd.concat(subject_dfs, ignore_index=True)
            print(f"Combined data for strategy {strategy} has shape: {combined_df.shape}")
            
            # Get feature columns (exclude metadata)
            feature_cols = [col for col in combined_df.columns 
                            if col not in ['Trial Number', 'Machine ID', 'Subject ID']]
            
            X = combined_df[feature_cols].values
            y = combined_df['Machine ID'].values
            
            clf, report, accuracy, y_test, y_pred = train_and_evaluate(X, y, f"{strategy} (combined subjects)")
            
            # Save the classification report to a file
            with open(f"classification_results_combined_{strategy}.txt", "w") as f:
                f.write(f"Results for combined subjects using {strategy} strategy\n")
                f.write(f"Parameters: {params}\n\n")
                f.write(classification_report(y_test, y_pred))
            
            # Save the trained model for this strategy
            model_data = {
                'model': clf,
                'feature_columns': feature_cols,
                'strategy': strategy,
                'params': params,
                'subjects': target_subjects
            }
            joblib.dump(model_data, f'motion_classifier_combined_{strategy}.pkl')
            
            # Track the best model based on accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_data = model_data
                best_model_info = {
                    'strategy': strategy,
                    'params': params,
                    'accuracy': accuracy,
                    'subjects': target_subjects
                }
        else:
            print(f"No valid data found for strategy {strategy}.")

    # Save the best performing model (across all strategies)
    if best_model_data is not None:
        print("\n=== Saving best model ===")
        print(f"Strategy: {best_model_info['strategy']}")
        print(f"Parameters: {best_model_info['params']}")
        print(f"Subjects: {best_model_info['subjects']}")
        print(f"Accuracy: {best_model_info['accuracy']:.4f}")
        
        joblib.dump(best_model_data, 'best_motion_classifier_combined.pkl')
        
        with open('best_model_info_combined.txt', 'w') as f:
            f.write("Best Model Information:\n")
            f.write(f"Strategy: {best_model_info['strategy']}\n")
            f.write(f"Parameters: {best_model_info['params']}\n")
            f.write(f"Subjects: {best_model_info['subjects']}\n")
            f.write(f"Accuracy: {best_model_info['accuracy']:.4f}\n")

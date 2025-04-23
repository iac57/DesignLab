import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import os
import re
import glob
import joblib
import argparse

# Global variables
head_rigid_body_id = 2  # ID for the head rigid body
torso_rigid_body_id = 1  # ID for the torso rigid body
num_samples = 5

# Define feature columns globally
feature_columns = ['Position X', 'Position Z', 
                   'Orientation W', 'Orientation X', 'Orientation Y', 'Orientation Z']

def last_n_samples(trial_df, cols, n=5):
    """
    Extract the last n samples from the trial.
    Returns a flattened array of the features.
    """
    last_n = trial_df[cols].tail(n).values
    return last_n.flatten()

def extract_torso_only(trial_df, n=5, torso_rigid_body_id=1):
    """Extract last n samples from torso rigid body (ID 1)"""
    torso_data = trial_df[trial_df['Rigid Body ID'] == torso_rigid_body_id]
    return last_n_samples(torso_data, feature_columns, n)

def process_subject_data(trial_file, label_file, n=5):
    """
    Process trial and label CSV files for a single subject using torso-only strategy.
    
    Parameters:
    -----------  
    trial_file : str
        Path to the trial data CSV file
    label_file : str
        Path to the label data CSV file
    n : int
        Number of samples to extract
    
    Returns:
    --------
    DataFrame with features and labels for each trial
    """
    # Load trial data and label data
    data_df = pd.read_csv(trial_file)
    
    # The fix: properly handle files without headers
    labels_df = pd.read_csv(label_file, 
                       header=None,  # Explicitly no header
                       names=['Trial Number', 'Frame Number', 'Machine ID', 'Win/Lose', 'Blank'],
                       dtype={'Trial Number': int, 'Frame Number': int, 'Machine ID': int},
                       usecols=[0, 2])  # Use column indices instead of names
    
    # Rename the columns after loading
    labels_df.columns = ['Trial Number', 'Machine ID']
    
    # Filter data: only keep trials that have a label
    valid_trials = set(labels_df['Trial Number'])
    data_df = data_df[data_df['Trial Number'].isin(valid_trials)]
    
    features_list = []
    trial_numbers = []
    
    # Group data by trial and compute features for each trial
    for trial_num, group in data_df.groupby('Trial Number'):
        group = group.sort_values('Frame Number')
        
        # Only using extract_torso_only strategy
        features = extract_torso_only(group, n=n)
        
        features_list.append(features)
        trial_numbers.append(trial_num)
    
    # Create a DataFrame for the trial features
    feature_names = [f"{col}_sample_{i}"
                     for i in range(n-1, -1, -1)
                     for col in feature_columns]
    
    features_df = pd.DataFrame(features_list, columns=feature_names)
    features_df['Trial Number'] = trial_numbers
    
    # Merge with the labels (only keeping the relevant label columns)
    merged_df = pd.merge(features_df, labels_df[['Trial Number', 'Machine ID']], on='Trial Number')
    return merged_df

def assemble_dataset(trial_files, label_files, n_samples=5):
    """
    Assemble a complete dataset with features, labels, and subject IDs.
    
    Returns:
    --------
    X : numpy array
        Feature matrix
    y : numpy array
        Labels
    groups : numpy array
        Subject IDs for each sample (for cross-validation)
    feature_cols : list
        Names of feature columns
    """
    subject_dfs = []
    
    for t_file, l_file in zip(trial_files, label_files):
        # Extract subject ID
        match = re.search(r'rigid_body_data_(.+?)\.csv', t_file)
        subject = match.group(1) if match else t_file
        
        print(f"Processing subject {subject}")
        
        # Process this subject's data
        subject_df = process_subject_data(t_file, l_file, n=n_samples)
        subject_df['Subject ID'] = subject
        
        subject_dfs.append(subject_df)
    
    # Combine all subjects' data
    if not subject_dfs:
        raise ValueError("No valid subjects found")
        
    combined_df = pd.concat(subject_dfs, ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    
    # Get feature columns
    feature_cols = [col for col in combined_df.columns 
                    if col not in ['Trial Number', 'Machine ID', 'Subject ID']]
    
    # Prepare data matrices
    X = combined_df[feature_cols].values
    y = combined_df['Machine ID'].values
    groups = combined_df['Subject ID'].values
    
    return X, y, groups, feature_cols

def filter_files_by_subjects(all_files, subject_ids):
    """Filter file list to include only specified subject IDs"""
    filtered_files = []
    for f in all_files:
        match = re.search(r'rigid_body_data_([A-Z]\d+)\.csv', f)
        if match and match.group(1) in subject_ids:
            filtered_files.append(f)
    return filtered_files

def main(subject_ids=None):
    """
    Main function to run the model training with optional subject ID filtering.
    
    Parameters:
    -----------
    subject_ids : list of str
        List of subject IDs to include (e.g., ['M3', 'T2'])
        If None, all available subjects will be used.
    """
    # Find all trial files
    all_trials = glob.glob("rigid_body_data_*.csv")
    
    # Filter by subject IDs if specified
    if subject_ids:
        print(f"Filtering for subjects: {', '.join(subject_ids)}")
        trials = filter_files_by_subjects(all_trials, subject_ids)
    else:
        trials = all_trials
        print("Using all available subjects")
        
    # No subjects found
    if not trials:
        print("No matching trial files found!")
        return
        
    # Find corresponding label files
    labels = []
    valid_trials = []
    
    for f in trials:
        match = re.search(r'rigid_body_data_([A-Z]\d+)\.csv', f)
        if match:
            subject_id = match.group(1)
            label_file = f"machine_play_log_{subject_id}.csv"
            
            if os.path.exists(label_file):
                valid_trials.append(f)
                labels.append(label_file)
                print(f"Found data for subject {subject_id}")
            else:
                print(f"Label file {label_file} not found for subject {subject_id}")
    
    if not valid_trials:
        print("No valid trial files found")
        return
        
    print(f"Processing {len(valid_trials)} subjects")
    
    # Assemble dataset
    X, y, groups, feature_cols = assemble_dataset(valid_trials, labels)
    
    # Check for NaN values
    if np.isnan(X).any():
        print("Warning: NaN values detected in the feature matrix")
        print(f"NaN count: {np.isnan(X).sum()}")
        print("Replacing NaNs with zeros")
        X = np.nan_to_num(X, nan=0.0)
    
    # Leave-one-subject-out cross-validation
    n_subjects = len(np.unique(groups))
    cv = GroupKFold(n_splits=n_subjects)
    
    # Create pipeline
    pipeline = Pipeline([
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=42,
            alpha=0.01  # Add regularization to help with generalization
        ))
    ])
    
    # Perform cross-validation
    print("\n=== Leave-One-Subject-Out Cross-Validation ===")
    scores = cross_val_score(pipeline, X, y, groups=groups, cv=cv)
    
    print("Subject-wise accuracy scores:", scores)
    print(f"Mean accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Detailed prediction analysis for each fold
    print("\n=== Detailed Leave-One-Subject-Out Analysis ===")
    unique_subjects = np.unique(groups)
    all_y_test = []
    all_y_pred = []
    
    for i, subject in enumerate(unique_subjects):
        # Split data for this subject
        test_mask = (groups == subject)
        train_mask = ~test_mask
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Train and predict
        clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, 
                           random_state=42, alpha=0.01)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Save results
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        
        # Print subject-specific results
        accuracy = np.mean(y_test == y_pred)
        print(f"\nSubject {subject} test accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
    
    # Overall performance across all folds
    print("\n=== Overall Cross-Validation Performance ===")
    print(classification_report(all_y_test, all_y_pred))
    
    # Set output filenames to use "all_data" instead of subject IDs
    output_model = 'torso_only_all_data_classifier.pkl'
    output_report = 'torso_only_all_data_cv_report.txt'
    
    # Train final model on all data
    print(f"\n=== Training Final Model on All Data ===")
    final_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, 
                               random_state=42, alpha=0.01)
    final_model.fit(X, y)
    
    # Save the trained model
    model_data = {
        'model': final_model,
        'feature_columns': feature_cols,
        'n_samples': num_samples,
        'subjects': list(unique_subjects)
    }
    
    joblib.dump(model_data, output_model)
    print(f"Model saved as {output_model}")
    
    # Save cross-validation report
    with open(output_report, "w") as f:
        f.write(f"Subjects included: {list(unique_subjects)}\n\n")
        f.write("=== Leave-One-Subject-Out Cross-Validation ===\n")
        f.write(f"Subject-wise accuracy scores: {scores}\n")
        f.write(f"Mean accuracy: {scores.mean():.4f} ± {scores.std():.4f}\n\n")
        f.write("=== Overall Cross-Validation Performance ===\n")
        f.write(classification_report(all_y_test, all_y_pred))
    
    print(f"Cross-validation report saved as {output_report}")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Train a motion prediction model with specified subjects')
    parser.add_argument('--subjects', nargs='+', help='Subject IDs to include (e.g., M3 T2 I3)')
    args = parser.parse_args()
    
    # If run directly with arguments, use command line args
    if args.subjects:
        main(subject_ids=args.subjects)
    else:
        # Default to None to process all available subjects
        main(subject_ids=None)

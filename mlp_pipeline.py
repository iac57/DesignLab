import os
import glob
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    cross_val_score,
    GroupKFold,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
import joblib

from motion_rnn import (
    process_subject_data,
    last_n_samples,
    weighted_average,
    extract_head_only,
    extract_torso_only,
    extract_combined,
    feature_columns
)

class MotionMLPPipeline:
    def __init__(
        self,
        strategy='combined',
        n_samples=10,
        pca_components=0.95,
        hidden_layers=(100, 50),
        random_state=42
    ):
        self.strategy = strategy
        self.n = n_samples
        self.pca = PCA(n_components=pca_components, random_state=random_state)
        self.clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=1000,
            random_state=random_state
        )

    def _extract(self, trial_df):
        if self.strategy == 'weighted_avg':
            return weighted_average(trial_df, feature_columns)
        elif self.strategy == 'last_n':
            return last_n_samples(trial_df, feature_columns, n=self.n)
        elif self.strategy == 'head_only':
            return extract_head_only(trial_df, n=self.n)
        elif self.strategy == 'torso_only':
            return extract_torso_only(trial_df, n=self.n)
        elif self.strategy == 'combined':
            return extract_combined(trial_df, n=self.n)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

    def _assemble_dataset(self, trial_files, label_files):
        X_list, y_list, group_list = [], [], []
        for t_file, l_file in zip(trial_files, label_files):
            match = re.search(r'rigid_body_data_(.+?)\.csv', t_file)
            subject = match.group(1) if match else t_file
            df = process_subject_data(t_file, l_file, feature_extractor=self.strategy, n=self.n)
            feats = df.drop(columns=['Trial Number', 'Machine ID']).values
            labs  = df['Machine ID'].values
            X_list.append(feats)
            y_list.append(labs)
            group_list.append(np.array([subject] * len(labs)))
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        groups = np.concatenate(group_list)
        return X, y, groups

    def fit(self, trial_files, label_files):
        X, y, _ = self._assemble_dataset(trial_files, label_files)
        Xp = self.pca.fit_transform(X)
        self.clf.fit(Xp, y)
        return self

    def check_class_distribution(self, trial_files, label_files):
        """
        Prints and returns the class distribution for Machine ID across the assembled dataset.
        """
        _, y, _ = self._assemble_dataset(trial_files, label_files)
        unique, counts = np.unique(y, return_counts=True)
        dist = pd.Series(counts, index=unique, name='counts')
        print("Class distribution (Machine ID counts):")
        print(dist)
        return dist

    def cross_validate(self, trial_files, label_files):
        """
        Leave-one-subject-out cross-validation using GroupKFold.
        """
        X, y, groups = self._assemble_dataset(trial_files, label_files)
        pipeline = Pipeline([('pca', self.pca), ('clf', self.clf)])
        n_subjects = len(np.unique(groups))
        cv = GroupKFold(n_splits=n_subjects)
        scores = cross_val_score(pipeline, X, y, groups=groups, cv=cv, n_jobs=-1)
        print("Leave-One-Subject-Out CV accuracy:", scores)
        print(f"Mean accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        return scores

    def grid_search(self, trial_files, label_files, param_grid, n_splits=5):
        """
        Inner stratified k-fold grid search to narrow hyperparameters.
        """
        X, y, _ = self._assemble_dataset(trial_files, label_files)
        pipeline = Pipeline([('pca', self.pca), ('clf', self.clf)])
        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        gs = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            n_jobs=-1,
            return_train_score=False
        )
        gs.fit(X, y)
        print("Best params:", gs.best_params_)
        print("Best CV score:", gs.best_score_)
        return gs

    def process_trial(self, trial_df):
        vec = self._extract(trial_df)
        vec_p = self.pca.transform(vec.reshape(1, -1))
        pred = self.clf.predict(vec_p)[0]
        proba = self.clf.predict_proba(vec_p)[0]
        return {'prediction': pred, 'probabilities': dict(zip(self.clf.classes_, proba))}

    def save(self, path_prefix='mlp_pipeline'):
        joblib.dump({'pca': self.pca, 'clf': self.clf}, f'{path_prefix}.pkl')

    @classmethod #A "constructor" to build a new pipeline object from pickled file. Adjust model_path as needed.
    def load(cls, model_path='mlp_pipeline.pkl'):
        data = joblib.load(model_path)
        obj = cls()
        obj.pca = data['pca']
        obj.clf = data['clf']
        return obj

# Example usage:
if __name__ == '__main__':
    # Specify which subjects to include
    subject_ids = ['M1', 'M2', 'I1']

    all_trials = glob.glob('rigid_body_data_*.csv')
    trials = []
    for f in all_trials:
        match = re.search(r'rigid_body_data_(.+?)\.csv', f)
        if match and match.group(1) in subject_ids:
            trials.append(f)
    trials = sorted(trials)

    labels = []
    for t in trials:
        sid = re.search(r'rigid_body_data_(.+?)\.csv', t).group(1)
        labels.append(f'machine_play_log_{sid}.csv')

    pipeline = MotionMLPPipeline(strategy='combined', n_samples=10)

    # Check class distribution before deciding on mitigation
    pipeline.check_class_distribution(trials, labels)

    # Narrow hyperparameters via stratified grid search
    param_grid = {
        'clf__hidden_layer_sizes': [(50,), (100, 50), (200, 100, 50)],
        'pca__n_components': [0.90, 0.95, 0.99]
    }
    gs = pipeline.grid_search(trials, labels, param_grid)

    # Evaluate subject‑out generalization
    pipeline.cross_validate(trials, labels)

    # Fit final model and save
    pipeline.fit(trials, labels)
    pipeline.save()

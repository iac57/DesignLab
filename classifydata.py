import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Helper function to compute first and second derivatives.
def compute_derivatives(signal, dt):
    first_derivative = np.gradient(signal, dt)
    second_derivative = np.gradient(first_derivative, dt)
    return first_derivative, second_derivative

# Create a summary feature vector for a single trial.
# trial_data is a dict with keys: 'COM_x', 'COM_z', 'Gaze_angle', 'Torso_angle'
# Each key maps to a 1D array sampled at 10 Hz.
# The function divides the trial into 1-second windows (10 samples/window)
# and then uses the last n windows to build a fixed-length summary vector.
def create_summary_vector(trial_data, n_seconds, dt=0.1):
    window_size = int(1 / dt)  # 10 samples per window
    num_samples = len(trial_data['COM_x'])
    num_windows = num_samples // window_size
    # Select the last n windows (ignoring the initial seconds)
    selected_windows = range(num_windows - n_seconds, num_windows)
    summary_features = []
    
    # Loop over each selected window
    for window_idx in selected_windows:
        start = window_idx * window_size
        end = start + window_size
        window_features = []
        # Process each base feature
        for feature in ['COM_x', 'COM_z', 'Gaze_angle', 'Torso_angle']:
            data_window = trial_data[feature][start:end]
            avg_val = np.mean(data_window)
            # Compute derivatives for the window
            first_deriv, second_deriv = compute_derivatives(data_window, dt)
            avg_first = np.mean(first_deriv)
            avg_second = np.mean(second_deriv)
            # Append three numbers per feature: average, first derivative, second derivative
            window_features.extend([avg_val, avg_first, avg_second])
        summary_features.extend(window_features)
    
    # Integrate an external behavioral model prediction as an extra feature.
    # (Replace np.random.rand() with your actual behavioral prediction.)
    behavioral_prediction = np.random.rand()
    summary_features.append(behavioral_prediction)
    
    return np.array(summary_features)

# -------------------------
# Simulate Dataset Creation
# -------------------------

num_trials = 100  # Number of trials
n_seconds = 5     # Use the last 5 seconds of data per trial
dt = 0.1          # Sampling interval corresponding to 10 Hz. Our data is currently collected at 0.5 Hz.

X = []  # Feature matrix
y = []  # Labels: slot machine choices (e.g., 1, 2, 3, or 4)

for i in range(num_trials):
    # Simulate a trial duration of 10 seconds (100 samples at 10 Hz)
    T = int(10 / dt)
    trial_data = {
        'COM_x': np.random.randn(T) + i * 0.01,
        'COM_z': np.random.randn(T) + i * 0.01,
        'Gaze_angle': np.random.randn(T),
        'Torso_angle': np.random.randn(T)
    }
    summary_vector = create_summary_vector(trial_data, n_seconds, dt=dt)
    X.append(summary_vector)
    # Simulate a label (slot machine choice between 1 and 4)
    y.append(np.random.randint(1, 5))

X = np.array(X)
y = np.array(y)

# -------------------------
# Train/Test Split and Modeling
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a LinearSVC model with regularization (L2 penalty).
# The regularization parameter C controls the trade-off between fitting the data and keeping coefficients small.
clf = LinearSVC(C=1.0, penalty='l2', dual=False, random_state=42, max_iter=10000)
clf.fit(X_train, y_train)

# Evaluate the model.
y_pred = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def extract_features(X):
    """
    Extracts time-domain features from raw ECG beat waveforms.
    
    Parameters:
    X (numpy.ndarray): Raw ECG beats, shape [samples, 200]
    
    Returns:
    numpy.ndarray: Feature matrix, shape [samples, 5]
    """
    print("Extracting features from raw waveforms...")
    
    # Calculate features along the time axis (axis=1) for each beat
    max_amp = np.max(X, axis=1)
    min_amp = np.min(X, axis=1)
    mean_val = np.mean(X, axis=1)
    std_val = np.std(X, axis=1)
    area_val = np.sum(np.abs(X), axis=1)
    
    # Stack the 1D arrays as columns to form a 2D matrix of shape [samples, 5]
    features = np.column_stack((max_amp, min_amp, mean_val, std_val, area_val))
    
    return features

if __name__ == "__main__":
    # 1. Load the data
    print("Loading data")
    try:
        X_raw = np.load('X_data.npy')
        y = np.load('y_data.npy')
    except FileNotFoundError:
        print("Error: X_data.npy or y_data.npy not found")
        exit(1)
        
    print(f"Loaded raw data shape: {X_raw.shape}")
    print(f"Loaded labels shape: {y.shape}")
    
    # 2. Feature Extraction
    X_features = extract_features(X_raw)
    print(f"Feature matrix shape after extraction: {X_features.shape}")
    
    # 3. Data Split (80% training, 20% testing)
    print("\nSplitting data into training and testing sets (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Train the Random Forest Model
    print("Training Random Forest Classifier")
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 5. Evaluation
    print("\nEvaluating model on the test set\n")
    y_pred = rf_model.predict(X_test)
    
    print("-" * 40)
    print("Classification Report:")
    print("-" * 40)
    print(classification_report(y_test, y_pred))
    
    print("-" * 40)
    print("Confusion Matrix:")
    print("-" * 40)
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(rf_model, 'rf_model.joblib')
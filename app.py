import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import random
import joblib

# --- Feature Extraction for Random Forest ---
def extract_features_single(beat):
    """
    Counts the needed time-domain features for the Random Forest
    """
    feat_max = np.max(beat)
    feat_min = np.min(beat)
    feat_mean = np.mean(beat)
    feat_std = np.std(beat)
    feat_sum_abs = np.sum(np.abs(beat))
    
    return np.array([[feat_max, feat_min, feat_mean, feat_std, feat_sum_abs]])

# --- CNN Model Definition ---
class ECG_CNN1D(nn.Module):
    def __init__(self):
        super(ECG_CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 25, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Caching Data and Models ---
@st.cache_resource
def load_resources():
    try:
        # Data
        X = np.load('X_data.npy')
        y = np.load('y_data.npy')
        
        # 1D-CNN Model
        model_cnn = ECG_CNN1D()
        model_cnn.load_state_dict(torch.load('cnn_model.pth', map_location=torch.device('cpu'), weights_only=True))
        model_cnn.eval()
        
        # Random Forest Model
        model_rf = joblib.load('rf_model.joblib')
        
        return X, y, model_cnn, model_rf
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

# --- Main Application ---
def main():
    st.set_page_config(page_title="ECG Arrhythmia Detector", layout="wide")
    
    st.title("Clinical ECG Arrhythmia Detector")
    
    
    X, y, model_cnn, model_rf = load_resources()
    
    # Session state
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = random.randint(0, len(X) - 1)
        
    # --- Sidebar Controls ---
    st.sidebar.header("Choose")
    
    # Choice system
    beat_filter = st.sidebar.radio(
        "Which beats",
        ("Random (All)", "Only Normal(N)", "Only Premature Ventricular Contractions (PVC)")
    )
    
    if st.sidebar.button("Pick a new beat", width="stretch"):
        if beat_filter == "Only Normal(N)":
            valid_indices = np.where(y == 'N')[0]
            st.session_state.current_idx = random.choice(valid_indices)
        elif beat_filter == "Only Premature Ventricular Contractions (PVC)":
            valid_indices = np.where(y == 'V')[0]
            st.session_state.current_idx = random.choice(valid_indices)
        else:
            st.session_state.current_idx = random.randint(0, len(X) - 1)
            
    # Get a random valid beat
    idx = st.session_state.current_idx
    beat = X[idx]
    true_label = y[idx]
    
    # --- Inference: 1D-CNN ---
    beat_tensor = torch.tensor(beat, dtype=torch.float32).view(1, 1, -1)
    with torch.no_grad():
        logit = model_cnn(beat_tensor).item()
    prob_v_cnn = 1.0 / (1.0 + np.exp(-logit))
    
    cnn_pred = 'V' if prob_v_cnn > 0.5 else 'N'
    cnn_conf = prob_v_cnn * 100 if cnn_pred == 'V' else (1.0 - prob_v_cnn) * 100

    # --- Inference: Random Forest ---
    features = extract_features_single(beat)
    rf_pred = model_rf.predict(features)[0]
    
    # Get the probabilities. predict_proba returns [N_prob, V_prob] if the classes are in alphabetical order
    rf_probs = model_rf.predict_proba(features)[0]
    rf_conf = np.max(rf_probs) * 100

    # --- Display Results ---
    st.markdown("### Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Ground Truth:**")
        if true_label == 'N':
            st.success("Normal beat (N)")
        else:
            st.error("PVC (V)")
            
    with col2:
        st.markdown("**1D-CNN (Deep Learning):**")
        if cnn_pred == 'N':
            st.success(f"Normal (Confidence: {cnn_conf:.1f}%)")
        else:
            st.error(f"PVC (Confidence: {cnn_conf:.1f}%)")
            
    with col3:
        st.markdown("**Random Forest (Traditional ML):**")
        if rf_pred == 'N':
            st.success(f"Normal (Confidence: {rf_conf:.1f}%)")
        else:
            st.error(f"PVC (Confidence: {rf_conf:.1f}%)")

    # --- Plotly Visualization ---
    st.markdown("### ECG Waveform (200 samples)")
    
    time_ms = np.arange(-90, 110) * (1000 / 360)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_ms, 
        y=beat, 
        mode='lines', 
        name='ECG Signal',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="R-peak (t=0)")
    
    fig.update_layout(
        xaxis_title="Time around R-peak (ms)",
        yaxis_title="Amplitude (mV)",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_white"
    )
    
    st.plotly_chart(fig, width="stretch")

if __name__ == "__main__":
    main()
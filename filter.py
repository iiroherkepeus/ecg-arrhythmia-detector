import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def filter_ecg_signal(signal_data, fs):
    """
    Applies a Butterworth high-pass filter (0.5 Hz) and a low-pass filter (40 Hz) 
    to an ECG signal to remove baseline wander and high-frequency noise.
    
    Uses filtfilt for zero-phase filtering to prevent phase distortion, 
    which is critical for preserving the exact shape of the ECG complex.
    """
    # 1. High-pass filter configuration (0.5 Hz cut-off)
    # Removes baseline wander (e.g., breathing artifacts)
    hp_cutoff = 0.5
    b_high, a_high = butter(N=4, Wn=hp_cutoff, btype='highpass', fs=fs)
    signal_highpassed = filtfilt(b_high, a_high, signal_data)
    
    # 2. Low-pass filter configuration (40 Hz cut-off)
    # Removes muscle noise (EMG artifacts) and powerline interference (50/60 Hz)
    lp_cutoff = 40.0
    b_low, a_low = butter(N=4, Wn=lp_cutoff, btype='lowpass', fs=fs)
    signal_filtered = filtfilt(b_low, a_low, signal_highpassed)
    
    return signal_filtered

# --- Main execution block ---

# Define the record and database
record_name = '100'
database_dir = 'mitdb'
fs = 360 # Sampling frequency for MIT-BIH
duration_seconds = 10
total_samples = fs * duration_seconds

# Load the raw ECG signal (first 10 seconds)
signals, fields = wfdb.rdsamp(record_name, pn_dir=database_dir, sampto=total_samples)

# Extract the first channel (MLII)
raw_ecg = signals[:, 0]

# Apply the custom filter function to the raw signal
filtered_ecg = filter_ecg_signal(raw_ecg, fs)

# Create a time vector for the x-axis
time_vector = np.arange(total_samples) / fs

# Create the plot with two subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Top subplot: Raw ECG signal
ax1.plot(time_vector, raw_ecg, color='#1f77b4', linewidth=1.2, label='Raw ECG (MLII)')
ax1.set_title(f'Record {record_name}: Raw ECG Signal vs Filtered Signal', fontsize=14, fontweight='bold')
ax1.set_ylabel('Amplitude (mV)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper right')

# Bottom subplot: Filtered ECG signal
ax2.plot(time_vector, filtered_ecg, color='#2ca02c', linewidth=1.2, label='Filtered ECG (0.5 - 40 Hz)')
ax2.set_xlabel('Time (seconds)', fontsize=12)
ax2.set_ylabel('Amplitude (mV)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper right')

# Adjust layout so labels don't overlap
plt.tight_layout()

# Render the plot
plt.show()
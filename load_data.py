import wfdb
import matplotlib.pyplot as plt
import numpy as np

# Define the record name and database directory on PhysioNet
record_name = '100'
database_dir = 'mitdb'

# The MIT-BIH Arrhythmia Database has a sampling frequency of 360 Hz.
# To extract exactly 10 seconds of data: 360 samples/sec * 10 sec = 3600 samples.
sampling_frequency = 360
duration_seconds = 10
total_samples = sampling_frequency * duration_seconds

# 1. Load the ECG signal data directly from PhysioNet for the first 10 seconds.
# wfdb.rdsamp returns a 2D numpy array of signals and a dictionary of metadata (fields).
signals, fields = wfdb.rdsamp(record_name, pn_dir=database_dir, sampto=total_samples)

# 2. Load the cardiologist annotations for the same 10-second window.
# 'atr' is the standard file extension for reference annotations in this database.
annotation = wfdb.rdann(record_name, 'atr', pn_dir=database_dir, sampto=total_samples)

# Extract the first channel (MLII). 
# 'signals' is shaped (samples, channels), so index 0 is the first channel.
ecg_mlii = signals[:, 0]

# Create a time vector in seconds for the x-axis.
time_vector = np.arange(total_samples) / fields['fs']

# Extract the sample indices where annotations (heartbeats) occur.
ann_indices = annotation.sample

# Convert annotation sample indices to time in seconds for plotting.
ann_times = ann_indices / fields['fs']

# Get the corresponding amplitude of the ECG signal at each annotated index.
ann_amplitudes = ecg_mlii[ann_indices]

# 3. Plot the continuous signal and overlay the annotations.
plt.figure(figsize=(14, 6))

# Plot the raw ECG signal as a blue line.
plt.plot(time_vector, ecg_mlii, color='#1f77b4', linewidth=1.2, label='ECG Signal (MLII)')

# Overlay the annotations as red crosses ('rx').
plt.plot(ann_times, ann_amplitudes, 'rx', markersize=10, markeredgewidth=2.5, label='Cardiologist Annotations (R-peaks)')

# Format the plot for readability.
plt.title(f'MIT-BIH Arrhythmia Database - Record {record_name} (First 10 Seconds)', fontsize=14, fontweight='bold')
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Amplitude (mV)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', framealpha=1)
plt.tight_layout()

# Render the plot.
plt.show()
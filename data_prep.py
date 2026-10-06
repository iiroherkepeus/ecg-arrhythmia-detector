import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def filter_ecg_signal(signal_data, fs):
    """
    Applies a Butterworth high-pass filter of 0.5 Hz and low-pass filter of 40 Hz.
    """
    hp_cutoff = 0.5
    b_high, a_high = butter(N=4, Wn=hp_cutoff, btype='highpass', fs=fs)
    signal_highpassed = filtfilt(b_high, a_high, signal_data)
    
    lp_cutoff = 40.0
    b_low, a_low = butter(N=4, Wn=lp_cutoff, btype='lowpass', fs=fs)
    signal_filtered = filtfilt(b_low, a_low, signal_highpassed)
    
    return signal_filtered

def process_record(record_name, db_dir='mitdb', fs=360):
    """
    Loads a full record, filters the signal, and extracts 200-sample segments 
    (90 before R-peak, 110 after) for 'N' and 'V' beats.
    """
    print(f"Processing record: {record_name}")
    
    # 1. Load the full raw ECG signal
    signals, fields = wfdb.rdsamp(record_name, pn_dir=db_dir)
    raw_ecg = signals[:, 0]  # Use first channel (MLII)
    
    # 2. Filter the signal
    filtered_ecg = filter_ecg_signal(raw_ecg, fs)
    
    # 3. Load annotations
    ann = wfdb.rdann(record_name, 'atr', pn_dir=db_dir)
    samples = ann.sample
    symbols = ann.symbol
    
    X_record = []
    y_record = []
    
    # 4. Iterate through annotations and extract beats
    for i in range(len(symbols)):
        sym = symbols[i]
        
        # Keep only Normal ('N') and Premature Ventricular Contractions ('V') for the sake of scope in project
        if sym in ['N', 'V']:
            idx = samples[i]
            
            # Define window: 90 samples before, 110 after
            start_idx = idx - 90
            end_idx = idx + 110
            
            # There were problems, so safety check: Ensure indices are within signal bounds
            if start_idx >= 0 and end_idx < len(filtered_ecg):
                beat_segment = filtered_ecg[start_idx:end_idx]
                X_record.append(beat_segment)
                y_record.append(sym)
                
    return np.array(X_record), np.array(y_record)


# Main

if __name__ == "__main__":
    records_to_process = ['100', '101', '102', '103', '104', '105',
                          '106', '107', '108', '109', '111', '112',
                          '113', '114', '115', '116', '117', '118',
                          '119', '121', '122', '123', '124', '200',
                          '201', '202', '203', '205', '207', '208',
                          '209', '210', '212', '213', '214', '215',
                          '217', '219', '220', '221', '222', '223',
                          '228', '230', '231', '232', '233', '234']
    
    X_list = []
    y_list = []
    
    # Process each record and collect data
    for record in records_to_process:
        X_rec, y_rec = process_record(record)
        
        # FIX THIS: Only append if the record actually contained 'N' or 'V' beats
        # This prevents 1D empty arrays from crashing the program later
        if len(X_rec) > 0:
            X_list.append(X_rec)
            y_list.append(y_rec)
        else:
            print(f"Skipped record {record} (No 'N' or 'V' beats found)")
        
    # Combine lists into full datasets
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(f"\nData extraction complete.")
    print(f"Total shape of X: {X.shape}")
    print(f"Total shape of y: {y.shape}")
    print(f"Total 'N' beats: {np.sum(y == 'N')}")
    print(f"Total 'V' beats: {np.sum(y == 'V')}")
    
    #  Visualization of data
    """
    # Find the indices of 'N' and 'V' beats
    idx_N = np.where(y == 'N')[0]
    idx_V = np.where(y == 'V')[0]
    
    # Randomly select 10 indices of each class
    # (Setting seed for reproducibility)
    np.random.seed(42) 
    sample_N = np.random.choice(idx_N, size=10, replace=False)
    sample_V = np.random.choice(idx_V, size=10, replace=False)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    time_ms = np.arange(-90, 110) * (1000 / 360) # Convert samples to milliseconds
    
    # Plot 'N' beats in blue
    for i, idx in enumerate(sample_N):
        label = 'Normal (N)' if i == 0 else "_nolegend_"
        plt.plot(time_ms, X[idx], color='blue', alpha=0.5, linewidth=1.5, label=label)
        
    # Plot 'V' beats in red
    for i, idx in enumerate(sample_V):
        label = 'PVC (V)' if i == 0 else "_nolegend_"
        plt.plot(time_ms, X[idx], color='red', alpha=0.5, linewidth=1.5, label=label)
        
    plt.title('Comparison: Normal vs. PVC Beats (Filtered)', fontsize=14, fontweight='bold')
    plt.xlabel('Time around R-peak (ms)', fontsize=12)
    plt.ylabel('Amplitude (mV)', fontsize=12)
    plt.axvline(0, color='black', linestyle='--', alpha=0.7, label='R-peak (t=0)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.show()
    """

    # Save the extracted matrices to disk for the machine learning models
    np.save('X_data.npy', X)
    np.save('y_data.npy', y)
    print("Data saved successfully as 'X_data.npy' and 'y_data.npy'")
    

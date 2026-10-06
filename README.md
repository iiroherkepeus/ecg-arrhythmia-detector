# Clinical ECG Arrhythmia Detector

This project classifies normal ECG beats ('N') and Premature Ventricular Contractions ('V') using the MIT-BIH Arrhythmia Database. It compares a 1D-CNN (PyTorch) against a Random Forest baseline.

## Setup Instructions
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

## How to Run
Run the scripts in the following order:
1. **Prepare Data:** `python data_prep.py` (Downloads data, filters signals, segments beats, and saves matrices)
2. **Train Baseline:** `python train_rf.py` (Trains the Random Forest model and saves it)
3. **Train Deep Learning:** `python train_cnn.py` (Trains the 1D-CNN model and saves it)
4. **Launch UI:** `streamlit run app.py` (Opens the interactive dashboard in your browser)
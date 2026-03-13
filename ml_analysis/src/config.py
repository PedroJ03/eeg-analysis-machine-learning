import os

# --- Paths ---
DATA_PATH = "/home/pedroj/Desktop/pps/data/*.mat"
# For Config 2 compatibility (legacy was using hardcoded Windows paths)
DEFAULT_SUBJECT_PATTERN = 'subj*.mat'

# --- Signal Parameters ---
SFREQ = 200
TRIAL_DUR = 2.5  # Consistent with legacy scripts

# --- Frequency Bands ---
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 100.0)
}

# --- Channel Names (MNE / Config 2) ---
CH_NAMES = [
    'LSL_time', 'ECG', 'EOGh', 'EOGv', 'Fp1', 'F7', 'Fz', 'F3', 'FT9', 'T7',
    'C3', 'Cz', 'CP1', 'TP9', 'P7', 'P3', 'Pz', 'O1', 'Fp2', 'F8', 'FT10',
    'F4', 'C4', 'T8', 'CP2', 'TP10', 'P4', 'P8', 'O2', 'Oz', 'InEar'
]

CH_TYPES = [
    'misc', 'ecg', 'eog', 'eog', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
    'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
    'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg'
]

INEAR_CH = 'InEar'
EOGH_CH = 'EOGh'

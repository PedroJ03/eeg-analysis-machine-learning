import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "*.mat")
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

# --- Channel Names (MNE / Window-based) ---
CH_NAMES = ['LSL_time', 'InEar']
CH_TYPES = ['misc', 'eeg']

INEAR_CH = 'InEar'
EOGH_CH = 'EOGh'

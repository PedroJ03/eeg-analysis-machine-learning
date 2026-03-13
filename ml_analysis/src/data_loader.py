import numpy as np
import pandas as pd
from scipy.io import loadmat
import mne
from .config import SFREQ, TRIAL_DUR, CH_NAMES, CH_TYPES

def load_mat_data(path):
    """Loads a .mat file and returns data and experiment matrices."""
    mat = loadmat(path, squeeze_me=True)
    return mat['data'], mat['experiment']

def get_trial_segments(data, experiment):
    """
    Extracts trials using Trial-based logic (Config 1).
    Returns (segments, levels, groups)
    """
    ts = data[:, 0].astype(float)
    inear = data[:, 30].astype(float)
    
    exp = np.array(experiment)
    if exp.ndim == 1 and exp.dtype == object:
        exp = np.vstack([np.asarray(r).ravel() for r in experiment])
    
    exp_ts = exp[:, 0].astype(float)
    exp_codes = exp[:, 1].astype(float)
    
    segments = []
    levels = []
    i = 0
    in_nback = False
    
    while i < len(exp_codes):
        code = int(exp_codes[i])
        if code == -99:
            if in_nback: break
            else:
                in_nback = True
                i += 1
                continue
        if not in_nback:
            i += 1
            continue
        
        if i + 3 >= len(exp_codes): break
        
        nback_level = int(exp_codes[i + 1])
        if nback_level not in [0, 1, 2, 3]:
            i += 4
            continue
            
        trial_start = float(exp_ts[i])
        trial_end = float(exp_ts[i+4])
        dur = trial_end - trial_start
        
        if dur > 4.0:
            trial_end = trial_start + TRIAL_DUR
            
        mask = (ts >= trial_start) & (ts < trial_end)
        seg = inear[mask]
        
        if len(seg) > 0:
            segments.append(seg)
            levels.append(nback_level)
            
        i += 4
    return segments, np.array(levels), np.arange(len(levels)) # Each trial is a unique segment

def get_window_epochs(data, experiment):
    """
    Extracts epochs using Window-based logic (Config 2).
    Returns (epochs_data, labels, segment_ids)
    """
    # 1. Create Raw MNE
    data_scaled = data.copy()
    data_scaled[:, 4:30] *= 1e-6 # EEG channels
    data_scaled[:, 30] *= 1e-6   # InEar channel
    
    info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types=CH_TYPES)
    raw = mne.io.RawArray(data_scaled.T, info)
    
    # 2. Process Annotations from experiment
    exp_array = np.asarray(experiment)
    puntos_de_corte = np.where(exp_array[:, 1] == -99)[0]
    exp1_array = exp_array[puntos_de_corte[0] : puntos_de_corte[1]]
    
    levels_array = exp1_array[2::4]
    second_column = levels_array[:, 1]
    cambios = np.diff(second_column) != 0
    mascara_cambios = np.insert(cambios, 0, True)
    level_changes = levels_array[mascara_cambios]
    
    onsets_ts = level_changes[:, 0]
    data_lsl = data[:, 0]
    onsets_sec = [np.abs(data_lsl - ts).argmin() / SFREQ for ts in onsets_ts]
    
    ultimo_ts = levels_array[-1, 0]
    ultimo_onset_sec = np.abs(data_lsl - ultimo_ts).argmin() / SFREQ
    
    full_onsets = onsets_sec + [ultimo_onset_sec]
    durations = np.diff(full_onsets)
    descriptions = level_changes[:, 1].astype(str)
    
    # 3. Crop Raw to experiment end
    ultimo_ts_exp1 = exp1_array[-1, 0]
    end_sample_idx = int(np.abs(data_lsl - ultimo_ts_exp1).argmin())
    end_time_sec = end_sample_idx / SFREQ
    raw.crop(tmin=0.0, tmax=end_time_sec, include_tmax=True)
    
    annotations = mne.Annotations(onset=onsets_sec, duration=durations, description=descriptions)
    raw.set_annotations(annotations)
    
    # 3. Create Events and Sliding Windows (Epochs)
    events, event_id = mne.events_from_annotations(raw)
    
    X = []
    y = []
    groups = []
    window_size_samples = int(TRIAL_DUR * SFREQ)
    
    for idx, ev in enumerate(events):
        sample_idx = ev[0]
        event_code = ev[2] 
        
        level = int(float(list(event_id.keys())[list(event_id.values()).index(event_code)]))
        
        end_idx = events[idx+1][0] if idx+1 < len(events) else raw.n_times
        
        for start in range(sample_idx, end_idx - window_size_samples, window_size_samples):
            segment = raw.get_data(picks='InEar', start=start, stop=start + window_size_samples)
            X.append(segment.squeeze())
            y.append(level)
            groups.append(idx) # Group by original segment ID to prevent leakage
            
    return np.array(X), np.array(y), np.array(groups)

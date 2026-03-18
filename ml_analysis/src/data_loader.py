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
    # Select channel: legacy uses 30, but current data may only have 2 columns
    ch_idx = 30 if data.shape[1] > 30 else 1
    inear = data[:, ch_idx].astype(float)
    
    exp = np.array(experiment)
    if exp.ndim == 1 and exp.dtype == object:
        exp = np.vstack([np.asarray(r).ravel() for r in experiment])
    
    exp_ts = exp[:, 0].astype(float)
    exp_codes = exp[:, 1].astype(float)
    
    segments = []
    levels = []
    groups = []
    i = 0
    in_nback = False
    
    level_counts = {}
    current_group_per_level = {}
    
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
        
        trial_id = int(exp_codes[i])
        nback_level = int(exp_codes[i + 1])
        if nback_level not in [0, 1, 2, 3]:
            i += 4
            continue
            
        if trial_id == 0:
            if nback_level not in level_counts:
                level_counts[nback_level] = 0
            current_group_per_level[nback_level] = level_counts[nback_level] % 2
            level_counts[nback_level] += 1
            
        trial_group = current_group_per_level.get(nback_level, 0)
            
        trial_start = float(exp_ts[i])
        trial_end = float(exp_ts[i+4])
        dur = trial_end - trial_start
        
        if dur > 4.0:
            # Legacy skips trials with duration > 4.0s instead of clipping them
            i += 4
            continue
            
        mask = (ts >= trial_start) & (ts < trial_end)
        seg = inear[mask]
        
        if len(seg) > 0:
            segments.append(seg)
            levels.append(nback_level)
            groups.append(trial_group)
            
        i += 4
    return segments, np.array(levels), np.array(groups)

def get_window_epochs(data, experiment):
    """
    Extracts epochs using Window-based logic (Config 2).
    Returns (epochs_data, labels, segment_ids)
    """
    # Determine channel index: legacy used 30
    ch_idx = 30 if data.shape[1] > 30 else 1
    
    # 1. Create Raw MNE
    # We only take the timestamp and the selected InEar channel to match CH_NAMES
    data_subset = data[:, [0, ch_idx]].copy()
    data_subset[:, 1] *= 1e-6   # Scale InEar channel
    
    info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types=CH_TYPES)
    raw = mne.io.RawArray(data_subset.T, info)
    
    # 2. Process Annotations from experiment
    exp_array = np.asarray(experiment)
    puntos_de_corte = np.where(exp_array[:, 1] == -99)[0]
    exp1_array = exp_array[puntos_de_corte[0] : puntos_de_corte[1]]
    
    #levels_array = exp1_array[2::4]
    #second_column = levels_array[:, 1]
    #cambios = np.diff(second_column) != 0
    #mascara_cambios = np.insert(cambios, 0, True)
    #level_changes = levels_array[mascara_cambios]
    
    trial_ids_array = exp1_array[1::4] # Aquí están los IDs (0 al 17)
    levels_array = exp1_array[2::4]    # Aquí están los niveles (0 al 3)
    mascara_cambios = (trial_ids_array[:, 1] == 0)
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
    
    # Apply 1-99Hz band-pass filter as in legacy scripts
    raw.filter(l_freq=1.0, h_freq=99.0, fir_design='firwin')
    
    annotations = mne.Annotations(onset=onsets_sec, duration=durations, description=descriptions)
    raw.set_annotations(annotations)
    
    # 3. Create Events and Sliding Windows (Epochs)
    events, event_id = mne.events_from_annotations(raw)
    
    X = []
    y = []
    groups = []
    window_size_samples = int(TRIAL_DUR * SFREQ)
    
    # Counter for each level to alternate groups as in legacy
    level_counts = {}
    
    # Invert event_id to map code back to level string
    code_to_level = {v: k for k, v in event_id.items()}
    
    for idx, ev in enumerate(events):
        sample_idx = ev[0]
        event_code = ev[2] 
        
        level_str = code_to_level[event_code]
        level = int(float(level_str))
        
        # Legacy alternating group logic
        if level not in level_counts:
            level_counts[level] = 0
        grupo_actual = level_counts[level] % 2
        
        # Determine block end (legacy used next event)
        if idx + 1 < len(events):
            end_idx = events[idx+1][0]
        else:
            end_idx = raw.n_times
            
        for start in range(sample_idx, end_idx - window_size_samples, window_size_samples):
            segment = raw.get_data(picks='InEar', start=start, stop=start + window_size_samples)
            X.append(segment.squeeze().copy())
            y.append(level)
            groups.append(grupo_actual)
            
        level_counts[level] += 1
            
    return np.array(X), np.array(y), np.array(groups)

import numpy as np
from scipy import signal, stats
import pywt
from .config import SFREQ, BANDS

def bandpower(x, sf, fmin, fmax):
    f, Pxx = signal.welch(x, fs=sf, nperseg=min(256, len(x)))
    mask = (f >= fmin) & (f <= fmax)
    return np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 0.0

def wavelet_energies(x, wavelet='db4', level=4, max_coeffs=5):
    coeffs = pywt.wavedec(x, wavelet, level=level)
    energies = [np.sum((c.astype(float))**2) for c in coeffs]
    if len(energies) < max_coeffs:
        energies += [0.0] * (max_coeffs - len(energies))
    return energies[:max_coeffs]

def extract_features_vector(x, groups='BEW', sf=SFREQ):
    """
    Extracts a feature vector for a given signal segment based on requested groups.
    groups: string containing 'B'/'Bg' (Bands), 'E' (Stats), 'W' (Wavelets).
    'B' is bands without gamma. 'Bg' is bands with gamma.
    """
    x = np.asarray(x).astype(float)
    if len(x) == 0:
        return np.array([])
    
    feats = []
    
    # Group E: Time-domain / Statistical
    if 'E' in groups:
        feats.append(np.mean(x))
        feats.append(np.std(x))
        feats.append(np.max(x))
        feats.append(np.min(x))
        feats.append(np.median(x))
        feats.append(stats.skew(x))
        feats.append(stats.kurtosis(x))
        feats.append(np.sum(np.abs(np.diff(x)))) # line length
    
    # Group B / Bg: Band Powers
    if 'B' in groups or 'Bg' in groups:
        for band, (fmin, fmax) in BANDS.items():
            if band == 'gamma' and 'Bg' not in groups:
                continue
            feats.append(bandpower(x, sf, fmin, fmax))
        
    # Group W: Wavelet Energies
    if 'W' in groups:
        feats.extend(wavelet_energies(x))
    
    return np.array(feats)

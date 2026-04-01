import numpy as np
from scipy import signal, stats
import pywt
from .config import SFREQ, BANDS

def hjorth_parameters(x):
    """Compute Hjorth Activity, Mobility and Complexity."""
    dx = np.diff(x)
    ddx = np.diff(dx)
    
    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)
    
    eps = 1e-9
    
    activity = var_x
    mobility = np.sqrt(var_dx / (var_x + eps))
    complexity = np.sqrt(var_ddx / (var_dx + eps)) / (mobility + eps)
    
    return activity, mobility, complexity

def spectral_entropy(x, sf):
    f, Pxx = signal.welch(x, fs=sf, nperseg=min(256, len(x)))
    P_norm = Pxx / (np.sum(Pxx) + 1e-9)
    return stats.entropy(P_norm)

def shannon_entropy(x, bins=10):
    hist, bin_edges = np.histogram(x, bins=bins, density=True)
    p = hist * np.diff(bin_edges)
    return stats.entropy(p + 1e-9)

def zero_crossing_rate(x):
    return np.sum(np.diff(np.sign(x)) != 0) / len(x)

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
    groups: string containing 'B'/'Bg' (Bands), 'E' (Stats), 'W' (Wavelets),
    'R' (Spectral Ratios), 'H' (Hjorth Parameters), 'N' (Entropy), 'Z' (Zero-Crossing).
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
    
    # Evaluate band powers if B, Bg or R are in groups
    bands_power = {}
    if 'B' in groups or 'Bg' in groups or 'R' in groups:
        for band, (fmin, fmax) in BANDS.items():
            bands_power[band] = bandpower(x, sf, fmin, fmax)
            
        if 'B' in groups or 'Bg' in groups:
            for band in BANDS.keys():
                if band == 'gamma' and 'Bg' not in groups:
                    continue
                feats.append(bands_power[band])
    
    # Group R: Spectral Ratios
    if 'R' in groups:
        t = bands_power['theta']
        a = bands_power['alpha']
        b = bands_power['beta']
        g = bands_power['gamma']
        
        eps = 1e-9 
        
        # Theta/Alpha Ratio (Esfuerzo mental)
        tar = t / (a + eps)
        feats.append(tar)
        
        # Beta/Theta Ratio (Alerta / Concentración)
        btr = b / (t + eps)
        feats.append(btr)
        
        # Gamma/Alpha Ratio (Alto procesamiento vs Relajación)
        gar = g / (a + eps)
        feats.append(gar)
        
    # Group H: Hjorth Parameters
    if 'H' in groups:
        activity, mobility, complexity = hjorth_parameters(x)
        feats.append(activity)
        feats.append(mobility)
        feats.append(complexity)
        
    # Group N: Entropy (Nonlinear)
    if 'N' in groups:
        feats.append(spectral_entropy(x, sf))
        feats.append(shannon_entropy(x))
        
    # Group Z: Zero-Crossing Rate
    if 'Z' in groups:
        feats.append(zero_crossing_rate(x))
    
    # Group W: Wavelet Energies
    if 'W' in groups:
        feats.extend(wavelet_energies(x))
    
    return np.array(feats)

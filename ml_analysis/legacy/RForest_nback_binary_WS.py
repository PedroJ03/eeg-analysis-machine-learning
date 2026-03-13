import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import glob
import pywt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Parámetros ---
SFREQ = 200
TRIAL_DUR = 2.5
DATA_PATH = "/home/pedroj/Desktop/pps/senales_eeg/data/*.mat"  # carpeta con los 16 archivos


# -------------------- FUNCIONES AUXILIARES -------------------- #

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


def extract_features(x, sf=SFREQ):
    feats = {}
    x = np.asarray(x).astype(float)
    if len(x) == 0:
        return {k: 0.0 for k in [
            'mean','std','max','min','median','skew','kurtosis','line_length',
            'delta','theta','alpha','beta','gamma',
            'wave_e0','wave_e1','wave_e2','wave_e3','wave_e4'
        ]}
    
    # Time-domain
    feats['mean'] = np.mean(x)
    feats['std'] = np.std(x)
    feats['max'] = np.max(x)
    feats['min'] = np.min(x)
    feats['median'] = np.median(x)
    feats['skew'] = stats.skew(x)
    feats['kurtosis'] = stats.kurtosis(x)
    feats['line_length'] = np.sum(np.abs(np.diff(x)))
    
    # Band powers
    feats['delta'] = bandpower(x, sf, 0.5, 4)
    feats['theta'] = bandpower(x, sf, 4, 8)
    feats['alpha'] = bandpower(x, sf, 8, 12)
    feats['beta'] = bandpower(x, sf, 12, 30)
    #feats['gamma'] = bandpower(x, sf, 30, 100)
    
    # Wavelet energies
    wes = wavelet_energies(x, wavelet='db4', level=4, max_coeffs=5)
    for i, e in enumerate(wes):
        feats[f'wave_e{i}'] = e

    return feats


def process_subject(path, debug=False):
    mat = loadmat(path, squeeze_me=True)
    data = mat['data']
    experiment = mat['experiment']

    ts = data[:, 0].astype(float)
    inear = data[:, 30].astype(float)

    exp = np.array(experiment)
    if exp.ndim == 1 and exp.dtype == object:
        exp = np.vstack([np.asarray(r).ravel() for r in experiment])
    if exp.ndim == 2 and exp.shape[1] >= 2:
        exp_ts = exp[:, 0].astype(float)
        exp_codes = exp[:, 1].astype(float)
    else:
        raise ValueError(f"Formato inesperado en 'experiment' de {path}: shape {exp.shape}")

    X, y = [], []
    i = 0
    in_nback = False

    while i < len(exp_codes):
        code = int(exp_codes[i])

        if code == -99:
            if in_nback:
                break
            else:
                in_nback = True
                i += 1
                continue

        if not in_nback:
            i += 1
            continue

        if i + 3 >= len(exp_codes):
            break

        nback_level = int(exp_codes[i + 1])
        if nback_level not in [0, 1, 2, 3]:
            i += 4
            continue

        trial_start = float(exp_ts[i])
        trial_end = float(exp_ts[i + 4])
        dur = trial_end - trial_start
        if dur > 4.0:
            trial_end = trial_start + TRIAL_DUR
            i += 4
            continue

        mask = (ts >= trial_start) & (ts < trial_end)
        segment = inear[mask]

        if len(segment) > 0:
            feats = extract_features(segment)
            X.append(feats)
            # --- Re-map a binario ---
            y_bin = 0 if nback_level <= 1 else 1  # 0–1 → 0 ; 2–3 → 1
            y.append(y_bin)

        i += 4

    if debug:
        print(f"{path.split('/')[-1]} → {len(y)} trials extraídos")

    return pd.DataFrame(X), np.array(y)


# -------------------- ENTRENAMIENTO WITHIN-SUBJECT (RANDOM FOREST BINARIO 0–1 vs 2–3) -------------------- #

subject_paths = sorted(glob.glob(DATA_PATH))
all_accuracies = []
all_y_true = []
all_y_pred = []
total_epochs = 0

if not subject_paths:
    print("❌ No se encontraron archivos en:", DATA_PATH)
else:
    print(f"Procesando {len(subject_paths)} sujetos...\n")
  

for path in subject_paths:
    X_df, y = process_subject(path)
    subj_name = path.split("/")[-1].replace(".mat", "")

    print(f"{subj_name}: clases originales -> {np.unique(y)} (n={len(y)})")

    if X_df.empty or len(np.unique(y)) < 2:
        print(f"⚠️ {subj_name}: sin suficientes datos, se omite.")
        continue

    X = X_df.values
    total_epochs += len(y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_acc = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        fold_acc.append(acc)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    subj_acc = np.mean(fold_acc)
    all_accuracies.append(subj_acc)
    print(f"✅ {subj_name}: precisión media (5-fold, binario 0–1 vs 2–3) = {subj_acc:.3f}")


# -------------------- RESULTADOS FINALES -------------------- #

if all_accuracies:
    print("\n📈 RESULTADOS GLOBALES (Within-subject, binario 0–1 vs 2–3)")
    print(f"Total de épocas procesadas: {total_epochs}")
    print(f"Precisión promedio: {np.mean(all_accuracies):.3f}")
    print(f"Precisión mediana:  {np.median(all_accuracies):.3f}")
    print(f"Desvío estándar:    {np.std(all_accuracies):.3f}")

    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    print("\n🧠 Matriz de confusión global (bajo vs alto esfuerzo):")
    print(pd.DataFrame(cm, index=["Real bajo", "Real alto"], columns=["Pred bajo", "Pred alto"]))
else:
    print("No se obtuvieron resultados válidos.")

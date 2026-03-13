import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score
from sklearn.linear_model import LinearRegression
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pywt

# --- Parámetros ---
SFREQ = 200
TRIAL_DUR = 2.5
DATA_PATH = "/home/pedroj/Desktop/pps/senales_eeg/data/*.mat"  # carpeta con los 16 archivos

def bandpower(x, sf, fmin, fmax):
    f, Pxx = signal.welch(x, fs=sf, nperseg=min(256, len(x)))
    mask = (f >= fmin) & (f <= fmax)
    return np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 0.0

def wavelet_energies(x, wavelet='db4', level=4, max_coeffs=5):
    coeffs = pywt.wavedec(x, wavelet, level=level)
    energies = [np.sum((c.astype(float))**2) for c in coeffs]
    # pad/truncate to max_coeffs
    if len(energies) < max_coeffs:
        energies += [0.0] * (max_coeffs - len(energies))
    return energies[:max_coeffs]

def extract_features(x, sf=SFREQ):
    feats = {}
    x = np.asarray(x).astype(float)
    n = len(x)
    if n == 0:
        # devolver zeros si segmento vacío (mejor evitar antes)
        return {k:0.0 for k in ['mean','std', 'max','min','median',
                                'skew','kurtosis','line_length','delta','theta','alpha','beta','gamma',
                                'wave_e0','wave_e1','wave_e2','wave_e3','wave_e4']}
    
    # Time-domain
    feats['mean'] = np.mean(x)
    feats['std'] = np.std(x)
    feats['max'] = np.max(x)
    feats['min'] = np.min(x)
    feats['median'] = np.median(x)
    feats['skew'] = stats.skew(x)
    feats['kurtosis'] = stats.kurtosis(x)
    feats['line_length'] = np.sum(np.abs(np.diff(x))) 

    
    # Band powers (absolute)
    total_power = bandpower(x, sf, 0.5, 100)
    feats['delta'] = bandpower(x, sf, 0.5, 4)
    feats['theta'] = bandpower(x, sf, 4, 8)
    feats['alpha'] = bandpower(x, sf, 8, 12)
    feats['beta'] = bandpower(x, sf, 12, 30)
    feats['gamma'] = bandpower(x, sf, 30, 100)
    

    # Wavelet energies (first 5 coeff groups)
    wes = wavelet_energies(x, wavelet='db4', level=4, max_coeffs=5)
    for i, e in enumerate(wes):
        feats[f'wave_e{i}'] = e
    
    return feats


def process_subject(path, debug=False):
    mat = loadmat(path, squeeze_me=True)
    data = mat['data']
    experiment = mat['experiment']

    # Señales
    ts = data[:, 0].astype(float)         # timestamps
    inear = data[:, 30].astype(float)     # canal in-ear

    # --- Manejo robusto de experiment (por si squeeze cambia la forma) ---
    exp = np.array(experiment)
    if exp.ndim == 1 and exp.dtype == object:
        # cada fila puede venir como tupla -> convertir a array 2D
        exp = np.vstack([np.asarray(r).ravel() for r in experiment])
    if exp.ndim == 2 and exp.shape[1] >= 2:
        exp_ts = exp[:, 0].astype(float)
        exp_codes = exp[:, 1].astype(float)
    else:
        raise ValueError(f"Formato inesperado en 'experiment' de {path}: shape {exp.shape}")

    if debug:
        print(f"\n--- Procesando {path} ---")
        print("Primeros 30 exp_codes:", exp_codes[:30])
        print("Primeros 30 exp_ts  :", exp_ts[:30])

    X, y = [], []
    i = 0
    in_nback = False
    trial_count = 0

    while i < len(exp_codes):
        code = int(exp_codes[i])

        # marcador -99: inicio primer experimento / posible inicio del segundo
        if code == -99:
            if in_nback:
                if debug:
                    print("Encontrado segundo -99: asumo que empieza experimento 2 -> termino N-back.")
                break   # dejamos de procesar N-back para este sujeto
            else:
                in_nback = True
                i += 1
                continue

        # si todavía no estamos en la sección N-back, avanzamos
        if not in_nback:
            print("Avanzando hasta encontrar -99...")
            i += 1
            continue

        # ahora estamos en N-back: cada trial ocupa 4 elementos (id, nback, stim, resp)
        if i + 3 >= len(exp_codes):
            if debug:
                print("Entrada de trial incompleta en index", i, "- rompo.")
            break

        trial_id = int(exp_codes[i])          # ID (0-17)
        nback_level = int(exp_codes[i + 1])  # nivel 0..3
        # stimulus = exp_codes[i+2]
        # response = exp_codes[i+3]

        # validar nivel
        if nback_level not in [0, 1, 2, 3]:
            if debug:
                print(f"Saltando: nivel inválido {nback_level} en idx {i}")
            i += 4
            continue

        trial_start = float(exp_ts[i])
        trial_end = float(exp_ts[i+4])  
        #print("Tiempo de trial: ", trial_end - trial_start)
        dur = trial_end - trial_start
        if ( dur > 4.0):
            trial_end = trial_start + TRIAL_DUR
            i += 4
            continue 

        mask = (ts >= trial_start) & (ts < trial_end)
        segment = inear[mask]

        if len(segment) > 0:
            feats = extract_features(segment)
            feats['subject'] = path
            X.append(feats)
            y.append(nback_level)
            trial_count += 1
            if debug:
                print(f"Trial {trial_count}: id={trial_id}, nback={nback_level}, samples={len(segment)}, start={trial_start:.3f}")

        i += 4  # avanzamos exactamente al próximo trial

    if debug:
        print(f"Total trials N-back extraídos: {trial_count}")

    return pd.DataFrame(X), np.array(y)

'''
def remove_eog_from_inear(path, debug=False, plot=False):
    """
    Carga el archivo .mat y remueve la contribución ocular (HEOG, canal 3)
    del canal in-ear mediante regresión lineal.
    """
    mat = loadmat(path, squeeze_me=True)
    data = mat["data"].astype(float)

    ts = data[:, 0]
    heog = data[:, 2]      # canal 3 según README (index 2)
    inear = data[:, 30]    # canal 31 según README (index 30)

    # Regr. lineal simple: inear ~ heog
    X = heog.reshape(-1, 1)
    y = inear
    reg = LinearRegression().fit(X, y)
    inear_pred = reg.predict(X)
    inear_clean = y - inear_pred

    if debug:
        print(f"Coeficiente β_HEOG = {reg.coef_[0]:.4f}, Intercepto = {reg.intercept_:.4f}")
        print(f"Varianza original: {np.var(inear):.4f}, varianza corregida: {np.var(inear_clean):.4f}")

    if plot:
        # comparar señales (segmento)
        plt.figure(figsize=(10,4))
        plt.plot(ts[:2000], inear[:2000], label="In-ear original")
        plt.plot(ts[:2000], inear_clean[:2000], label="In-ear limpio", alpha=0.7)
        plt.legend()
        plt.title("In-ear: antes y después de remover HEOG")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud")
        plt.show()

        # comparar espectros
        f1, P1 = signal.welch(inear, fs=SFREQ, nperseg=1024)
        f2, P2 = signal.welch(inear_clean, fs=SFREQ, nperseg=1024)
        plt.figure(figsize=(8,4))
        plt.semilogy(f1, P1, label='Original')
        plt.semilogy(f2, P2, label='Limpio')
        plt.xlim(0, 40)
        plt.title("PSD antes y después de la corrección ocular")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Densidad espectral (PSD)")
        plt.legend()
        plt.show()

    return ts, inear_clean
'''

# --- Recorrer todos los sujetos ---
all_X, all_y = [], []
for path in glob.glob(DATA_PATH):
    X_df, y_arr = process_subject(path)
    all_X.append(X_df)
    all_y.append(y_arr)

X_df = pd.concat(all_X, ignore_index=True)
y = np.concatenate(all_y)

print("Dataset total:", X_df.shape)
print("Distribución de clases:", np.bincount(y))



# --- Re-map etiquetas a binario ---
# 0-1 → clase 0 (bajo workload), 2-3 → clase 1 (alto workload)
y_binary = np.where(y <= 1, 0, 1)

print("Distribución binaria:", np.bincount(y_binary))


X_train, X_test, y_train, y_test = train_test_split(
    X_df.drop(columns=['subject']), y_binary, test_size=0.2, stratify=y_binary, random_state=42
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))


from sklearn.metrics import classification_report, confusion_matrix

# --- Reporte de clasificación binaria ---
report = classification_report(
    y_test, y_pred,
    target_names=["Bajo workload","Alto workload"],
    output_dict=True
)
report_df = pd.DataFrame(report).transpose()
print("\n📊 Reporte de clasificación (Random Forest - Binario)")
print(report_df.round(2))

# --- Matriz de confusión como DataFrame ---
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
                     index=["Real Bajo","Real Alto"],
                     columns=["Pred Bajo","Pred Alto"])
print("\n📉 Matriz de confusión (Binaria)")
print(cm_df)

'''
# --- Distribución de clases ---
plt.figure(figsize=(6,4))
sns.countplot(x=y_binary, palette="viridis")
plt.title("Distribución de ensayos (Bajo vs Alto workload)")
plt.xlabel("Workload (0=Bajo, 1=Alto)")
plt.ylabel("Cantidad de ensayos")
plt.show()

# --- Importancia de características ---
importances = clf.feature_importances_
feat_names = X_train.columns

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=feat_names, palette="mako")
plt.title("Importancia de características en Random Forest (Binario)")
plt.xlabel("Importancia")
plt.ylabel("Feature")
plt.show()

# --- Matriz de confusión como heatmap ---
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred Bajo","Pred Alto"],
            yticklabels=["Real Bajo","Real Alto"])
plt.title("Matriz de confusión - Clasificación Binaria")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.show()

# --- Boxplot de potencia theta (ejemplo) ---
plt.figure(figsize=(6,4))
sns.boxplot(x=y_binary, y=X_df['theta'], palette="Set2")
plt.title("Distribución de potencia theta por workload")
plt.xlabel("Workload (0=Bajo, 1=Alto)")
plt.ylabel("Potencia theta")
plt.show()

# --- Recall por clase ---
recalls = [report[c]["recall"] for c in ["Bajo workload","Alto workload"]]

plt.figure(figsize=(6,4))
sns.barplot(x=["Bajo workload","Alto workload"], y=recalls, palette="rocket")
plt.title("Recall por clase (Binario)")
plt.ylim(0,1)
plt.ylabel("Recall")
plt.show()
'''
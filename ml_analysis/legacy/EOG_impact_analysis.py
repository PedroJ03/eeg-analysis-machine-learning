import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import glob
import matplotlib.pyplot as plt
import seaborn as sns


# --- Parámetros ---
SFREQ = 200
TRIAL_DUR = 2.5
DATA_PATH = "/home/pedroj/Desktop/pps/senales_eeg/data/*.mat"  # carpeta con los 16 archivos

#analsis de la importancia de EOG en la señal in-ear
def analyze_eog_effect(heog, inear, subject_name="subj", sf=SFREQ, nplot=2000, do_plot=False):
    heog = np.asarray(heog).ravel().astype(float)
    inear = np.asarray(inear).ravel().astype(float)
    n = min(len(heog), len(inear))
    heog = heog[:n]; inear = inear[:n]

    # Ajuste lineal HEOG → Inear
    reg = LinearRegression().fit(heog.reshape(-1,1), inear)
    pred = reg.predict(heog.reshape(-1,1))
    inear_clean = inear - pred

    # Métricas
    corr = np.corrcoef(heog, inear)[0,1]
    r2 = r2_score(inear, pred)
    var_orig = np.var(inear)
    var_clean = np.var(inear_clean)
    pct_red = 100.0 * (var_orig - var_clean) / var_orig if var_orig > 0 else 0.0

    result = dict(
        subject=subject_name,
        beta=float(reg.coef_[0]),
        intercept=float(reg.intercept_),
        corr=float(corr),
        r2=float(r2),
        var_orig=float(var_orig),
        var_clean=float(var_clean),
        pct_var_reduction=float(pct_red)
    )

    if do_plot:
        t = np.arange(n) / sf
        plt.figure(figsize=(12,4))
        plt.plot(t[:nplot], inear[:nplot], label="In-ear original", alpha=0.8)
        plt.plot(t[:nplot], inear_clean[:nplot], label="In-ear corregida", alpha=0.8)
        plt.plot(t[:nplot], pred[:nplot], label="Pred. HEOG", alpha=0.5)
        plt.legend(); plt.xlabel("Tiempo (s)")
        plt.title(f"{subject_name}: señal original vs corregida")
        plt.show()

        # PSD antes/después
        f1, P1 = signal.welch(inear, fs=sf, nperseg=2048)
        f2, P2 = signal.welch(inear_clean, fs=sf, nperseg=2048)
        plt.figure(figsize=(8,4))
        plt.semilogy(f1, P1, label="Original")
        plt.semilogy(f2, P2, label="Corregida")
        plt.xlim(0, 40)
        plt.xlabel("Hz"); plt.ylabel("PSD")
        plt.legend()
        plt.title(f"{subject_name}: PSD before/after HEOG regression")
        plt.show()

        # Scatter
        plt.figure(figsize=(6,5))
        plt.scatter(heog[::50], inear[::50], s=6, alpha=0.4)
        xs = np.linspace(np.min(heog), np.max(heog), 100)
        plt.plot(xs, reg.predict(xs.reshape(-1,1)), 'r', lw=2)
        plt.xlabel("HEOG"); plt.ylabel("In-ear")
        plt.title(f"{subject_name}: relación HEOG–In-ear (R²={r2:.3f})")
        plt.show()

    return result


# --- Recorrer todos los sujetos ---
results = []
for path in glob.glob(DATA_PATH):
    mat = loadmat(path, squeeze_me=True)
    data = mat["data"]

    heog = data[:, 2]   # canal 3 en descripción → índice 2
    inear = data[:, 30] # canal 31 → índice 30

    subj_name = path.split("/")[-1].replace(".mat", "")
    print(f"Analizando {subj_name}...")
    res = analyze_eog_effect(heog, inear, subject_name=subj_name)
    results.append(res)

# Convertir resultados a DataFrame
df_results = pd.DataFrame(results)
print("\n📊 Resultados de regresión HEOG → In-ear")
print(df_results.round(5))

# Guardar tabla
df_results.to_csv("heog_regression_results.csv", index=False)

# --- Mostrar resumen general ---
print("\nResumen estadístico:")
print(df_results.describe().round(4))

# --- Plots globales ---
plt.figure(figsize=(8,5))
sns.barplot(x="subject", y="pct_var_reduction", data=df_results, palette="viridis")
plt.xticks(rotation=45, ha="right")
plt.title("Reducción porcentual de varianza (HEOG → In-ear)")
plt.ylabel("% reducción de varianza")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x="r2", y="pct_var_reduction", data=df_results, hue="subject", legend=False)
plt.title("Relación entre R² y reducción de varianza")
plt.xlabel("R²"); plt.ylabel("% reducción varianza")
plt.show()

# --- Graficar los 3 sujetos con mayor impacto ---
top3 = df_results.nlargest(3, "pct_var_reduction")
print("\n📈 Sujetos con mayor efecto de regresión:")
print(top3)

for path in glob.glob(DATA_PATH):
    name = path.split("/")[-1].replace(".mat", "")
    if name in top3["subject"].values:
        print(f"\nGraficando detalles de {name}...")
        mat = loadmat(path, squeeze_me=True)
        data = mat["data"]
        heog = data[:, 2]
        inear = data[:, 30]
        analyze_eog_effect(heog, inear, subject_name=name, do_plot=True)


plt.figure(figsize=(7,4))
sns.barplot(x='subject', y='r2', data=df_results, color='steelblue')
plt.xticks(rotation=45)
plt.title("R² HEOG → In-Ear por sujeto")
plt.ylabel("R² (proporción de varianza explicada)")
plt.xlabel("Sujeto")
plt.show()

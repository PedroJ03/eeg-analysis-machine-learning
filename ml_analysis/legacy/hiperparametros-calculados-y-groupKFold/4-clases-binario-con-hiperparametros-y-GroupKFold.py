# EN ESTE CODIGO SE ENTRENA UN MODELO POR CADA SUJETO, LUEGO PARA EL RESULTADO GENERAL SE LE CALCULA LA MEDIA Y LA MEDIANA A LA LISTA DE RESULTADOS

import scipy.io as sio
import os
import glob

import mne
from mne.preprocessing import EOGRegression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt

from scipy.signal import welch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
# Configuración de parámetros
DATASET_PATH = 'C:\\Users\\Rober\\PPS\\Nuevo_dataset_idun\\'
SUBJECT_PATTERN = 'subj*.mat'  # Patrón para encontrar archivos de sujetos

# Definir las bandas de frecuencia para el análisis
bands = {
    "delta": (0.5, 3.99),
    "theta": (4, 7.99),
    "alpha": (8, 12.99),
    "beta": (13, 29.99),
    "gamma": (30, 100)
}


def get_band_powers(epoch_data, fs, bands):
    """Extrae la potencia de banda de una época de EEG usando el método de Welch."""
    freqs, psd = welch(epoch_data, fs, nperseg=fs)
    
    features = []
    for band, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        # Asegurar salida escalar por banda (sumar por canales si hay más de uno)
        if psd.ndim > 1:
            band_val = np.trapezoid(psd[:, mask], freqs[mask], axis=1)
            band_power = float(np.maximum(0.0, np.sum(band_val)))
        else:
            band_val = np.trapezoid(psd[mask], freqs[mask])
            band_power = float(np.maximum(0.0, band_val))
        features.append(band_power)
    return features # psd RETORNANDO LA LISTA DE PSD QUEDAN FEATURES DE CADA FRECUENCIA, NO DE CADA BANDA

def wavelet_energies(x, wavelet='db4', level=4, max_coeffs=5):
    coeffs = pywt.wavedec(x, wavelet, level=level)
    energies = [np.sum((c.astype(float))**2) for c in coeffs]
    # pad/truncate to max_coeffs
    if len(energies) < max_coeffs:
        energies += [0.0] * (max_coeffs - len(energies))
    return energies[:max_coeffs]

def trainModel(X_feature, y, groups): # Recibe los grupos
    
    # Instanciamos el GroupKFold para 2 pliegues (mitad y mitad temporal)
    gkf = GroupKFold(n_splits=2)
    
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'],
        'kernel': ['rbf']
    }
    
    accuracies = []
    fold_idx = 1
    
    print("Iniciando validación cruzada Leave-One-Block-Out (2 Folds)...")
    
    # El bucle va a correr 2 veces
    for train_idx, test_idx in gkf.split(X_feature, y, groups):
        print(f"\n--- Fold {fold_idx} ---")
        
        X_train, X_test = X_feature[train_idx], X_feature[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Estandarizar los datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        svc = SVC()
        grid_search = GridSearchCV(
            estimator=svc, 
            param_grid=param_grid, 
            cv=3, # Bajamos a 3 interno para no sobrecargar de tiempo la PC
            n_jobs=-1, 
            verbose=0, 
            scoring='accuracy'
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        best_params = grid_search.best_params_
        best_clf = grid_search.best_estimator_
        
        print(f"Mejores hiperparámetros: {best_params}")
        
        y_pred = best_clf.predict(X_test_scaled)

        # Matriz de confusión para este fold
        labels = np.unique(np.concatenate([y_test, y_pred]))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        index = [f"Real {lbl}" for lbl in labels]
        columns = [f"Pred {lbl}" for lbl in labels]
        cm_df = pd.DataFrame(cm, index=index, columns=columns)
        
        print(cm_df)

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy Fold {fold_idx}: {acc:.4f}")
        
        accuracies.append(acc)
        fold_idx += 1

    # Retorna el promedio de accuracy de los 2 pliegues
    mean_acc = np.mean(accuracies)
    print(f"\n=> Accuracy Promedio del Sujeto: {mean_acc:.4f}")
    return mean_acc


def process_single_subject(mat_file_path):
    """Procesa un solo archivo de sujeto y extrae características."""
    print(f"\nProcesando archivo: {mat_file_path}")
    
    try:
        # Cargar el archivo .mat
        mat_data = sio.loadmat(mat_file_path)
        
        # Extraer las matrices principales
        data_matrix = mat_data['data']
        experiment_matrix = mat_data['experiment']

        # --- 2. Definir los metadatos del experimento ---
        sfreq = 200  # Frecuencia de muestreo en Hz

        # Nombres de los 31 canales
        ch_names = [
            'LSL_time', 'ECG', 'EOGh', 'EOGv', 'Fp1', 'F7', 'Fz', 'F3', 'FT9', 'T7',
            'C3', 'Cz', 'CP1', 'TP9', 'P7', 'P3', 'Pz', 'O1', 'Fp2', 'F8', 'FT10',
            'F4', 'C4', 'T8', 'CP2', 'TP10', 'P4', 'P8', 'O2', 'Oz', 'InEar'
        ]

        # Tipos correspondientes a cada canal
        ch_types = [
            'misc', 'ecg', 'eog', 'eog', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
            'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
            'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg'
        ]

        # --- 3. Preparar los datos para MNE ---
        # MNE necesita los datos en formato (n_canales, n_muestras)
        # y en unidades estándar (Volts). Asumimos que los datos originales
        # están en microvolts y los convertimos.
        data_scaled = data_matrix.copy()
        scaling_factor = 1e-6 # Factor para convertir de microvolts a Volts

        # Escalar solo los canales que lo necesitan (EEG, EOG, ECG)
        for i, ch_type in enumerate(ch_types):
            if ch_type in ['eeg']:
                data_scaled[:, i] *= scaling_factor

        # Transponer los datos al formato requerido por MNE
        data_for_mne = data_scaled.T

        # --- 4. Crear los objetos 'Info' y 'Raw' de MNE ---
        # El objeto 'Info' contiene todos los metadatos
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        # El objeto 'Raw' contiene los datos y los metadatos
        raw = mne.io.RawArray(data_for_mne, info)

        # Añadir la información de la posición de los electrodos (montura)
        # Excluir canales que no están en la montura estándar
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')

        # --- 5. Filtrar solo la señal InEar ---
        # Encontrar el índice del canal InEar
        inear_index = ch_names.index('InEar')

        # Extraer solo los datos del canal InEar (sin escalar, manteniendo valores originales)
        inear_signal = data_matrix[:, inear_index]
        eog_signal = data_matrix[:, ch_names.index('EOGh')]
        
        
        # --- SE CREA UN INeAR RAW QUE CONTIENE EL CANAL INeAR y el EOG para luego quitar los artefactos y quedarme con el inear resultante---#
        inear_info = mne.create_info(ch_names=['InEar'], sfreq=sfreq, ch_types=['eeg'])  
        # MNE necesita datos en formato (n_channels, n_samples), así que convertimos el array 1D a 2D
        inear_data_2d = inear_signal.reshape(1, -1)  # (1, n_samples)
        
        inear_raw = mne.io.RawArray(inear_data_2d, inear_info)
        # --- Extraer niveles que se repiten cada 4 elementos de 'experiment' ---
        ########## OBTENGO ARREGLOS A PARTIR DE EXPERIMENT QUE SOLO TIENE LOS NIVELES Y OTOR QUE SOLO TIENE LOS TRIALS ###########
        # Asegurar forma 2D del experiment
        exp_array = np.asarray(experiment_matrix)

        puntos_de_corte = np.where(exp_array[:, 1] == -99)[0]

        # Usar el índice para seleccionar la primera parte del arreglo
        # Se suma 1 para incluir la fila con el -99 en el resultado
        exp1_array = exp_array[puntos_de_corte[0] : puntos_de_corte[1]]

        trials_array=exp1_array[1::4]
        levels_array=exp1_array[2::4]

        
        # --- LÓGICA DE REINICIO DE TRIALS (Bloques enteros) ---
        mascara_cambios_2 = (trials_array[:, 1] == 0)
        level_changes_all = levels_array[mascara_cambios_2] # TODOS los bloques (0, 1, 2, 3)

        primeras_ocurrencias_tiempo = level_changes_all[:, 0]
        data = np.asarray(data_matrix)
        data_lsl = data[:, 0]

        # Calculamos los onsets y duraciones de TODOS los bloques
        lista_onsets = []
        for timestamp in primeras_ocurrencias_tiempo:
            indice_mas_cercano = np.abs(data_lsl - timestamp).argmin()
            lista_onsets.append(indice_mas_cercano / sfreq)

        ultimo_elemento = levels_array[-1, 0] 
        indice_ultimo = np.abs(data_lsl - ultimo_elemento).argmin()
        lista_onsets.append(indice_ultimo / sfreq)
        
        durations = np.diff(lista_onsets)
        lista_onsets = np.array(lista_onsets[:-1])
        
        # Como usamos todas las clases, las descripciones son directamente las de todos los bloques
        descriptions = level_changes_all[:, 1].astype(str)

        # Recorte de RAW
        ultimo_ts_exp1 = exp1_array[-1, 0]
        end_sample_idx = int(np.abs(data_lsl - ultimo_ts_exp1).argmin())
        end_time_sec = end_sample_idx / sfreq
        inear_raw.crop(tmin=0.0, tmax=end_time_sec, include_tmax=True)

        annotations = mne.Annotations(onset=lista_onsets,
                                      duration=durations,
                                      description=descriptions)
        inear_raw.set_annotations(annotations)
        inear_raw.filter(l_freq=1, h_freq=99.0, fir_design='firwin')

        events, event_id_dict = mne.events_from_annotations(inear_raw)
        print("events ids", event_id_dict)

        window_size = 2.5  
        samples_per_window = int(window_size * sfreq)

        X = []
        y = []
        groups_list = [] 
        conteo_niveles = {} 

        for idx, ev in enumerate(events):
            sample_idx = ev[0]
            # MNE asigna IDs a los eventos basados en orden alfanumérico. 
            # Si descriptions son '0.0', '1.0', '2.0', '3.0', event_code suele ser 1, 2, 3, 4.
            event_code_mne = ev[2] 
            
            # Para estar 100% seguros del nivel original (0, 1, 2 o 3), lo buscamos en el diccionario
            # event_id_dict tiene formato {'0.0': 1, '1.0': 2, ...}
            # Invertimos el dict para buscar el string original
            inv_event_id_dict = {v: k for k, v in event_id_dict.items()}
            nivel_original_str = inv_event_id_dict[event_code_mne]
            nivel_original = int(float(nivel_original_str)) # Convierte '0.0' -> 0

            # Calcular fin del bloque usando la duración exacta
            duration_sec = inear_raw.annotations.duration[idx]
            duration_samples = int(duration_sec * sfreq)
            end_idx = sample_idx + duration_samples

            # Asignación de Grupo (0, 1, 0...) basado en el NIVEL ORIGINAL
            if nivel_original not in conteo_niveles:
                conteo_niveles[nivel_original] = 0
                
            grupo_actual = conteo_niveles[nivel_original] % 2

            for start in range(sample_idx, end_idx - samples_per_window, samples_per_window):
                segment = inear_raw.get_data(start=start, stop=start+samples_per_window)
                X.append(segment)
                
                # --- AGRUPACIÓN DE CLASES ---
                # Si el nivel original es 0 o 1, la clase es 0. Si es 2 o 3, la clase es 1.
                clase_final = 0 if nivel_original < 2 else 1
                y.append(clase_final) 
                
                groups_list.append(grupo_actual) 

            conteo_niveles[nivel_original] += 1
            
        X = np.array(X)
        y = np.array(y)
        groups = np.array(groups_list)

# --- FORMA EN QUE SE OBTIENEN LOS FEATURES --- #        
        X_features = [
            np.concatenate((
                get_band_powers(x.squeeze(), sfreq, bands),
                wavelet_energies(x.squeeze()),
                #np.array([np.mean(x.squeeze())]),
                #np.array([np.std(x.squeeze())]),
                #np.array([np.max(x.squeeze())]),
                #np.array([np.min(x.squeeze())]),
                #np.array([np.median(x.squeeze())]),
                #np.array([stats.skew(x.squeeze())]),
                #np.array([stats.kurtosis(x.squeeze())]), 
                #np.array([np.sum(np.abs(np.diff(x.squeeze())))]) #line_length
            ))
            for x in X
        ] 
        
        # Convertir la lista de características a array de NumPy
        X_features = np.array(X_features)
        
        
        train_model_result = trainModel(X_features, y, groups)


        print(f"Sujeto procesado: {len(X)} ventanas, shape características: {X_features.shape}")
        
        return train_model_result

    except Exception as e:
        print(f"Error procesando {mat_file_path}: {str(e)}")
        return None, None

# --- FUNCIÓN PRINCIPAL PARA PROCESAR MÚLTIPLES SUJETOS ---
def process_multiple_subjects():
    """Procesa múltiples archivos de sujetos y entrena un modelo conjunto."""
    
    # Buscar todos los archivos de sujetos
    subject_files = glob.glob(os.path.join(DATASET_PATH, SUBJECT_PATTERN))
    subject_files.sort()  # Ordenar para procesamiento consistente
    
    print(f"Encontrados {len(subject_files)} archivos de sujetos:")
    for file in subject_files:
        print(f"  - {os.path.basename(file)}")
    
    if not subject_files:
        print("No se encontraron archivos de sujetos. Verifica la ruta y el patrón.")
        return
    
    subjects_results = []
    
    for mat_file in subject_files:
        subject_result = process_single_subject(mat_file)
        if subject_result is not None :
            subjects_results.append(subject_result)
            
    
    if not subjects_results:
        print("No se pudieron procesar archivos de sujetos.")
        return
    
    # ---  RESULTADOS --- #
    mean = np.mean(subjects_results)
    median = np.median(subjects_results)

    print("precision de modelos entrenados para cada sujeto:")
    for i, result in enumerate(subjects_results):
        print(f"sujeto {i + 1}:", result)
    
    print("media de accuracy: ", mean)
    print("mediana de accuracy: ", median)
    

    
    return 0, 1
# --- EJECUTAR EL PROCESAMIENTO ---
if __name__ == "__main__":
    model, scaler = process_multiple_subjects()
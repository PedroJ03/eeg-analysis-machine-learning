# eeg-analysis-machine-learning

Este proyecto implementa una arquitectura modular para la clasificación de carga cognitiva (workload) basada en señales EEG (In-Ear). Reemplaza una colección de scripts dispersos por un sistema unificado, eficiente y científicamente riguroso.

---

## 📁 Estructura del Proyecto

### Directorios Principales
- **`ml_analysis/`**: Paquete principal que contiene la lógica del sistema.
  - **`src/`**: Módulos internos de procesamiento (ver sección abajo).
  - **`legacy/`**: Carpeta que contiene los scripts originales para referencia histórica.
  - **`main.py`**: El orquestador principal del proyecto y único punto de entrada para ejecución.
- **`data/`**: Contiene los archivos `.mat` de los sujetos (`subj001.mat`, etc.) utilizados en la investigación.
- **`entorno_pps/`**: Carpeta del entorno virtual de Python (excluida de Git).

### 🛠️ Detalles de `/src`

Cada archivo en `src/` cumple una función específica dentro del pipeline:

1. **`config.py`**: 
   - Centraliza constantes globales como la frecuencia de muestreo (`SFREQ`), duración de ensayos (`TRIAL_DUR`), nombres de canales y las bandas de frecuencia EEG (Delta a Gamma).
2. **`data_loader.py`**: 
   - Contiene los cargadores de datos. Diferencia entre la carga por ensayos segmentados (modo `trial`) y la segmentación continua por ventanas usando la librería MNE (modo `window`).
3. **`features.py`**: 
   - Implementa la extracción de características. Permite seleccionar dinámicamente entre potencias de banda, energías Wavelet y estadísticas temporales.
4. **`models.py`**: 
   - Factoría de modelos. Centraliza la creación de clasificadores (Random Forest, SVM), la evaluación de métricas y el formateo de resultados para la consola.

---

## 🚀 Instalación

Asegúrate de tener Python 3.10 o superior.

1. **Preparar el entorno**:
   ```bash
   python -m venv entorno_pps
   source entorno_pps/bin/activate
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Guía de Ejecución

El proyecto se ejecuta llamando al módulo principal. La sintaxis básica es:

```bash
python -m ml_analysis.main [ARGUMENTOS]
```

### Argumentos del Comando

| Argumento | Opciones | Descripción |
| :--- | :--- | :--- |
| `--config` | `trial`, `window` | **trial**: Basado en bloques del experimento. **window**: Ventanas deslizantes (Sliding Windows). |
| `--task` | `4level`, `binary`, `extremes` | **4level**: Niveles 0-3. **binary**: 0-1 vs 2-3. **extremes**: Solo 0 vs 3. |
| `--model` | `rf`, `svm` | Algoritmo classificatorio: Random Forest o SVM con RBF. |
| `--eval` | `ws`, `group` | **ws**: Validación cruzada sujeto a sujeto. **group**: Entrenamiento en grupo completo. |
| `--features` | (String) | Combinación de letras: `B` (Bands), `Bg` (Bands + Gamma), `E` (Stats), `W` (Wavelet). |

### Ejemplos Prácticos

- **Reproducir Tabla 1 (Binario, RF):**
  ```bash
  python -m ml_analysis.main --config trial --task binary --model rf --features BEW
  ```

- **Análisis de Extremos riguroso (Configuración Window):**
  ```bash
  python -m ml_analysis.main --config window --task extremes --model rf --features BW
  ```

- **Clasificación SVM con todas las características:**
  ```bash
  python -m ml_analysis.main --config trial --task 4level --model svm --features BgEW
  ```

---

## 🔬 Consideraciones Científicas

- **Data Leakage**: En el modo `window`, el sistema utiliza `GroupKFold`. Esto previene que ventanas contiguas del mismo bloque se mezclen en entrenamiento y testeo, garantizando que el accuracy reportado sea realista y no inflado por proximidad temporal.
- **Escalado**: Para modelos SVM, el sistema aplica automáticamente a `StandardScaler` por cada fold de entrenamiento.
- **Filtros**: El modo `window` aplica un filtro FIR (1-99 Hz) sobre el canal In-Ear.

---

## 📚 Referencia y Créditos

Los datos utilizados en este proyecto provienen del siguiente estudio:

**"Estimating cognitive workload using a commercial in-ear EEG headset"**

Si utilizas este código o los datos para fines académicos, por favor cita el trabajo original.

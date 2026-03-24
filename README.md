# Clasificación de Carga Cognitiva mediante EEG In-Ear

Sistema de Machine Learning para la clasificación de workload cognitivo basado en señales EEG capturadas mediante auriculares in-ear comerciales. Implementa una arquitectura modular y científicamente rigurosa, reemplazando scripts dispersos por un pipeline unificado y profesional.

---

## 📊 Resumen de Resultados

**Mejor Accuracy Alcanzado:** 77.27% (Intra-Sujeto, Tarea Extremos, Features Bg+W)

### Hallazgos Principales

| Aspecto | Resultado Clave |
|---------|----------------|
| **Mejor Combinación de Features** | Bandas con Gamma + Wavelet (Bg+W) |
| **Mejor Modelo** | Random Forest (consistentemente superior a SVM) |
| **Tarea más Fácil** | Extremos (0-back vs 3-back) - 77.27% accuracy |
| **Tarea más Difícil** | 4 Clases (0,1,2,3) - 49.19% accuracy máximo |
| **Brecha WS-LOSO** | 27.3% de diferencia entre Intra-Sujeto y Entre-Sujetos |
| **Impacto de Gamma** | +7.6% de mejora al incluir banda gamma |

---

## 📁 Estructura del Proyecto

```
eeg-analysis-machine-learning/
├── 📁 ml_analysis/              # Código fuente principal
│   ├── 📁 src/                  # Módulos del pipeline
│   │   ├── config.py            # Constantes globales (SFREQ, BANDS)
│   │   ├── data_loader.py       # Carga de datos .mat
│   │   ├── features.py          # Extracción de características
│   │   └── models.py            # Clasificadores (RF, SVM)
│   ├── 📁 legacy/               # Scripts originales (referencia)
│   ├── main.py                  # Entry point CLI
│   └── run_all_experiments.py   # Ejecutor de experimentos batch
│
├── 📁 docs/                     # Documentación y figuras
│   ├── 📁 figuras/              # Gráficos de análisis
│   │   ├── comparativa_detallada/
│   │   ├── loso/
│   │   ├── modelo_config/
│   │   └── ws/
│   ├── 📁 tablas/               # Tablas formateadas
│   └── analisis_resultados_tesis.txt
│
├── 📁 output/                   # Resultados generados
│   ├── 📁 figures/              # Figuras de salida
│   └── 📁 results/              # Logs de experimentos
│
├── 📁 data/                     # Datos EEG (.mat)
├── requirements.txt             # Dependencias
└── README.md                    # Este archivo
```

---

## 🚀 Instalación Rápida

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/eeg-analysis-machine-learning.git
cd eeg-analysis-machine-learning

# Crear entorno virtual
python -m venv entorno_pps
source entorno_pps/bin/activate  # Linux/Mac
# entorno_pps\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## 💻 Uso

### Ejecución Individual

```bash
python -m ml_analysis.main [ARGUMENTOS]
```

**Parámetros:**

| Argumento | Opciones | Descripción |
|-----------|----------|-------------|
| `--config` | `trial`, `window` | Segmentación: por bloques o ventanas deslizantes |
| `--task` | `4level`, `binary`, `extremes` | Tipo de clasificación |
| `--model` | `rf`, `svm` | Algoritmo: Random Forest o SVM |
| `--eval` | `ws`, `group` | Evaluación: Intra-Sujeto o Entre-Sujetos (LOSO) |
| `--features` | Ej: `Bg,E,W` | Combinación de características |

**Ejemplos:**

```bash
# Mejor configuración encontrada
python -m ml_analysis.main --config trial --task extremes --model rf --features Bg,W

# Clasificación binaria con SVM
python -m ml_analysis.main --config window --task binary --model svm --features Bg,E,W

# Todas las características, 4 clases
python -m ml_analysis.main --config trial --task 4level --model rf --features Bg,E,W
```

### Ejecución Batch (Todos los Experimentos)

```bash
# Ejecutar todas las combinaciones para un conjunto de features
python ml_analysis/run_all_experiments.py --features Bg,W
```

---

## 📈 Características (Features)

| Código | Descripción | Dimensión |
|--------|-------------|-----------|
| `B` | Band Power (δ, θ, α, β) - Sin Gamma | 16 features |
| `Bg` | Band Power (δ, θ, α, β, γ) - Con Gamma | 20 features |
| `E` | Estadísticas Temporales (media, std, skewness, kurtosis, line length) | 15 features |
| `W` | Energías Wavelet (Daubechies-4, nivel 4) | ~80 features |

**Combinaciones Evaluadas:**
- Bg,E,W (Todas las features)
- Bg,W (Mejor rendimiento)
- Bg (Solo bandas con gamma)
- E,W (Sin información espectral)
- E (Solo estadísticas)
- B (Bandas sin gamma)

---

## 🔬 Metodología

### Pipeline de Procesamiento

```
Datos .mat → Segmentación → Extracción Features → Clasificación → Evaluación
     ↓              ↓                ↓                ↓              ↓
  MNE Raw      Trial/Window    B/Bg/E/W         RF/SVM       WS/LOSO
```

### Consideraciones Científicas

1. **Prevención de Data Leakage**
   - `GroupKFold` para ventanas contiguas
   - `LeaveOneGroupOut` para evaluación inter-sujeto

2. **Normalización**
   - `StandardScaler` intra-sujeto antes de pooling
   - Normalización por fold en validación cruzada

3. **Balance de Clases**
   - `class_weight='balanced'` en todos los clasificadores

4. **Validación Cruzada**
   - **Intra-Sujeto (WS):** Entrena y testea en el mismo sujeto
   - **Entre-Sujetos (LOSO):** Generaliza a sujetos no vistos

---

## 📊 Resultados Detallados

### Ranking de Combinaciones de Features

| Rank | Combinación | Mejor Acc | Promedio |
|------|-------------|-----------|----------|
| 1 | Bg + W | **77.27%** | 54.32% |
| 2 | Bg + E + W | 76.63% | 53.80% |
| 3 | E + W | 74.91% | 53.17% |
| 4 | Bg | 74.42% | 53.42% |
| 5 | E | 73.19% | 51.70% |
| 6 | B | 66.81% | 50.19% |

### Comparación por Tarea

| Tarea | WS Media | LOSO Media | Dificultad |
|-------|----------|------------|------------|
| Extremos (0 vs 3) | 67.22% | 49.75% | ⭐ Fácil |
| Binario (0-1 vs 2-3) | 62.50% | 51.14% | ⭐⭐ Media |
| 4 Clases (0,1,2,3) | 37.79% | 25.06% | ⭐⭐⭐ Difícil |

### Comparación de Modelos

| Modelo | WS Media | LOSO Media |
|--------|----------|------------|
| Random Forest | 56.11% | 43.05% |
| SVM | 54.90% | 40.63% |

---

## 📚 Documentación

### Archivos de Análisis

- **`docs/analisis_resultados_tesis.txt`** - Análisis completo con tablas formateadas
- **`docs/tablas/tablas_para_tesis.txt`** - Tablas listas para copiar a LaTeX/Word

### Figuras

Las figuras se organizan en `docs/figuras/`:

```
docs/figuras/
├── comparativa_detallada/     # Análisis comparativo WS vs LOSO
├── loso/                       # Resultados Entre-Sujetos
├── modelo_config/              # Comparación modelos y configuraciones
└── ws/                         # Resultados Intra-Sujeto
```

---

## 🎯 Configuración Óptima

Para reproducir los mejores resultados:

```bash
# Mejor resultado general (77.27%)
python -m ml_analysis.main \
  --config trial \
  --task extremes \
  --model rf \
  --features Bg,W

# Mejor resultado Entre-Sujetos (54.71%)
python -m ml_analysis.main \
  --config window \
  --task binary \
  --model rf \
  --features Bg,W
```

---

## 🛠️ Tecnologías

- **Python 3.10+**
- **scikit-learn 1.7.1** - ML y validación cruzada
- **MNE 1.10.1** - Procesamiento EEG
- **NumPy, Pandas** - Manipulación de datos
- **Matplotlib, Seaborn** - Visualización
- **SciPy, PyWavelets** - Procesamiento de señales

---

## 📖 Referencias

Los datos utilizados provienen del estudio:

> **"Estimating cognitive workload using a commercial in-ear EEG headset"**

Si utilizas este código o datos para fines académicos, por favor cita el trabajo original.

---

## 🤝 Contribuciones

Este proyecto fue desarrollado como parte de una tesis de Ingeniería en Sistemas. Para contribuciones o preguntas, abre un issue o contacta al autor.

---

## 📄 Licencia

Este proyecto es de uso académico. Consulta el archivo LICENSE para más detalles.

---

**Autor:** [Tu Nombre]  
**Tesis:** Ingeniería en Sistemas - Clasificación de Carga Cognitiva mediante EEG  
**Fecha:** 2025

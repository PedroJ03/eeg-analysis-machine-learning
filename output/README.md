# Estructura de Resultados y Figuras

Este directorio contiene todos los resultados numéricos y figuras generadas para el informe de tesis.

## 📁 results/

Contiene los datos numéricos en formato CSV y TXT.

### 📂 saturacion/
Resultados del análisis de saturación incremental de features:
- `feature_saturation_results.csv` - Análisis SIN banda Gamma (26 features máximo)
- `feature_saturation_results_CON_GAMMA.csv` - Análisis CON banda Gamma (27 features máximo)

### 📂 features_individuales/
Resultados de cada grupo de features evaluado individualmente:
- `results_B.txt` - Bandas espectrales (Delta, Theta, Alpha, Beta) = 4 features
- `results_Bg.txt` - Bandas espectrales CON Gamma = 5 features
- `results_E.txt` - Estadísticas temporales = 8 features
- `results_W.txt` - Energías Wavelet = 5 features
- `results_R.txt` - Ratios espectrales (TAR, BTR, GAR) = 3 features
- `results_H.txt` - Parámetros de Hjorth = 3 features
- `results_N.txt` - Entropía (Spectral + Shannon) = 2 features
- `results_Z.txt` - Zero-crossing rate = 1 feature
- `results_Bg_E_W.txt` - Combinación Bg+E+W
- `results_Bg_W.txt` - Combinación Bg+W
- `results_E_W.txt` - Combinación E+W

## 📁 figuras/

Contiene las 13 figuras oficiales del informe, organizadas por tipo.

### 📂 saturacion/ (2 figuras)
Análisis de saturación de features:
1. `01_saturacion_WS_CON_GAMMA.png` - Within-Subject (datos con Gamma)
2. `02_saturacion_LOSO_CON_GAMMA.png` - Leave-One-Subject-Out (datos con Gamma)

### 📂 features_individuales/ (3 figuras)
Comparación de features individuales WS vs LOSO:
3. `03_features_4CLASES.png` - Clasificación de 4 niveles (0,1,2,3-back)
4. `04_features_BINARIO.png` - Clasificación binaria (0-1 vs 2-3)
5. `05_features_EXTREMOS.png` - Clasificación de extremos (0 vs 3)

### 📂 comparativas/ (8 figuras)
Análisis comparativos diversos:
6. `06_comparacion_distribuciones.png` - Distribución de accuracy por método
7. `07_comparacion_violin.png` - Gráficos de violín por tarea
8. `08_comparacion_medias.png` - Comparación de medias WS vs LOSO
9. `09_comparacion_boxplot_features.png` - Boxplots por grupo de features
10. `10_comparacion_modelos_ws_loso.png` - RF vs SVM en WS y LOSO
11. `11_comparacion_trial_window.png` - Comparación Trial vs Window
12. `12_comparacion_rf_svm.png` - Desempeño RF vs SVM
13. `13_comparacion_detallada.png` - Análisis detallado comparativo

## 🔄 Regeneración de Figuras

### Script principal (recomendado)
Para regenerar las figuras principales del informe:

```bash
cd /home/pedroj/Desktop/eeg-analysis-machine-learning
python scripts/generar_graficos_separados.py
```

Esto generará:
- `saturation_WS_only.png`
- `saturation_LOSO_only.png`
- `features_individuales_4CLASES.png`
- `features_individuales_BINARIO.png`
- `features_individuales_EXTREMOS.png`

Luego mover manualmente a las subcarpetas correspondientes y renombrar con el esquema de numeración.

### Otros scripts disponibles en `/scripts/`

- `scripts/saturacion/` - Scripts para análisis de saturación alternativos
- `scripts/comparativas/` - Scripts para gráficos comparativos adicionales
- `scripts/diagramas/` - Scripts para diagramas de arquitectura

Ver `scripts/README.md` para más detalles.

## 📊 Notas

- Todas las figuras están en formato PNG a 300 DPI (calidad de publicación)
- Los datos numéricos están en formato CSV (saturación) y TXT (features individuales)
- El análisis con banda Gamma (Bg) mostró mejoras de +2-3 puntos porcentuales en tareas de extremos

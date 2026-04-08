# Scripts de Generación de Gráficos

Este directorio contiene todos los scripts utilizados para generar las figuras del informe de tesis.

## 📁 Estructura

### 📂 saturacion/
Scripts para generar gráficos de análisis de saturación de features:
- `run_saturation_2x2_mas_offset.py` - Genera gráficos 2x2 de saturación con offset
- `run_saturation_CON_GAMMA.py` - Análisis de saturación incluyendo banda Gamma

### 📂 features/
Scripts para generar gráficos de features individuales:
- (Scripts específicos para análisis de features individuales)

### 📂 comparativas/
Scripts para generar gráficos comparativos:
- `plot_extra.py` - Gráficos comparativos adicionales (modelos, tareas, evaluaciones)

### 📂 diagramas/
Scripts para generar diagramas de arquitectura y flujo:
- `generar_diagramas.py` - Diagramas del pipeline y arquitectura del sistema

### 📄 Raíz de scripts/
- `generar_graficos_separados.py` - **Script principal** que genera:
  - Gráficos de saturación (WS y LOSO separados)
  - Gráficos de features individuales (4 clases, binario, extremos)

## 🚀 Uso

### Generar figuras principales (recomendado)
```bash
python scripts/generar_graficos_separados.py
```

Esto crea en `output/figuras/`:
- `saturation_WS_only.png`
- `saturation_LOSO_only.png`
- `features_individuales_4CLASES.png`
- `features_individuales_BINARIO.png`
- `features_individuales_EXTREMOS.png`

### Generar diagramas de arquitectura
```bash
python scripts/diagramas/generar_diagramas.py
```

### Generar análisis de saturación completo
```bash
python scripts/saturacion/run_saturation_2x2_mas_offset.py
```

### Generar comparaciones adicionales
```bash
python scripts/comparativas/plot_extra.py
```

## 📝 Notas

- Los scripts en subcarpetas (`saturacion/`, `features/`, `comparativas/`, `diagramas/`) son versiones específicas o scripts auxiliares.
- El script principal `generar_graficos_separados.py` en la raíz es el que se debe usar para regenerar las figuras oficiales del informe.
- Todos los scripts generan figuras en alta resolución (300 DPI) adecuadas para publicación.
- Los scripts asumen que los datos están en `output/results/` (CSVs y TXTs).

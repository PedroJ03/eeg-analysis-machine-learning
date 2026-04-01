import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Diagrama 1: Arquitectura
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')
ax.text(5, 11.5, 'Arquitectura General del Sistema', fontsize=20, fontweight='bold', ha='center')

colors = ['#E8F5E9', '#FFF3E0', '#E3F2FD', '#FCE4EC']

b1 = FancyBboxPatch((0.5, 9.5), 9, 1.2, boxstyle="round,pad=0.1", facecolor=colors[0], edgecolor='#2E7D32', linewidth=2.5)
ax.add_patch(b1)
ax.text(5, 10.3, 'DATOS DE ENTRADA', fontsize=14, fontweight='bold', ha='center')
ax.text(5, 9.9, 'Archivos .mat (16 sujetos)', fontsize=11, ha='center', style='italic')

ax.annotate('', xy=(5, 8.8), xytext=(5, 9.5), arrowprops=dict(arrowstyle='->', lw=2.5, color='#1565C0'))

b2 = FancyBboxPatch((0.5, 7.2), 9, 1.4, boxstyle="round,pad=0.1", facecolor=colors[1], edgecolor='#E65100', linewidth=2.5)
ax.add_patch(b2)
ax.text(5, 8.3, 'PREPROCESAMIENTO', fontsize=14, fontweight='bold', ha='center')
ax.text(5, 7.9, 'data_loader.py', fontsize=12, ha='center', color='#E65100', fontweight='bold')
ax.text(5, 7.5, 'Carga .mat -> MNE RawArray -> Filtro FIR (1-99 Hz)', fontsize=10, ha='center')

ax.annotate('', xy=(5, 6.5), xytext=(5, 7.2), arrowprops=dict(arrowstyle='->', lw=2.5, color='#1565C0'))

b3a = FancyBboxPatch((0.5, 5.2), 4, 1.1, boxstyle="round,pad=0.1", facecolor='#D5E8D4', edgecolor='#2E7D32', linewidth=2)
ax.add_patch(b3a)
ax.text(2.5, 5.8, 'Modo TRIAL', fontsize=12, fontweight='bold', ha='center')
ax.text(2.5, 5.5, 'Por bloques experimentales', fontsize=9, ha='center')

b3b = FancyBboxPatch((5.5, 5.2), 4, 1.1, boxstyle="round,pad=0.1", facecolor='#D5E8D4', edgecolor='#2E7D32', linewidth=2)
ax.add_patch(b3b)
ax.text(7.5, 5.8, 'Modo WINDOW', fontsize=12, fontweight='bold', ha='center')
ax.text(7.5, 5.5, 'Ventanas deslizantes', fontsize=9, ha='center')

b4 = FancyBboxPatch((0.5, 3.0), 9, 1.3, boxstyle="round,pad=0.1", facecolor=colors[1], edgecolor='#E65100', linewidth=2.5)
ax.add_patch(b4)
ax.text(5, 4.0, 'EXTRACCION DE CARACTERISTICAS', fontsize=14, fontweight='bold', ha='center')
ax.text(5, 3.6, 'features.py', fontsize=12, ha='center', color='#E65100', fontweight='bold')
ax.text(5, 3.2, 'B/Bg (Band Power) | E (Stats) | W (Wavelet)', fontsize=10, ha='center')

ax.annotate('', xy=(5, 2.3), xytext=(5, 3.0), arrowprops=dict(arrowstyle='->', lw=2.5, color='#1565C0'))

b5 = FancyBboxPatch((0.5, 0.8), 9, 1.3, boxstyle="round,pad=0.1", facecolor=colors[2], edgecolor='#1565C0', linewidth=2.5)
ax.add_patch(b5)
ax.text(5, 1.8, 'CLASIFICACION ML', fontsize=14, fontweight='bold', ha='center')
ax.text(5, 1.4, 'models.py', fontsize=12, ha='center', color='#1565C0', fontweight='bold')
ax.text(5, 1.0, 'Random Forest | SVM (RBF) | Validacion: WS / LOSO', fontsize=10, ha='center')

ax.annotate('', xy=(5, 0.3), xytext=(5, 0.8), arrowprops=dict(arrowstyle='->', lw=2.5, color='#1565C0'))

b6 = FancyBboxPatch((3, 0), 4, 0.25, boxstyle="round,pad=0.05", facecolor=colors[3], edgecolor='#C62828', linewidth=2)
ax.add_patch(b6)
ax.text(5, 0.125, 'Accuracy | Classification Report | Confusion Matrix', fontsize=9, ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('docs/diagramas/01_arquitectura_general.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('OK: 01_arquitectura_general.png')

# Diagrama 2: Pipeline
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis('off')
ax.text(8, 5.5, 'Pipeline de Procesamiento de Datos EEG', fontsize=18, fontweight='bold', ha='center')

etapas = [
    (0.5, 2.5, 'Datos RAW', '.mat files', '#E3F2FD'),
    (2.5, 2.5, 'Carga', 'load_mat_data()', '#BBDEFB'),
    (4.5, 2.5, 'Filtros', 'FIR 1-99 Hz', '#90CAF9'),
    (6.5, 2.5, 'Segmentacion', 'Trial/Window', '#64B5F6'),
    (8.5, 2.5, 'Features', 'B/Bg/E/W', '#42A5F5'),
    (10.5, 2.5, 'Normalizacion', 'StandardScaler', '#2196F3'),
    (12.5, 2.5, 'ML Model', 'RF/SVM', '#1E88E5'),
    (14.5, 2.5, 'Evaluacion', 'WS/LOSO', '#1976D2'),
]

for i, (x, y, title, content, color) in enumerate(etapas):
    b = FancyBboxPatch((x, y), 1.6, 2, boxstyle="round,pad=0.08", facecolor=color, edgecolor='#1565C0', linewidth=2)
    ax.add_patch(b)
    ax.text(x+0.8, y+1.7, title, fontsize=10, fontweight='bold', ha='center', color='#0D47A1')
    ax.text(x+0.8, y+0.9, content, fontsize=8, ha='center', va='center')
    if i < len(etapas) - 1:
        ax.annotate('', xy=(etapas[i+1][0]-0.05, y+1), xytext=(x+1.65, y+1),
                   arrowprops=dict(arrowstyle='->', mutation_scale=25, lw=2.5, color='#1565C0'))

ax.text(8, 0.5, 'Pipeline modular: cada etapa es independiente y reutilizable', fontsize=10, ha='center', style='italic', color='#666666')

plt.tight_layout()
plt.savefig('docs/diagramas/02_pipeline_procesamiento.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('OK: 02_pipeline_procesamiento.png')

# Diagrama 3: Flujo de Trabajo
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')
ax.text(7, 9.5, 'Flujo de Trabajo: Ejecucion de Experimentos', fontsize=18, fontweight='bold', ha='center')

pasos = [
    (0.5, 8, '1. CONFIGURACION', 'Seleccionar parametros', '#E8F5E9', '#2E7D32'),
    (0.5, 6.5, '2. PREPARACION', 'Carga y preprocesamiento', '#FFF3E0', '#E65100'),
    (0.5, 5, '3. EXTRACCION', 'features.py', '#E3F2FD', '#1565C0'),
    (0.5, 3.5, '4. ENTRENAMIENTO', 'models.py', '#F3E5F5', '#7B1FA2'),
    (0.5, 2, '5. EVALUACION', 'WS / LOSO', '#FFEBEE', '#C62828'),
]

for i, (x, y, title, desc, fc, ec) in enumerate(pasos):
    b = FancyBboxPatch((x, y), 4, 1.1, boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec, linewidth=2.5)
    ax.add_patch(b)
    ax.text(x+2, y+0.75, title, fontsize=12, fontweight='bold', ha='center')
    ax.text(x+2, y+0.35, desc, fontsize=10, ha='center')
    if i < len(pasos) - 1:
        ax.annotate('', xy=(x+2, pasos[i+1][1]+1.1), xytext=(x+2, y),
                   arrowprops=dict(arrowstyle='->', mutation_scale=30, lw=2, color=ec))

# Panel derecho
detalles = [
    (6, 8.5, 'Metodos de Evaluacion', 'WS: Intra-Sujeto\\nLOSO: Entre-Sujetos', '#FFECB3', '#FF6F00'),
    (10, 8.5, 'Modelos ML', 'Random Forest\\nSVM (RBF)\\nGridSearchCV', '#C8E6C9', '#2E7D32'),
    (6, 6.5, 'Tareas', '4-Level: 0,1,2,3\\nBinary: 0-1 vs 2-3\\nExtremes: 0 vs 3', '#E1F5FE', '#0288D1'),
    (6, 4, 'Features', 'B: Band Power\\nBg: +Gamma\\nE: Estadisticas\\nW: Wavelet', '#F3E5F5', '#7B1FA2'),
]

for x, y, title, content, fc, ec in detalles:
    b = FancyBboxPatch((x, y), 3.5, 1.4 if 'Tareas' not in title else 1.8, boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec, linewidth=2)
    ax.add_patch(b)
    ax.text(x+1.75, y+1.1 if 'Tareas' not in title else 1.5, title, fontsize=11, fontweight='bold', ha='center')
    ax.text(x+1.75, y+0.6 if 'Tareas' not in title else 0.9, content, fontsize=9, ha='center', va='center')

# Mejor resultado
best = FancyBboxPatch((6, 1.5), 7.5, 1.5, boxstyle="round,pad=0.1", facecolor='#FFF8E1', edgecolor='#F57F17', linewidth=3)
ax.add_patch(best)
ax.text(9.75, 2.7, 'MEJOR RESULTADO', fontsize=12, fontweight='bold', ha='center', color='#F57F17')
ax.text(9.75, 2.2, 'Accuracy: 77.27%', fontsize=11, ha='center', fontweight='bold')
ax.text(9.75, 1.8, 'Trial | Extremes | RF | Bg+W', fontsize=9, ha='center')

plt.tight_layout()
plt.savefig('docs/diagramas/03_flujo_trabajo.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('OK: 03_flujo_trabajo.png')

# Diagrama 4: Componentes
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')
ax.text(7, 9.5, 'Diagrama de Componentes del Sistema', fontsize=18, fontweight='bold', ha='center')

modulos = [
    (0.5, 7, 'config.py', 'Constantes Globales\\nSFREQ=200\\nBANDS\\nTRIAL_DUR', '#E8F5E9', '#2E7D32'),
    (4, 7, 'data_loader.py', 'Carga de Datos\\nload_mat_data()\\nget_trial_segments()\\nget_window_epochs()', '#FFF3E0', '#E65100'),
    (7.5, 7, 'features.py', 'Extraccion Features\\nBand Power (B/Bg)\\nEstadisticas (E)\\nWavelet (W)', '#E3F2FD', '#1565C0'),
    (11, 7, 'models.py', 'Modelos ML\\nRandom Forest\\nSVM (RBF)\\nGridSearchCV', '#F3E5F5', '#7B1FA2'),
]

for x, y, title, content, fc, ec in modulos:
    b = FancyBboxPatch((x, y), 3, 2, boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec, linewidth=3)
    ax.add_patch(b)
    ax.text(x+1.5, y+1.7, title, fontsize=12, fontweight='bold', ha='center', color=ec)
    ax.text(x+1.5, y+0.9, content, fontsize=9, ha='center', va='center')

# Entry points
main = FancyBboxPatch((0.5, 4), 3, 1.2, boxstyle="round,pad=0.1", facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2.5)
ax.add_patch(main)
ax.text(2, 4.8, 'main.py', fontsize=12, fontweight='bold', ha='center', color='#B71C1C')
ax.text(2, 4.3, 'Entry Point CLI', fontsize=10, ha='center')

run = FancyBboxPatch((0.5, 2.2), 3, 1.2, boxstyle="round,pad=0.1", facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2.5)
ax.add_patch(run)
ax.text(2, 3, 'run_all_experiments.py', fontsize=10, fontweight='bold', ha='center', color='#B71C1C')
ax.text(2, 2.5, 'Batch Runner', fontsize=10, ha='center')

# Conexiones
conexiones = [
    ((2, 4), (2, 5.2), '#C62828'),
    ((2, 2.8), (2, 4), '#C62828'),
    ((3.5, 5), (4, 8), '#2E7D32'),
    ((5.5, 8), (7.5, 8), '#1565C0'),
    ((10, 8), (11, 8), '#7B1FA2'),
    ((12.5, 7), (12.5, 4), '#7B1FA2'),
]

for (x1, y1), (x2, y2), color in conexiones:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', color=color, lw=2))

plt.tight_layout()
plt.savefig('docs/diagramas/04_componentes.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('OK: 04_componentes.png')

print('\\nTodos los diagramas generados exitosamente!')

"""
Gráfico: Accuracy por Tarea y Modelo con Desvío (Error Bars)
Similar a "saturation by task and model" pero con barras de error
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from collections import defaultdict

# Configuración de estilo profesional
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11

RESULTS_DIR = Path('/home/pedroj/Desktop/eeg-analysis-machine-learning/output/results')

# Parsear resultados (solo features INDIVIDUALES: B, Bg, E, W, R, H, N, Z)
FEATURES_INDIVIDUALES = ['B', 'Bg', 'E', 'W', 'R', 'H', 'N', 'Z']
all_data = []

for feature in FEATURES_INDIVIDUALES:
    results_file = RESULTS_DIR / f'results_{feature}.txt'
    feature_name = results_file.stem.replace('results_', '')
    
    with open(results_file, 'r') as f:
        content = f.read()
    
    pattern = r'Config:\s+(\w+)\s+\|\s+Task:\s+(\w+)\s+\|\s+Eval:\s+(\w+)\s+\|\s+Model:\s+(\w+)\s+.*ACC:\s+([\d.]+)'
    matches = re.findall(pattern, content)
    
    for config, task, eval_type, model, acc in matches:
        all_data.append({
            'feature': feature_name,
            'config': config,
            'task': task,
            'eval': eval_type,
            'model': model,
            'accuracy': float(acc) * 100
        })

# Agrupar por (task, model, eval) - promedio sobre TODAS las features
grouped = defaultdict(list)
for d in all_data:
    key = (d['task'], d['model'], d['eval'])
    grouped[key].append(d['accuracy'])

# Calcular estadísticas
stats = {}
for key, values in grouped.items():
    stats[key] = {
        'mean': np.mean(values),
        'std': np.std(values, ddof=1),
        'n': len(values),
        'sem': np.std(values, ddof=1) / np.sqrt(len(values))  # Error estándar de la media
    }

# Configuración del gráfico - 1 fila, 3 columnas (una por tarea)
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('Accuracy por Tarea y Modelo con Barras de Error (±1 Desvío Estándar)\n' + 
             'Promedio sobre todas las features individuales (B, Bg, E, W, R, H, N, Z)', 
             fontsize=14, fontweight='bold', y=0.98)

# Tasks y modelos
tasks = ['4level', 'binary', 'extremes']
task_labels = ['4 Clases', 'Binario', 'Extremos']
models = ['rf', 'svm']
model_labels = ['Random Forest', 'SVM']

# Colores
ws_color = '#2E86AB'
loso_color = '#F24236'

# Crear subplots - uno por tarea
for idx, (task, task_label) in enumerate(zip(tasks, task_labels)):
    ax = axes[idx]
    
    # Preparar datos para cada modelo
    x_pos = np.arange(len(models))
    width = 0.35
    
    ws_means = []
    ws_stds = []
    loso_means = []
    loso_stds = []
    
    for model in models:
        ws_key = (task, model, 'ws')
        loso_key = (task, model, 'group')
        
        ws_stats = stats.get(ws_key, {'mean': 0, 'std': 0})
        loso_stats = stats.get(loso_key, {'mean': 0, 'std': 0})
        
        ws_means.append(ws_stats['mean'])
        ws_stds.append(ws_stats['std'])
        loso_means.append(loso_stats['mean'])
        loso_stds.append(loso_stats['std'])
    
    # Barras con error (usando desvío estándar, no SEM)
    bars1 = ax.bar(x_pos - width/2, ws_means, width,
                   yerr=ws_stds,
                   capsize=6,
                   label='Intra-Sujeto (WS)',
                   color=ws_color,
                   alpha=0.9,
                   edgecolor='black',
                   linewidth=1.2,
                   error_kw={'linewidth': 2, 'ecolor': 'black', 'capthick': 1.5})
    
    bars2 = ax.bar(x_pos + width/2, loso_means, width,
                   yerr=loso_stds,
                   capsize=6,
                   label='Entre-Sujetos (LOSO)',
                   color=loso_color,
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=1.2,
                   error_kw={'linewidth': 2, 'ecolor': 'black', 'capthick': 1.5})
    
    # Configurar ejes
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title(task_label, fontweight='bold', fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.set_ylim(0, 85)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Línea de chance
    if task == '4level':
        ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.text(0.5, 27, 'Chance (25%)', fontsize=9, color='gray', ha='center', style='italic')
    else:
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.text(0.5, 52, 'Chance (50%)', fontsize=9, color='gray', ha='center', style='italic')
    
    # Agregar valores en las barras
    for i, (mean, std) in enumerate(zip(ws_means, ws_stds)):
        if mean > 0:
            ax.text(i - width/2, mean + std + 3, f'{mean:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.text(i - width/2, mean + std + 7, f'±{std:.1f}%',
                   ha='center', va='bottom', fontsize=8, color='gray', style='italic')
    
    for i, (mean, std) in enumerate(zip(loso_means, loso_stds)):
        if mean > 0:
            ax.text(i + width/2, mean + std + 3, f'{mean:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.text(i + width/2, mean + std + 7, f'±{std:.1f}%',
                   ha='center', va='bottom', fontsize=8, color='gray', style='italic')

# Leyenda general
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=ws_color, alpha=0.9, edgecolor='black', label='Intra-Sujeto (WS)'),
    Patch(facecolor=loso_color, alpha=0.7, edgecolor='black', label='Entre-Sujetos (LOSO)')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12,
           frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Guardar
output_path = Path('/home/pedroj/Desktop/eeg-analysis-machine-learning/docs/figuras/saturation_by_task_and_model_con_desvio.png')
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Gráfico guardado: {output_path}")

plt.show()

# Mostrar tabla resumen
print("\n" + "="*100)
print("TABLA RESUMEN: PROMEDIO ± DESVÍO POR TAREA Y MODELO")
print("="*100)
print(f"{'Tarea':<15} {'Modelo':<15} {'Evaluación':<15} {'Promedio':<15} {'Desvío':<10} {'N':<5}")
print("-"*100)

for task, task_label in zip(tasks, task_labels):
    for model, model_label in zip(models, ['RF', 'SVM']):
        for eval_type, eval_label in [('ws', 'WS'), ('group', 'LOSO')]:
            key = (task, model, eval_type)
            if key in stats:
                s = stats[key]
                print(f"{task_label:<15} {model_label:<15} {eval_label:<15} {s['mean']:>8.2f}%      {s['std']:>6.2f}%    {s['n']:<5}")

print("="*100)

print("\n✅ Gráfico generado exitosamente!")

"""
Genera gráficos SEPARADOS para el informe de tesis:
1. Dos archivos de saturación: WS (arriba) y LOSO (abajo)
2. Tres archivos de features individuales: 4 Clases, Binario, Extremos

CAMBIOS:
- Saturación: misma estética que el gráfico original (colores, marcadores rellenos)
- Features individuales: sin línea de Chance
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from pathlib import Path
import re
from collections import defaultdict

# Configuración general - FUENTES MÁS GRANDES
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 17

# COLORES
COLOR_WS = '#D2691E'    # Naranja (Intra-Sujeto)
COLOR_LOSO = '#2E5C8A'  # Azul oscuro (Entre-Sujetos)

# Colores para tareas (IGUALES al gráfico original)
TASK_COLORS = {
    'binary': '#2196F3',   # Azul
    'extremes': '#4CAF50', # Verde
    '4level': '#FF5722'    # Naranja/Rojo coral
}

# Directorios - CORREGIDOS PARA /output/
BASE_DIR = Path('/home/pedroj/Desktop/eeg-analysis-machine-learning')
RESULTS_DIR = BASE_DIR / 'output' / 'results' / 'saturacion'
OUTPUT_DIR = BASE_DIR / 'output' / 'figuras' / 'saturacion'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# PARTE 1: GRÁFICOS DE SATURACIÓN SEPARADOS (POR MODELO Y EVALUACIÓN)
# ==============================================================================

def generate_individual_saturation_plots():
    """Genera 4 archivos individuales para saturación: (WS/LOSO) x (RF/SVM)"""
    
    csv_path = RESULTS_DIR / 'feature_saturation_results_CON_GAMMA.csv'
    if not csv_path.exists():
        print(f"Error: No se encontró el archivo de resultados en {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Labels actualizados con Gamma
    PROGRESSIVE_STEPS = [
        {'groups': 'E', 'label': 'Stats'},
        {'groups': 'EBg', 'label': '+ Bandas+γ'},
        {'groups': 'EBgW', 'label': '+ Wavelet'},
        {'groups': 'EBgWR', 'label': '+ Ratios'},
        {'groups': 'EBgWRH', 'label': '+ Hjorth'},
        {'groups': 'EBgWRHN', 'label': '+ Entropia'},
        {'groups': 'EBgWRHNZ', 'label': '+ Z-Cross'},
    ]
    
    step_labels = [s['label'] for s in PROGRESSIVE_STEPS]
    step_x = np.array(range(len(step_labels)))
    
    # Obtener dimensiones (número de features)
    n_features_per_step = []
    for step_idx in step_x:
        subset = df[df['step'] == step_idx]
        if not subset.empty:
            n_features_per_step.append(int(subset['n_features'].iloc[0]))
        else:
            n_features_per_step.append(0)
    
    # Configuraciones de exportación
    configs = [
        ('ws', 'rf', '01_saturacion_WS_RF', 'Intra-Sujeto (WS) - Random Forest'),
        ('ws', 'svm', '01_saturacion_WS_SVM', 'Intra-Sujeto (WS) - SVM'),
        ('group', 'rf', '02_saturacion_LOSO_RF', 'Entre-Sujetos (LOSO) - Random Forest'),
        ('group', 'svm', '02_saturacion_LOSO_SVM', 'Entre-Sujetos (LOSO) - SVM'),
    ]
    
    offset = 0.28
    
    for eval_mode, model, filename, title in configs:
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_title(title, fontsize=24, fontweight='bold', pad=25)
        
        # Extraer datos de forma robusta
        accs_bin, stds_bin = [], []
        accs_ext, stds_ext = [], []
        accs_4l, stds_4l = [], []
        
        for step_idx in step_x:
            # Binario
            b = df[(df['step'] == step_idx) & (df['eval'] == eval_mode) & 
                   (df['model'] == model) & (df['task'] == 'binary')]
            accs_bin.append(b['accuracy'].mean() if not b.empty else 0)
            stds_bin.append(b['accuracy'].std() if not b.empty else 0)
            
            # Extremos
            e = df[(df['step'] == step_idx) & (df['eval'] == eval_mode) & 
                   (df['model'] == model) & (df['task'] == 'extremes')]
            accs_ext.append(e['accuracy'].mean() if not e.empty else 0)
            stds_ext.append(e['accuracy'].std() if not e.empty else 0)
            
            # 4 Clases
            l = df[(df['step'] == step_idx) & (df['eval'] == eval_mode) & 
                   (df['model'] == model) & (df['task'] == '4level')]
            accs_4l.append(l['accuracy'].mean() if not l.empty else 0)
            stds_4l.append(l['accuracy'].std() if not l.empty else 0)

        # Plotear
        x_bin = step_x - offset
        ax.errorbar(x_bin, accs_bin, yerr=stds_bin, fmt='s-', color=TASK_COLORS['binary'],
                   label='Binario', linewidth=3.5, markersize=14, capsize=7, capthick=2.5,
                   markerfacecolor=TASK_COLORS['binary'], markeredgewidth=2.5)
        
        ax.errorbar(step_x, accs_ext, yerr=stds_ext, fmt='o-', color=TASK_COLORS['extremes'],
                   label='Extremos', linewidth=3.5, markersize=14, capsize=7, capthick=2.5,
                   markerfacecolor=TASK_COLORS['extremes'], markeredgewidth=2.5)
        
        x_4l = step_x + offset
        ax.errorbar(x_4l, accs_4l, yerr=stds_4l, fmt='^-', color=TASK_COLORS['4level'],
                   label='4 Clases', linewidth=3.5, markersize=14, capsize=7, capthick=2.5,
                   markerfacecolor=TASK_COLORS['4level'], markeredgewidth=2.5)
        
        # Anotaciones
        for i in range(len(step_x)):
            ax.annotate(f'{accs_bin[i]:.2f}', (x_bin[i], accs_bin[i]),
                       textcoords='offset points', xytext=(-5, 22),
                       fontsize=11, ha='center', color='black', fontweight='bold')
            
            y_off_ext = 25 if (eval_mode == 'ws' or i % 2 == 0) else -30
            ax.annotate(f'{accs_ext[i]:.2f}', (step_x[i], accs_ext[i]),
                       textcoords='offset points', xytext=(0, y_off_ext),
                       fontsize=11, ha='center', color='black', fontweight='bold')
            
            y_off_4 = 22 if eval_mode == 'ws' else -25
            ax.annotate(f'{accs_4l[i]:.2f}', (x_4l[i], accs_4l[i]),
                       textcoords='offset points', xytext=(5, y_off_4),
                       fontsize=11, ha='center', color='black', fontweight='bold')

        # Ejes y Leyenda
        ax.set_xticks(step_x)
        ax.set_xticklabels(step_labels, rotation=35, ha='right', fontsize=16)
        ax.set_ylabel('Accuracy', fontsize=22, fontweight='bold')
        ax.set_xlabel('Grupos de Features (acumulativos)', fontsize=20, fontweight='bold', labelpad=15)
        
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(step_x)
        ax2.set_xticklabels([f'{n}d' for n in n_features_per_step], fontsize=14, color='gray')
        ax2.set_xlabel('Dimensiones', fontsize=16, color='gray', labelpad=10)
        
        ax.legend(loc='lower right' if eval_mode == 'ws' else 'center left', 
                  fontsize=16, framealpha=0.95, edgecolor='black', shadow=True)
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        ax.set_ylim(0.20, 0.85 if eval_mode == 'ws' else 0.65)
            
        plt.tight_layout()
        out_path = OUTPUT_DIR / f'{filename}.png'
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Generado: {out_path.name}")
        plt.close(fig)


# ==============================================================================
# PARTE 2: FEATURES INDIVIDUALES SEPARADOS POR TAREA (SIN CHANCE)
# ==============================================================================

def generate_features_individuales_separated():
    """Genera 3 archivos separados: 4 Clases, Binario, Extremos (SIN línea de Chance)"""
    
    RESULTS_TEXT_DIR = BASE_DIR / 'output' / 'results' / 'features_individuales'
    FEATURES_INDIVIDUALES = ['B', 'Bg', 'E', 'W', 'R', 'H', 'N', 'Z']
    all_data = []
    
    for feature in FEATURES_INDIVIDUALES:
        results_file = RESULTS_TEXT_DIR / f'results_{feature}.txt'
        if not results_file.exists():
            print(f"Advertencia: No se encontró {results_file}")
            continue
        
        with open(results_file, 'r') as f:
            content = f.read()
        
        pattern = r'Config:\s+(\w+)\s+\|\s+Task:\s+(\w+)\s+\|\s+Eval:\s+(\w+)\s+\|\s+Model:\s+(\w+)\s+.*ACC:\s+([\d.]+)'
        matches = re.findall(pattern, content)
        
        for config, task, eval_type, model, acc in matches:
            all_data.append({
                'feature': feature,
                'config': config,
                'task': task,
                'eval': eval_type,
                'model': model,
                'accuracy': float(acc) * 100
            })
    
    if not all_data:
        print("No se encontraron datos de features individuales")
        return
    
    # Agrupar estadísticas
    grouped = defaultdict(lambda: defaultdict(list))
    for d in all_data:
        key = (d['task'], d['model'], d['eval'])
        grouped[key][d['feature']].append(d['accuracy'])
    
    stats = {}
    for key, features_dict in grouped.items():
        stats[key] = {}
        for feat, values in features_dict.items():
            stats[key][feat] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1)
            }
    
    feature_order = ['B', 'Bg', 'E', 'W', 'R', 'H', 'N', 'Z']
    feature_names = ['B\n(Bandas)', 'Bg\n(Bandas+γ)', 'E\n(Stats)', 'W\n(Wavelet)', 
                     'R\n(Ratios)', 'H\n(Hjorth)', 'N\n(Entropy)', 'Z\n(Zero-C)']
    
    models = ['rf', 'svm']
    model_labels = ['Random Forest (RF)', 'SVM']
    
    # Generar 3 archivos separados
    tasks_config = [
        ('4level', '4 Clases'),
        ('binary', 'Binario'),
        ('extremes', 'Extremos')
    ]
    
    for task, task_label in tasks_config:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        fig.suptitle(f'Accuracy por Features Individuales - {task_label}\n' + 
                     f'(B=Bandas, Bg=Bandas+γ, E=Estadísticas, W=Wavelet, R=Ratios, H=Hjorth, N=Entropía, Z=Zero-Crossing)',
                     fontsize=13, fontweight='bold', y=0.98)
        
        for col_idx, (model, model_label) in enumerate(zip(models, model_labels)):
            ax = axes[col_idx]
            ax.set_title(model_label, fontsize=12, fontweight='bold')
            
            x_pos = np.arange(len(feature_order))
            width = 0.35
            
            ws_means = []
            ws_stds = []
            loso_means = []
            loso_stds = []
            
            for feat in feature_order:
                ws_key = (task, model, 'ws')
                loso_key = (task, model, 'group')
                
                ws_stat = stats.get(ws_key, {}).get(feat, {'mean': 0, 'std': 0})
                loso_stat = stats.get(loso_key, {}).get(feat, {'mean': 0, 'std': 0})
                
                ws_means.append(ws_stat['mean'])
                ws_stds.append(ws_stat['std'])
                loso_means.append(loso_stat['mean'])
                loso_stds.append(loso_stat['std'])
            
            # Barras con colores corregidos (SIN línea de Chance)
            bars1 = ax.bar(x_pos - width/2, ws_means, width,
                          yerr=ws_stds,
                          capsize=5,
                          label='Intra-Sujeto (WS)',
                          color=COLOR_WS,
                          alpha=0.9,
                          edgecolor='black',
                          linewidth=1.2,
                          error_kw={'linewidth': 2, 'ecolor': 'black', 'capthick': 1.5})
            
            bars2 = ax.bar(x_pos + width/2, loso_means, width,
                          yerr=loso_stds,
                          capsize=5,
                          label='Entre-Sujetos (LOSO)',
                          color=COLOR_LOSO,
                          alpha=0.9,
                          edgecolor='black',
                          linewidth=1.2,
                          error_kw={'linewidth': 2, 'ecolor': 'black', 'capthick': 1.5})
            
            # Configurar ejes
            ax.set_ylabel('Accuracy (%)', fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(feature_names, fontsize=9)
            ax.set_ylim(0, 85)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # NOTA: Se eliminó la línea de Chance según solicitud del usuario
            
            # Valores en las barras
            for i, (mean, std) in enumerate(zip(ws_means, ws_stds)):
                if mean > 0:
                    ax.text(i - width/2, mean + std + 2, f'{mean:.1f}',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            for i, (mean, std) in enumerate(zip(loso_means, loso_stds)):
                if mean > 0:
                    ax.text(i + width/2, mean + std + 2, f'{mean:.1f}',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax.legend(loc='best', fontsize=10, framealpha=0.95, 
                     edgecolor='black', fancybox=True, shadow=True)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        
        # Nombre de archivo según tarea
        task_file_map = {
            '4level': 'features_individuales_4CLASES.png',
            'binary': 'features_individuales_BINARIO.png',
            'extremes': 'features_individuales_EXTREMOS.png'
        }
        
        path = OUTPUT_DIR / task_file_map[task]
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Gráfico {task_label} guardado: {path}")
        plt.close(fig)


# ==============================================================================
# EJECUTAR
# ==============================================================================

if __name__ == '__main__':
    print("="*70)
    print("GENERANDO GRÁFICOS SEPARADOS PARA EL INFORME")
    print("="*70)
    
    print("\n1. Generando gráficos de saturación individuales...")
    generate_individual_saturation_plots()
    
    # Comentado para cumplir con el pedido de 'solo saturación' en la ejecución actual
    # print("\n2. Generando gráficos de features individuales separados por tarea...")
    # generate_features_individuales_separated()
    
    print("\n" + "="*70)
    print("¡PROCESO FINALIZADO!")
    print("="*70)
    print(f"\nArchivos guardados en: {OUTPUT_DIR}")
    print("\n--- SATURACIÓN ---")
    print("- 01_saturacion_WS_RF.png")
    print("- 01_saturacion_WS_SVM.png")
    print("- 02_saturacion_LOSO_RF.png")
    print("- 02_saturacion_LOSO_SVM.png")

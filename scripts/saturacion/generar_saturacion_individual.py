import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from pathlib import Path

# Configuración general - FUENTES GIGANTES para máxima legibilidad
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 17

# COLORES
TASK_COLORS = {
    'binary': '#2196F3',   # Azul
    'extremes': '#4CAF50', # Verde
    '4level': '#FF5722'    # Naranja/Rojo coral
}

# Directorios
BASE_DIR = Path('/home/pedroj/Desktop/eeg-analysis-machine-learning')
RESULTS_DIR = BASE_DIR / 'output' / 'results' / 'saturacion'
OUTPUT_DIR = BASE_DIR / 'output' / 'figuras' / 'saturacion'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_individual_saturation_plots():
    """Genera 4 archivos individuales para saturación: (WS/LOSO) x (RF/SVM)"""
    
    csv_path = RESULTS_DIR / 'feature_saturation_results_CON_GAMMA.csv'
    if not csv_path.exists():
        print(f"Error: No se encontró el archivo de resultados en {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Labels de los pasos
    PROGRESSIVE_STEPS = [
        {'label': 'Stats'},
        {'label': '+ Bandas+γ'},
        {'label': '+ Wavelet'},
        {'label': '+ Ratios'},
        {'label': '+ Hjorth'},
        {'label': '+ Entropía'},
        {'label': '+ Z-Cross'},
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
        # (eval_mode, model, prefix, title_suffix)
        ('ws', 'rf', '01_saturacion_WS_RF', 'Intra-Sujeto (WS) - Random Forest'),
        ('ws', 'svm', '01_saturacion_WS_SVM', 'Intra-Sujeto (WS) - SVM'),
        ('group', 'rf', '02_saturacion_LOSO_RF', 'Entre-Sujetos (LOSO) - Random Forest'),
        ('group', 'svm', '02_saturacion_LOSO_SVM', 'Entre-Sujetos (LOSO) - SVM'),
    ]
    
    offset = 0.28  # Separación entre líneas
    
    for eval_mode, model, filename, title in configs:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(title, fontsize=22, fontweight='bold', pad=20)
        
        # 1. Binario (izquierda)
        subset_bin = df[(df['step'].isin(step_x)) & (df['eval'] == eval_mode) & 
                        (df['model'] == model) & (df['task'] == 'binary')].sort_values('step')
        accs_bin = subset_bin['accuracy'].values
        stds_bin = df[(df['step'].isin(step_x)) & (df['eval'] == eval_mode) & 
                      (df['model'] == model) & (df['task'] == 'binary')].groupby('step')['accuracy'].std().values
        
        x_bin = step_x - offset
        ax.errorbar(x_bin, accs_bin, yerr=stds_bin, fmt='s-', color=TASK_COLORS['binary'],
                   label='Binario', linewidth=3, markersize=12,
                   capsize=6, capthick=2, elinewidth=2,
                   markerfacecolor=TASK_COLORS['binary'], markeredgewidth=2)
        
        # 2. Extremos (centro)
        subset_ext = df[(df['step'].isin(step_x)) & (df['eval'] == eval_mode) & 
                        (df['model'] == model) & (df['task'] == 'extremes')].sort_values('step')
        accs_ext = subset_ext['accuracy'].values
        stds_ext = df[(df['step'].isin(step_x)) & (df['eval'] == eval_mode) & 
                      (df['model'] == model) & (df['task'] == 'extremes')].groupby('step')['accuracy'].std().values
        
        ax.errorbar(step_x, accs_ext, yerr=stds_ext, fmt='o-', color=TASK_COLORS['extremes'],
                   label='Extremos', linewidth=3, markersize=12,
                   capsize=6, capthick=2, elinewidth=2,
                   markerfacecolor=TASK_COLORS['extremes'], markeredgewidth=2)
        
        # 3. 4 Clases (derecha)
        subset_4l = df[(df['step'].isin(step_x)) & (df['eval'] == eval_mode) & 
                       (df['model'] == model) & (df['task'] == '4level')].sort_values('step')
        accs_4l = subset_4l['accuracy'].values
        stds_4l = df[(df['step'].isin(step_x)) & (df['eval'] == eval_mode) & 
                     (df['model'] == model) & (df['task'] == '4level')].groupby('step')['accuracy'].std().values
        
        x_4l = step_x + offset
        ax.errorbar(x_4l, accs_4l, yerr=stds_4l, fmt='^-', color=TASK_COLORS['4level'],
                   label='4 Clases', linewidth=3, markersize=12,
                   capsize=6, capthick=2, elinewidth=2,
                   markerfacecolor=TASK_COLORS['4level'], markeredgewidth=2)
        
        # Anotaciones de valores
        for i in range(len(step_x)):
            # Binario
            ax.annotate(f'{accs_bin[i]:.2f}', (x_bin[i], accs_bin[i]),
                       textcoords='offset points', xytext=(-5, 20),
                       fontsize=10, ha='center', color='black', fontweight='bold')
            # Extremos (alternar para evitar solapamiento)
            y_off = 22 if (eval_mode == 'ws' or i % 2 == 0) else -25
            ax.annotate(f'{accs_ext[i]:.2f}', (step_x[i], accs_ext[i]),
                       textcoords='offset points', xytext=(0, y_off),
                       fontsize=10, ha='center', color='black', fontweight='bold')
            # 4 Clases
            y_off_4 = 20 if eval_mode == 'ws' else -22
            ax.annotate(f'{accs_4l[i]:.2f}', (x_4l[i], accs_4l[i]),
                       textcoords='offset points', xytext=(5, y_off_4),
                       fontsize=10, ha='center', color='black', fontweight='bold')

        # Configuración de ejes
        ax.set_xticks(step_x)
        ax.set_xticklabels(step_labels, rotation=35, ha='right', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=20, fontweight='bold')
        ax.set_xlabel('Grupos de Features (acumulativos)', fontsize=18, fontweight='bold', labelpad=15)
        
        # Eje superior con dimensiones
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(step_x)
        ax2.set_xticklabels([f'{n}d' for n in n_features_per_step], fontsize=12, color='gray')
        ax2.set_xlabel('Dimensiones', fontsize=14, color='gray', labelpad=10)
        
        ax.legend(loc='lower right' if eval_mode == 'ws' else 'center left', 
                  fontsize=14, framealpha=0.9, edgecolor='black', shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        
        # Ajustar límites según modo para que no se corten las etiquetas
        if eval_mode == 'ws':
            ax.set_ylim(0.20, 0.85)
        else:
            ax.set_ylim(0.20, 0.65)
            
        plt.tight_layout()
        out_path = OUTPUT_DIR / f'{filename}.png'
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Generado: {out_path.name}")
        plt.close(fig)

if __name__ == '__main__':
    generate_individual_saturation_plots()

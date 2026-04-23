import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Configuración de colores
COLOR_WS = '#ff7f0e'  # Naranja
COLOR_LOSO = '#1f77b4' # Azul

def load_data():
    base_dir = Path('/home/pedroj/Desktop/eeg-analysis-machine-learning/output/results/features_individuales')
    feature_order = ['B', 'Bg', 'E', 'W', 'R', 'H', 'N', 'Z']
    feature_names = ['B\n(Bandas)', 'Bg\n(Bandas+γ)', 'E\n(Stats)', 'W\n(Wavelet)', 
                     'R\n(Ratios)', 'H\n(Hjorth)', 'N\n(Entropy)', 'Z\n(Zero-C)']
    
    results = []
    for feat in feature_order:
        path = base_dir / f"results_{feat}.txt"
        if not path.exists(): continue
        
        with open(path, 'r') as f:
            content = f.read()
            pattern = r'(\w+)\s+\|\s+(4level|binary|extremes)\s+\|\s+(ws|group)\s+\|\s+(\w+)\s+\|\s+([\d.]+)'
            matches = re.findall(pattern, content)
            for m in matches:
                results.append({
                    'Feature': feat,
                    'Config': m[0],
                    'Task': m[1],
                    'Eval': 'WS' if m[2] == 'ws' else 'LOSO',
                    'Model': m[3],
                    'Accuracy': float(m[4]) * 100
                })
    return pd.DataFrame(results), feature_order, feature_names

def generar_grafico_04():
    df, feature_order, feature_names = load_data()
    df_task = df[df['Task'] == 'binary']
    models = ['rf', 'svm']
    model_labels = ['Random Forest', 'SVM']
    
    output_dir = Path('/home/pedroj/Desktop/eeg-analysis-machine-learning/output/figuras/features_individuales')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 20
    
    for model, label in zip(models, model_labels):
        df_model = df_task[df_task['Model'] == model]
        stats = df_model.groupby(['Feature', 'Eval'])['Accuracy'].agg(['mean', 'std']).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        x_pos = np.arange(len(feature_order))
        width = 0.35
        
        ws_data = stats[stats['Eval'] == 'WS']
        loso_data = stats[stats['Eval'] == 'LOSO']
        
        ws_means = [ws_data[ws_data['Feature'] == f]['mean'].values[0] if f in ws_data['Feature'].values else 0 for f in feature_order]
        ws_stds = [ws_data[ws_data['Feature'] == f]['std'].values[0] if f in ws_data['Feature'].values else 0 for f in feature_order]
        loso_means = [loso_data[loso_data['Feature'] == f]['mean'].values[0] if f in loso_data['Feature'].values else 0 for f in feature_order]
        loso_stds = [loso_data[loso_data['Feature'] == f]['std'].values[0] if f in loso_data['Feature'].values else 0 for f in feature_order]
        
        ax.bar(x_pos - width/2, ws_means, width, yerr=ws_stds, capsize=5, label='Intra-Sujeto (WS)', color=COLOR_WS, alpha=0.9, edgecolor='black', linewidth=1.5)
        ax.bar(x_pos + width/2, loso_means, width, yerr=loso_stds, capsize=5, label='Entre-Sujetos (LOSO)', color=COLOR_LOSO, alpha=0.9, edgecolor='black', linewidth=1.5)
        
        for i, (m, s) in enumerate(zip(ws_means, ws_stds)):
            if m > 0: ax.text(i - width/2, m + s + 1, f'{m:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        for i, (m, s) in enumerate(zip(loso_means, loso_stds)):
            if m > 0: ax.text(i + width/2, m + s + 1, f'{m:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
        ax.set_title(f'Accuracy por Features Individuales - Binario\n({label})', fontweight='bold', pad=20)
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names, rotation=30, ha='right')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=14, shadow=True)
        
        plt.tight_layout()
        suffix = 'RF' if model == 'rf' else 'SVM'
        filename = f"04_features_BINARIO_{suffix}.png"
        save_path = output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {save_path}")
        plt.close()

if __name__ == '__main__':
    generar_grafico_04()

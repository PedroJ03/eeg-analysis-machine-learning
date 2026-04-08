import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def generate_model_task_plot():
    output_dir = os.path.join('/home/pedroj/Desktop/eeg-analysis-machine-learning/ml_analysis/results')
    csv_path = os.path.join(output_dir, 'feature_saturation_results.csv')
    df = pd.read_csv(csv_path)

    PROGRESSIVE_STEPS = [
        {'groups': 'E',          'label': 'Stats'},
        {'groups': 'BE',         'label': '+ Bandas'},
        {'groups': 'BEW',        'label': '+ Wavelet'},
        {'groups': 'BEWR',       'label': '+ Ratios'},
        {'groups': 'BEWRH',      'label': '+ Hjorth'},
        {'groups': 'BEWRHN',     'label': '+ Entropia'},
        {'groups': 'BEWRHNZ',    'label': '+ Z-Cross'},
    ]
    step_labels = [s['label'] for s in PROGRESSIVE_STEPS]
    step_x = list(range(len(step_labels)))

    # Compute num features per step easily
    n_features_per_step = []
    for step_idx in step_x:
        subset = df[df['step'] == step_idx]
        if not subset.empty:
            n_features_per_step.append(subset['n_features'].iloc[0])
        else:
            n_features_per_step.append(0)

    # 4 panels: 2x2 grid. rows: WS vs Group. cols: RF vs SVM
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True, sharey=False)
    fig.suptitle('Saturacion de Features - Tareas Evaluadas por Modelo de ML',
                 fontsize=16, fontweight='bold', y=0.96)

    task_colors = {'binary': '#2196F3', 'extremes': '#4CAF50', '4level': '#FF5722'}
    task_names  = {'binary': 'Binario (0-1 vs 2-3)', 'extremes': 'Extremos (0 vs 3)', '4level': '4 Niveles'}

    modes = ['ws', 'group']
    mode_titles = ['Within-Subject (WS)', 'Leave-One-Subject-Out (LOSO)']
    models = ['rf', 'svm']
    model_titles = ['Random Forest', 'SVM (RBF)']

    for row_idx, eval_mode in enumerate(modes):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            ax.set_title(f"{mode_titles[row_idx]} - {model_titles[col_idx]}", fontsize=12, fontweight='bold')
            
            # Plot the baseline (random chance for 4-level and binary/extremes if needed)
            ax.axhline(0.25, color='gray', linestyle=':', alpha=0.5, label='Azar (4Lv)')
            ax.axhline(0.50, color='gray', linestyle='--', alpha=0.5, label='Azar (Bin/Ext)')

            for task in ['binary', 'extremes', '4level']:
                accs = []
                for step_idx in step_x:
                    subset = df[(df['step'] == step_idx) &
                                (df['eval'] == eval_mode) &
                                (df['model'] == model) &
                                (df['task'] == task)]
                    accs.append(subset['accuracy'].mean())

                ax.plot(step_x, accs, 'o-', color=task_colors[task],
                        label=task_names[task], linewidth=2.5, markersize=8)

                # Annotations for each dot
                for i, acc in enumerate(accs):
                    ax.annotate(f"{acc:.3f}", (step_x[i], acc),
                                textcoords="offset points", xytext=(0, 10),
                                fontsize=8, ha='center', color=task_colors[task])

            ax.set_xticks(step_x)
            if row_idx == 1:
                ax.set_xticklabels(step_labels, rotation=30, ha='right', fontsize=10)
            
            if col_idx == 0:
                ax.set_ylabel('Accuracy', fontsize=11)
            
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
            
            if row_idx == 0 and col_idx == 1:
                # Put a legend only on top right
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=10)

    # Top feature annotations
    for col_idx in range(2):
        ax = axes[0, col_idx]
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(step_x)
        ax2.set_xticklabels([f'{n}d' for n in n_features_per_step], fontsize=9, color='gray')
        ax2.set_xlabel('Dimensiones (Numero de Features)', fontsize=10, color='gray', labelpad=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, right=0.85) # leave room for title and global legend
    
    out_path = os.path.join(output_dir, 'saturation_by_task_and_model.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Grafico 2x2 de Modelos y Tareas generado exitosamente en: {out_path}")

if __name__ == '__main__':
    generate_model_task_plot()

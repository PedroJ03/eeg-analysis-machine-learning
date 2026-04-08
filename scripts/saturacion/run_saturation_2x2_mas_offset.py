"""
Gráfico de Saturación 2x2 con MAYOR offset y colores mejorados
"""

import glob
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.model_selection import KFold, StratifiedGroupKFold, LeaveOneGroupOut, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import mne
mne.set_log_level('ERROR')

from src.config import DATA_PATH
from src.data_loader import load_mat_data, get_trial_segments, get_window_epochs
from src.features import extract_features_vector
from src.models import get_classifier


PROGRESSIVE_STEPS = [
    {'groups': 'E', 'label': 'Stats', 'added': 'E'},
    {'groups': 'BE', 'label': '+ Bandas', 'added': 'B'},
    {'groups': 'BEW', 'label': '+ Wavelet', 'added': 'W'},
    {'groups': 'BEWR', 'label': '+ Ratios', 'added': 'R'},
    {'groups': 'BEWRH', 'label': '+ Hjorth', 'added': 'H'},
    {'groups': 'BEWRHN', 'label': '+ Entropia', 'added': 'N'},
    {'groups': 'BEWRHNZ', 'label': '+ Z-Cross', 'added': 'Z'},
]


def map_labels(y, task):
    if task == 'binary':
        return np.where(y <= 1, 0, 1), None
    elif task == 'extremes':
        mask = (y == 0) | (y == 3)
        y_mapped = np.where(y == 0, 0, 1)
        return y_mapped, mask
    return y, None


def evaluate_ws(X_list, y_list, g_list, config_type, task, model_type):
    all_acc = []
    for X, y, g in zip(X_list, y_list, g_list):
        y_final, mask = map_labels(y, task)
        if mask is not None:
            X_subj, y_subj, g_subj = X[mask], y_final[mask], g[mask]
        else:
            X_subj, y_subj, g_subj = X, y_final, g
        if len(np.unique(y_subj)) < 2:
            continue
        if config_type == "window":
            n_groups = len(np.unique(g_subj))
            if n_groups < 2:
                continue
            kf = StratifiedGroupKFold(n_splits=min(4, n_groups))
            split_iterator = kf.split(X_subj, y_subj, groups=g_subj)
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            split_iterator = kf.split(X_subj)
        subject_fold_acc = []
        for train_idx, test_idx in split_iterator:
            X_train, X_test = X_subj[train_idx], X_subj[test_idx]
            y_train, y_test = y_subj[train_idx], y_subj[test_idx]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            clf = get_classifier(model_type)
            if isinstance(clf, GridSearchCV):
                clf.fit(X_train, y_train, groups=g_subj[train_idx] if config_type == 'window' else None)
            else:
                clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            subject_fold_acc.append(accuracy_score(y_test, y_pred))
        if subject_fold_acc:
            all_acc.append(np.mean(subject_fold_acc))
    return np.mean(all_acc) if all_acc else 0.0


def evaluate_group(X_list, y_list, g_list, config_type, task, model_type):
    all_X, all_y, all_groups = [], [], []
    for i, (X, y, g) in enumerate(zip(X_list, y_list, g_list)):
        y_final, mask = map_labels(y, task)
        if mask is not None:
            X_subj, y_subj = X[mask], y_final[mask]
        else:
            X_subj, y_subj = X, y_final
        if len(y_subj) == 0:
            continue
        scaler = StandardScaler()
        X_subj = scaler.fit_transform(X_subj)
        all_X.append(X_subj)
        all_y.append(y_subj)
        all_groups.append(np.full(len(y_subj), i))
    if not all_X:
        return 0.0
    X_pool = np.concatenate(all_X)
    y_pool = np.concatenate(all_y)
    g_pool = np.concatenate(all_groups)
    logo = LeaveOneGroupOut()
    all_acc = []
    for train_idx, test_idx in logo.split(X_pool, y_pool, g_pool):
        X_train, X_test = X_pool[train_idx], X_pool[test_idx]
        y_train, y_test = y_pool[train_idx], y_pool[test_idx]
        clf = get_classifier(model_type)
        if isinstance(clf, GridSearchCV):
            clf = clf.estimator
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        all_acc.append(accuracy_score(y_test, y_pred))
    return np.mean(all_acc) if all_acc else 0.0


def load_and_extract(subject_paths, groups_str):
    cache = {'trial': {'X': [], 'y': [], 'g': []}, 'window': {'X': [], 'y': [], 'g': []}}
    for path in subject_paths:
        data, exp = load_mat_data(path)
        seg_t, y_t, g_t = get_trial_segments(data, exp)
        if len(y_t) > 0:
            X_t = np.array([extract_features_vector(s, groups=groups_str) for s in seg_t])
            cache['trial']['X'].append(X_t)
            cache['trial']['y'].append(y_t)
            cache['trial']['g'].append(g_t)
        seg_w, y_w, g_w = get_window_epochs(data, exp)
        if len(y_w) > 0:
            X_w = np.array([extract_features_vector(s, groups=groups_str) for s in seg_w])
            cache['window']['X'].append(X_w)
            cache['window']['y'].append(y_w)
            cache['window']['g'].append(g_w)
    return cache


def run_saturation_analysis():
    subject_paths = sorted(glob.glob(DATA_PATH))
    if not subject_paths:
        print("ERROR: No .mat files found.")
        return
    print(f"Sujetos encontrados: {len(subject_paths)}")
    configs = ['trial', 'window']
    tasks = ['binary', 'extremes', '4level']
    evals = ['ws', 'group']
    models = ['rf', 'svm']
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'feature_saturation_results.csv')
    evaluated_cache = {}
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        for _, r in df_old.iterrows():
            key = (r['groups'], r['config'], r['task'], r['eval'], r['model'])
            evaluated_cache[key] = r['accuracy']
    all_results = []
    n_features_per_step = []
    from itertools import product
    for step_idx, step in enumerate(PROGRESSIVE_STEPS):
        groups_str = step['groups']
        label = step['label']
        print(f"\nPASO {step_idx + 1}/{len(PROGRESSIVE_STEPS)}: {label}")
        combos = list(product(configs, tasks, evals, models))
        need_calc = False
        for combo in combos:
            chk_key = (groups_str,) + combo
            if chk_key not in evaluated_cache:
                need_calc = True
                break
        if not need_calc:
            print("  [CACHE] Leyendo resultados previos...")
            dummy_data, dummy_exp = load_mat_data(subject_paths[0])
            dummy_seg, dummy_y, _ = get_trial_segments(dummy_data, dummy_exp)
            n_feats = extract_features_vector(dummy_seg[0], groups=groups_str).shape[0] if len(dummy_y)>0 else 0
            n_features_per_step.append(n_feats)
            for config, task, eval_mode, model in combos:
                chk_key = (groups_str, config, task, eval_mode, model)
                acc = evaluated_cache[chk_key]
                all_results.append({'step': step_idx, 'groups': groups_str, 'label': label, 
                    'n_features': n_feats, 'config': config, 'task': task, 'eval': eval_mode,
                    'model': model, 'accuracy': acc})
            continue
        cache = load_and_extract(subject_paths, groups_str)
        n_feats = 0
        for cfg in configs:
            if cache[cfg]['X']:
                n_feats = cache[cfg]['X'][0].shape[1]
                break
        n_features_per_step.append(n_feats)
        for combo_idx, (config, task, eval_mode, model) in enumerate(combos):
            chk_key = (groups_str, config, task, eval_mode, model)
            if chk_key in evaluated_cache:
                acc = evaluated_cache[chk_key]
            else:
                X_list = cache[config]['X']
                y_list = cache[config]['y']
                g_list = cache[config]['g']
                if not X_list:
                    continue
                if eval_mode == 'ws':
                    acc = evaluate_ws(X_list, y_list, g_list, config, task, model)
                else:
                    acc = evaluate_group(X_list, y_list, g_list, config, task, model)
            all_results.append({'step': step_idx, 'groups': groups_str, 'label': label,
                'n_features': n_feats, 'config': config, 'task': task, 'eval': eval_mode,
                'model': model, 'accuracy': acc})
            pd.DataFrame(all_results).to_csv(csv_path, index=False)
    df = pd.DataFrame(all_results)
    generate_2x2_mas_offset(df, n_features_per_step, output_dir)
    print("\n✅ Gráficos generados!")


def generate_2x2_mas_offset(df, n_features_per_step, output_dir):
    """Generate 2x2 layout with MÁS offset y colores mejorados"""
    step_labels = [s['label'] for s in PROGRESSIVE_STEPS]
    step_x = np.array(range(len(step_labels)))
    
    # MAYOR OFFSET para más separación
    offset = 0.22  # Aumentado de 0.12 a 0.22
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True)
    fig.suptitle('Análisis de Saturación de Features\n(Mayor separación y colores mejorados)',
                 fontsize=15, fontweight='bold', y=0.98)
    
    # COLORES MEJORADOS - Más distinguibles
    task_colors = {
        'binary': '#0066CC',   # Azul oscuro
        'extremes': '#009900', # Verde oscuro
        '4level': '#CC3300'    # Rojo/naranja oscuro
    }
    
    task_names = {'binary': 'Binario', 'extremes': 'Extremos', '4level': '4 Clases'}
    
    configs_2x2 = [
        ('ws', 'rf', 'Intra-Sujeto (WS) - Random Forest', 0, 0),
        ('ws', 'svm', 'Intra-Sujeto (WS) - SVM', 0, 1),
        ('group', 'rf', 'Entre-Sujetos (LOSO) - Random Forest', 1, 0),
        ('group', 'svm', 'Entre-Sujetos (LOSO) - SVM', 1, 1),
    ]
    
    for eval_mode, model, title, row, col in configs_2x2:
        ax = axes[row, col]
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        # Tarea 1: Binario (desplazado a la izquierda) - Línea sólida
        accs_bin = []
        stds_bin = []
        for step_idx in step_x:
            subset = df[(df['step'] == step_idx) & (df['eval'] == eval_mode) & (df['model'] == model) & (df['task'] == 'binary')]
            accs_bin.append(subset['accuracy'].mean())
            stds_bin.append(subset['accuracy'].std())
        
        x_bin = step_x - offset
        ax.errorbar(x_bin, accs_bin, yerr=stds_bin, fmt='o-', color=task_colors['binary'],
                   label='Binario', linewidth=2.5, markersize=10,
                   capsize=6, capthick=2.5, elinewidth=2, alpha=0.95,
                   markerfacecolor='white', markeredgewidth=2)
        
        # Tarea 2: Extremos (sin desplazamiento - en el centro) - Línea punteada
        accs_ext = []
        stds_ext = []
        for step_idx in step_x:
            subset = df[(df['step'] == step_idx) & (df['eval'] == eval_mode) & (df['model'] == model) & (df['task'] == 'extremes')]
            accs_ext.append(subset['accuracy'].mean())
            stds_ext.append(subset['accuracy'].std())
        
        ax.errorbar(step_x, accs_ext, yerr=stds_ext, fmt='s--', color=task_colors['extremes'],
                   label='Extremos', linewidth=2.5, markersize=10,
                   capsize=6, capthick=2.5, elinewidth=2, alpha=0.95,
                   markerfacecolor='white', markeredgewidth=2)
        
        # Tarea 3: 4 Clases (desplazado a la derecha) - Línea punto-raya
        accs_4l = []
        stds_4l = []
        for step_idx in step_x:
            subset = df[(df['step'] == step_idx) & (df['eval'] == eval_mode) & (df['model'] == model) & (df['task'] == '4level')]
            accs_4l.append(subset['accuracy'].mean())
            stds_4l.append(subset['accuracy'].std())
        
        x_4l = step_x + offset
        ax.errorbar(x_4l, accs_4l, yerr=stds_4l, fmt='^-.', color=task_colors['4level'],
                   label='4 Clases', linewidth=2.5, markersize=10,
                   capsize=6, capthick=2.5, elinewidth=2, alpha=0.95,
                   markerfacecolor='white', markeredgewidth=2)
        
        # Anotaciones con mejor contraste
        for i in range(len(step_x)):
            ax.annotate(f'{accs_bin[i]:.2f}', (x_bin[i], accs_bin[i]),
                       textcoords='offset points', xytext=(0, 15),
                       fontsize=8, ha='center', color=task_colors['binary'], 
                       fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', 
                       facecolor='white', edgecolor=task_colors['binary'], alpha=0.8))
            
            ax.annotate(f'{accs_ext[i]:.2f}', (step_x[i], accs_ext[i]),
                       textcoords='offset points', xytext=(0, 15),
                       fontsize=8, ha='center', color=task_colors['extremes'], 
                       fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', 
                       facecolor='white', edgecolor=task_colors['extremes'], alpha=0.8))
            
            ax.annotate(f'{accs_4l[i]:.2f}', (x_4l[i], accs_4l[i]),
                       textcoords='offset points', xytext=(0, 15),
                       fontsize=8, ha='center', color=task_colors['4level'], 
                       fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', 
                       facecolor='white', edgecolor=task_colors['4level'], alpha=0.8))
        
        # Configurar ejes
        ax.set_xticks(step_x)
        ax.set_xticklabels(step_labels, rotation=40, ha='right', fontsize=9)
        if col == 0:
            ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        if row == 1:
            ax.set_xlabel('Grupos de Features (acumulativos)', fontsize=11, fontweight='bold')
        
        # Leyenda más grande y clara
        ax.legend(loc='lower right', fontsize=10, framealpha=0.95, 
                 edgecolor='black', fancybox=True, shadow=True)
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        ax.set_ylim(0.15, 0.85)
        
        # Fondo blanco para mejor contraste
        ax.set_facecolor('white')
        
        # Eje superior con dimensiones
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(step_x)
        ax2.set_xticklabels([f'{n}d' for n in n_features_per_step], fontsize=8, color='gray')
        ax2.set_xlabel('Dimensiones', fontsize=9, color='gray', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    path = os.path.join(output_dir, 'saturation_2x2_mas_offset.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Gráfica guardada: {path}")
    plt.close(fig)


if __name__ == '__main__':
    run_saturation_analysis()

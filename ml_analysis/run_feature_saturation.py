"""
Feature Saturation Analysis
============================
Ejecuta experimentos incrementales, agregando grupos de features de forma
progresiva para observar si la precisión se estanca (satura) o sigue
aumentando con cada nuevo grupo.

Genera gráficas aptas para incluir en un informe de tesis.
"""

import glob
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from itertools import product
from sklearn.model_selection import (
    KFold, GroupKFold, StratifiedGroupKFold,
    LeaveOneGroupOut, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import mne
mne.set_log_level('ERROR')

from src.config import DATA_PATH
from src.data_loader import load_mat_data, get_trial_segments, get_window_epochs
from src.features import extract_features_vector
from src.models import get_classifier


# ─── Feature Group Definitions ─────────────�PROGRESSIVE_STEPS = [
    {'groups': 'E',       'label': 'Stats',                    'added': 'E (Estadísticas)'},
    {'groups': 'BE',      'label': '+ Bandas (sin γ)',         'added': 'B (Bands δ, θ, α, β)'},
    {'groups': 'BgE',     'label': '+ Gamma',                  'added': 'γ (Gamma Band)'},
    {'groups': 'BgEW',    'label': '+ Wavelet',                'added': 'W (Wavelet)'},
    {'groups': 'BgEWR',   'label': '+ Ratios',                 'added': 'R (Ratios Espectrales)'},
    {'groups': 'BgEWRH',  'label': '+ Hjorth',                 'added': 'H (Hjorth)'},
    {'groups': 'BgEWRHN', 'label': '+ Entropía',               'added': 'N (Sample/Spectral Ent)'},
    {'groups': 'BgEWRHNZ','label': '+ Z-Cross',                'added': 'Z (Zero Crossing)'},
]

# Feature count per group (for annotations)
GROUP_FEATURE_COUNT = {
    'E': 8,   # mean, std, max, min, median, skew, kurtosis, line_length
    'B': 4,   # delta, theta, alpha, beta
    'Bg': 1,  # extra feature for gamma (1 added to cumulative)
    'W': 5,   # 5 wavelet energy coefficients
    'R': 3,   # θ/α, β/θ, γ/α
    'H': 3,   # activity, mobility, complexity
    'N': 2,   # spectral entropy, shannon entropy
    'Z': 1,   # zero-crossing rate
}

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

        if len(y_subj) == 0: continue

        scaler = StandardScaler()
        X_subj = scaler.fit_transform(X_subj)
        all_X.append(X_subj)
        all_y.append(y_subj)
        all_groups.append(np.full(len(y_subj), i))

    if not all_X: return 0.0

    X_pool = np.concatenate(all_X)
    y_pool = np.concatenate(all_y)
    g_pool = np.concatenate(all_groups)
    logo = LeaveOneGroupOut()
    all_acc = []

    for train_idx, test_idx in logo.split(X_pool, y_pool, g_pool):
        X_train, X_test = X_pool[train_idx], X_pool[test_idx]
        y_train, y_test = y_pool[train_idx], y_pool[test_idx]

        clf = get_classifier(model_type)
        if isinstance(clf, GridSearchCV): clf = clf.estimator
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
    tasks   = ['binary', 'extremes', '4level']
    evals   = ['ws', 'group']
    models  = ['rf', 'svm']

    # Load existing CSV if it exists to SKIP already computed steps
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'feature_saturation_results.csv')
    
    existing_records = []
    evaluated_keys = set()
    if os.path.exists(csv_path):
        try:
            df_old = pd.read_csv(csv_path)
            for _, row in df_old.iterrows():
                # Store all old info but we might need to overwrite step ID since PROGRESSIVE_STEPS index changed
                key = (row['groups'], row['config'], row['task'], row['eval'], row['model'])
                evaluated_keys.add(key)
                existing_records.append(row.to_dict())
            print(f"Cargado cache con {len(evaluated_keys)} experimentos previos.")
        except Exception as e:
            print(f"Error cargando CSV previo: {e}. Empezando de cero.")

    all_results = []
    n_features_per_step = []

    for step_idx, step in enumerate(PROGRESSIVE_STEPS):
        groups_str = step['groups']
        label = step['label']

        print(f"\n{'='*70}")
        print(f"  PASO {step_idx + 1}/{len(PROGRESSIVE_STEPS)}: {label}")
        print(f"  Groups: '{groups_str}' | Nuevo: {step['added']}")
        print(f"{'='*70}")

        # Quickly check if ALL combinations for this group are already in cache
        need_to_run = False
        combos = list(product(configs, tasks, evals, models))
        for combo in combos:
            chk_key = (groups_str,) + combo
            if chk_key not in evaluated_keys:
                need_to_run = True
                break
                
        if not need_to_run:
            print("  --- Paso completado previamente, cargando del cache directo --- ")
            n_feats = [r['n_features'] for r in existing_records if r['groups'] == groups_str][0]
            n_features_per_step.append(n_feats)
            for r in existing_records:
                if r['groups'] == groups_str:
                    r_copy = r.copy()
                    r_copy['step'] = step_idx # update step index if changed
                    all_results.append(r_copy)
            continue

        # Extract features
        cache = load_and_extract(subject_paths, groups_str)

        n_feats = 0
        for cfg in configs:
            if cache[cfg]['X']:
                n_feats = cache[cfg]['X'][0].shape[1]
                break
        n_features_per_step.append(n_feats)
        print(f"  Dimensión del vector de features: {n_feats}")

        for combo_idx, (config, task, eval_mode, model) in enumerate(combos):
            chk_key = (groups_str, config, task, eval_mode, model)
            print(f"  [{combo_idx+1:02d}/{len(combos)}] {config:<7} {task:<9} {eval_mode:<6} {model:<4} ...", end="", flush=True)

            if chk_key in evaluated_keys:
                old_acc = [r['accuracy'] for r in existing_records if 
                            r['groups'] == groups_str and r['config'] == config and 
                            r['task'] == task and r['eval'] == eval_mode and r['model'] == model][0]
                all_results.append({
                    'step': step_idx, 'groups': groups_str, 'label': label, 'added': step['added'],
                    'n_features': n_feats, 'config': config, 'task': task, 'eval': eval_mode,
                    'model': model, 'accuracy': old_acc
                })
                print(f" [CACHED] ACC={old_acc:.4f}")
                continue

            X_list, y_list, g_list = cache[config]['X'], cache[config]['y'], cache[config]['g']
            if not X_list:
                print(" [ERROR] Sin datos")
                continue

            acc = evaluate_ws(X_list, y_list, g_list, config, task, model) if eval_mode == 'ws' else evaluate_group(X_list, y_list, g_list, config, task, model)

            all_results.append({
                'step': step_idx, 'groups': groups_str, 'label': label, 'added': step['added'],
                'n_features': n_feats, 'config': config, 'task': task, 'eval': eval_mode,
                'model': model, 'accuracy': acc
            })
            print(f" ACC={acc:.4f}")
            
            # Autoguardado temporal
            pd.DataFrame(all_results).to_csv(csv_path, index=False)

    df = pd.DataFrame(all_results)
              'accuracy': acc
            })
            print(f"  ACC={acc:.4f}")

    df = pd.DataFrame(all_results)

    # ─── Save raw results CSV ────────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'feature_saturation_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResultados guardados en: {csv_path}")

    # ─── Generate Plots ──────────────────────────────────────────────────
    generate_plots(df, n_features_per_step, output_dir)

    # ─── Print Summary Table ─────────────────────────────────────────────
    print_summary_table(df)


def generate_plots(df, n_features_per_step, output_dir):
    """Generate publication-quality saturation plots."""

    step_labels = [s['label'] for s in PROGRESSIVE_STEPS]
    step_x = list(range(len(step_labels)))

    # ─── Plot 1: Accuracy vs Feature Step (one line per task, averaged over models/configs) ───
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle('Análisis de Saturación de Features — Accuracy vs Grupos de Features',
                 fontsize=14, fontweight='bold', y=1.02)

    task_colors = {'binary': '#2196F3', 'extremes': '#4CAF50', '4level': '#FF5722'}
    task_names  = {'binary': 'Binario (0-1 vs 2-3)', 'extremes': 'Extremos (0 vs 3)', '4level': '4 Niveles'}

    for ax_idx, eval_mode in enumerate(['ws', 'group']):
        ax = axes[ax_idx]
        eval_title = 'Within-Subject' if eval_mode == 'ws' else 'Leave-One-Subject-Out'
        ax.set_title(eval_title, fontsize=12, fontweight='bold')

        for task in ['binary', 'extremes', '4level']:
            accs = []
            for step_idx in step_x:
                subset = df[(df['step'] == step_idx) &
                            (df['eval'] == eval_mode) &
                            (df['task'] == task)]
                accs.append(subset['accuracy'].mean())

            ax.plot(step_x, accs, 'o-', color=task_colors[task],
                    label=task_names[task], linewidth=2, markersize=8)

            # Annotate accuracy values
            for i, acc in enumerate(accs):
                ax.annotate(f'{acc:.3f}', (step_x[i], acc),
                            textcoords='offset points', xytext=(0, 10),
                            fontsize=8, ha='center', color=task_colors[task])

        ax.set_xticks(step_x)
        ax.set_xticklabels(step_labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Accuracy' if ax_idx == 0 else '', fontsize=11)
        ax.set_xlabel('Grupos de Features (acumulativos)', fontsize=11)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    # Add feature count annotations at top
    for ax in axes:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(step_x)
        ax2.set_xticklabels([f'{n}d' for n in n_features_per_step], fontsize=8, color='gray')
        ax2.set_xlabel('Dimensiones', fontsize=9, color='gray')

    plt.tight_layout()
    path1 = os.path.join(output_dir, 'saturation_by_task.png')
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"Gráfica guardada: {path1}")
    plt.close(fig)

    # ─── Plot 2: Accuracy vs Feature Step (one line per model, split by eval) ───
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle('Análisis de Saturación — Comparación por Modelo',
                 fontsize=14, fontweight='bold', y=1.02)

    model_colors = {'rf': '#9C27B0', 'svm': '#FF9800'}
    model_names  = {'rf': 'Random Forest', 'svm': 'SVM (RBF)'}

    for ax_idx, eval_mode in enumerate(['ws', 'group']):
        ax = axes[ax_idx]
        eval_title = 'Within-Subject' if eval_mode == 'ws' else 'Leave-One-Subject-Out'
        ax.set_title(eval_title, fontsize=12, fontweight='bold')

        for model in ['rf', 'svm']:
            accs = []
            for step_idx in step_x:
                subset = df[(df['step'] == step_idx) &
                            (df['eval'] == eval_mode) &
                            (df['model'] == model)]
                accs.append(subset['accuracy'].mean())

            ax.plot(step_x, accs, 's-', color=model_colors[model],
                    label=model_names[model], linewidth=2, markersize=8)

            for i, acc in enumerate(accs):
                ax.annotate(f'{acc:.3f}', (step_x[i], acc),
                            textcoords='offset points', xytext=(0, 10),
                            fontsize=8, ha='center', color=model_colors[model])

        ax.set_xticks(step_x)
        ax.set_xticklabels(step_labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Accuracy' if ax_idx == 0 else '', fontsize=11)
        ax.set_xlabel('Grupos de Features (acumulativos)', fontsize=11)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    for ax in axes:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(step_x)
        ax2.set_xticklabels([f'{n}d' for n in n_features_per_step], fontsize=8, color='gray')
        ax2.set_xlabel('Dimensiones', fontsize=9, color='gray')

    plt.tight_layout()
    path2 = os.path.join(output_dir, 'saturation_by_model.png')
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"Gráfica guardada: {path2}")
    plt.close(fig)

    # ─── Plot 3: Delta Accuracy (marginal gain per step) ───
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('Ganancia Marginal de Accuracy por Grupo de Features Agregado',
                 fontsize=14, fontweight='bold')

    bar_width = 0.15
    task_list = ['binary', 'extremes', '4level']
    eval_list = ['ws', 'group']

    combo_labels = []
    combo_colors = []
    for task in task_list:
        for eval_mode in eval_list:
            short_eval = 'WS' if eval_mode == 'ws' else 'LOSO'
            combo_labels.append(f'{task_names[task]} ({short_eval})')
            base = task_colors[task]
            # Lighter shade for group
            combo_colors.append(base if eval_mode == 'ws' else base + '80')

    # Steps 1-4 have deltas (step 0 is baseline)
    delta_steps = step_x[1:]
    delta_labels = [PROGRESSIVE_STEPS[i]['added'] for i in delta_steps]

    for combo_idx, (task, eval_mode) in enumerate(product(task_list, eval_list)):
        deltas = []
        for step_idx in delta_steps:
            prev_subset = df[(df['step'] == step_idx - 1) &
                             (df['eval'] == eval_mode) &
                             (df['task'] == task)]
            curr_subset = df[(df['step'] == step_idx) &
                             (df['eval'] == eval_mode) &
                             (df['task'] == task)]
            delta = curr_subset['accuracy'].mean() - prev_subset['accuracy'].mean()
            deltas.append(delta * 100)  # Convert to percentage points

        x_pos = np.arange(len(delta_steps)) + combo_idx * bar_width
        bars = ax.bar(x_pos, deltas, bar_width, label=combo_labels[combo_idx],
                      color=combo_colors[combo_idx], edgecolor='white', linewidth=0.5)

        for bar, d in zip(bars, deltas):
            y_offset = 0.1 if d >= 0 else -0.3
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset,
                    f'{d:+.1f}', ha='center', va='bottom', fontsize=7)

    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    center_offset = (len(combo_labels) - 1) * bar_width / 2
    ax.set_xticks(np.arange(len(delta_steps)) + center_offset)
    ax.set_xticklabels(delta_labels, fontsize=9)
    ax.set_ylabel('Δ Accuracy (puntos porcentuales)', fontsize=11)
    ax.set_xlabel('Feature Agregada', fontsize=11)
    ax.legend(loc='best', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path3 = os.path.join(output_dir, 'saturation_marginal_gain.png')
    fig.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"Gráfica guardada: {path3}")
    plt.close(fig)


def print_summary_table(df):
    """Print a comprehensive summary table."""
    print("\n" + "=" * 90)
    print(" TABLA RESUMEN — FEATURE SATURATION ANALYSIS ".center(90, "="))
    print("=" * 90)

    # Pivot: rows = step label, columns = task+eval, values = mean accuracy
    task_names = {'binary': 'Bin', 'extremes': 'Ext', '4level': '4Lv'}

    header = f"{'Step':<16} | {'N':<3}"
    for task in ['binary', 'extremes', '4level']:
        for eval_mode in ['ws', 'group']:
            short_eval = 'WS' if eval_mode == 'ws' else 'LOSO'
            header += f" | {task_names[task]}-{short_eval:>4}"
    header += f" | {'Prom':>6}"
    print(header)
    print("-" * 90)

    for step_idx, step in enumerate(PROGRESSIVE_STEPS):
        step_df = df[df['step'] == step_idx]
        n_feats = step_df['n_features'].iloc[0] if len(step_df) > 0 else '?'
        row = f"{step['label']:<16} | {n_feats:<3}"

        for task in ['binary', 'extremes', '4level']:
            for eval_mode in ['ws', 'group']:
                subset = step_df[(step_df['task'] == task) & (step_df['eval'] == eval_mode)]
                acc = subset['accuracy'].mean() if len(subset) > 0 else 0
                row += f" | {acc:.4f}  "

        overall = step_df['accuracy'].mean()
        row += f" | {overall:.4f}"
        print(row)

    print("=" * 90)


if __name__ == '__main__':
    run_saturation_analysis()

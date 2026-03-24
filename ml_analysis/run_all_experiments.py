import argparse
import glob
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split, KFold, GroupKFold, StratifiedGroupKFold, LeaveOneGroupOut, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import mne
mne.set_log_level('ERROR')

from src.config import DATA_PATH
from src.data_loader import load_mat_data, get_trial_segments, get_window_epochs
from src.features import extract_features_vector
from src.models import get_classifier


def get_feature_description(features_str):
    """Devuelve descripción legible de las features utilizadas."""
    descriptions = {
        'B': 'Band Power (δ, θ, α, β) - Sin Gamma',
        'Bg': 'Band Power (δ, θ, α, β, γ) - Con Gamma',
        'E': 'Estadísticas Temporales (media, std, skewness, kurtosis, line length)',
        'W': 'Energías Wavelet (Daubechies-4, nivel 4)'
    }
    
    feature_list = features_str.split(',')
    desc_parts = []
    for f in feature_list:
        f = f.strip()
        if f in descriptions:
            desc_parts.append(f"  • {f}: {descriptions[f]}")
    return "\n".join(desc_parts)


def get_feature_combination_name(features_str):
    """Genera nombre legible de la combinación de features."""
    feature_map = {
        'B': 'Bandas',
        'Bg': 'Bandas+γ',
        'E': 'Estadísticas',
        'W': 'Wavelet'
    }
    feature_list = [f.strip() for f in features_str.split(',')]
    names = [feature_map.get(f, f) for f in feature_list]
    return ' + '.join(names)


def print_config_summary(features_str):
    """Imprime resumen descriptivo de la configuración de features."""
    combo_name = get_feature_combination_name(features_str)
    print("\n" + "="*76)
    print(" CONFIGURACIÓN DE FEATURES ".center(76, "="))
    print("="*76)
    print(f"\nCombinación: {combo_name}")
    print(f"Código: {features_str}")
    print(f"\nDescripción detallada:")
    print(get_feature_description(features_str))
    print("="*76 + "\n")

def map_labels(y, task):
    if task == 'binary':
        # 0-1 -> 0, 2-3 -> 1
        return np.where(y <= 1, 0, 1), None
    elif task == 'extremes':
        # only 0 and 3
        mask = (y == 0) | (y == 3)
        y_mapped = np.where(y == 0, 0, 1) # map 0->0, 3->1
        return y_mapped, mask
    return y, None # 4level returns as is

def evaluate_ws(X_list, y_list, g_list, config_type, task, model_type):
    all_acc = []
    
    for X, y, g in zip(X_list, y_list, g_list):
        y_final, mask = map_labels(y, task)
        if mask is not None:
            X_subj = X[mask]
            y_subj = y_final[mask]
            g_subj = g[mask]
        else:
            X_subj = X
            y_subj = y_final
            g_subj = g
            
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
    all_X = []
    all_y = []
    all_groups = []
    
    for i, (X, y, g) in enumerate(zip(X_list, y_list, g_list)):
        y_final, mask = map_labels(y, task)
        if mask is not None:
             X_subj = X[mask]
             y_subj = y_final[mask]
        else:
             X_subj = X
             y_subj = y_final
             
        if len(y_subj) == 0: continue
        
        # --- Modificacion 1: Normalización Intra-Sujeto ---
        # Scaling each subject alone guarantees immunity to absolute power baselines.
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
        g_train = g_pool[train_idx]
        
        # StandardScaler is removed here because X_train and X_test are already globally normalized
        
        clf = get_classifier(model_type)
        if isinstance(clf, GridSearchCV):
            # Bypass slow GridSearch for massive pooled group data to speed up drastically
            clf = clf.estimator
            clf.fit(X_train, y_train)
        else:
            clf.fit(X_train, y_train)
            
        y_pred = clf.predict(X_test)
        all_acc.append(accuracy_score(y_test, y_pred))
        
    return np.mean(all_acc) if all_acc else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="BgWE", help="Feature groups to extract")
    args = parser.parse_args()
    
    subject_paths = sorted(glob.glob(DATA_PATH))
    if not subject_paths:
        print("No subjects found.")
        return
        
    print_config_summary(args.features)
    print(f"--- Extrayendo descriptores (Features: {args.features}) ---")
    print("Esto puede tardar un momento, pero se cacheará para todas las configuraciones...")
    
    # Precompute features to save massive amounts of time
    data_cache = {'trial': {'X':[], 'y':[], 'g':[]}, 'window': {'X':[], 'y':[], 'g':[]}}
    
    for path in subject_paths:
        data, exp = load_mat_data(path)
        
        # Trial config
        seg_t, y_t, g_t = get_trial_segments(data, exp)
        if len(y_t) > 0:
            X_t = np.array([extract_features_vector(s, groups=args.features) for s in seg_t])
            data_cache['trial']['X'].append(X_t)
            data_cache['trial']['y'].append(y_t)
            data_cache['trial']['g'].append(g_t)
            
        # Window config
        seg_w, y_w, g_w = get_window_epochs(data, exp)
        if len(y_w) > 0:
            X_w = np.array([extract_features_vector(s, groups=args.features) for s in seg_w])
            data_cache['window']['X'].append(X_w)
            data_cache['window']['y'].append(y_w)
            data_cache['window']['g'].append(g_w)
            
    configs = ['trial', 'window']
    tasks = ['4level', 'binary', 'extremes']
    evals = ['ws', 'group']
    models = ['rf', 'svm']
    
    results = []
    
    print("\n--- Ejecutando Experimentos Combinados ---")
    total_runs = len(configs) * len(tasks) * len(evals) * len(models)
    current = 0
    
    for config, task, eval_mode, model in product(configs, tasks, evals, models):
        current += 1
        print(f"[{current:02d}/{total_runs}] Config: {config:<6} | Task: {task:<8} | Eval: {eval_mode:<5} | Model: {model:<3} ...", end="", flush=True)
        
        X_list = data_cache[config]['X']
        y_list = data_cache[config]['y']
        g_list = data_cache[config]['g']
        
        if not X_list:
            print(" Skipping (No data)")
            continue
            
        if eval_mode == 'ws':
            acc = evaluate_ws(X_list, y_list, g_list, config, task, model)
        else:
            acc = evaluate_group(X_list, y_list, g_list, config, task, model)
            
        results.append({
            'Config': config,
            'Task': task,
            'Eval': eval_mode,
            'Model': model,
            'Accuracy': acc
        })
        print(f" ACC: {acc:.4f}")
        
    # Print Results
    combo_name = get_feature_combination_name(args.features)
    print("\n" + "="*100)
    print(f" RESULTADOS: {combo_name} ".center(100, "="))
    print("="*100)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Print detailed table
    print(f"\n{'Config':<8} | {'Tarea':<10} | {'Eval':<5} | {'Modelo':<5} | {'Accuracy':<10} | {'Descripción'}")
    print("-" * 100)
    
    # Sort: by Config, then by Eval, then by Accuracy (descending)
    df_sorted = df.sort_values(['Config', 'Eval', 'Accuracy'], ascending=[True, True, False])
    
    task_desc = {
        '4level': '4 clases (0,1,2,3)',
        'binary': 'Binario (0-1 vs 2-3)',
        'extremes': 'Extremos (0 vs 3)'
    }
    eval_desc = {
        'ws': 'Within-Subject',
        'group': 'Leave-One-Subject-Out'
    }
    
    for _, r in df_sorted.iterrows():
        desc = f"{task_desc.get(r['Task'], r['Task'])} | {eval_desc.get(r['Eval'], r['Eval'])}"
        print(f"{r['Config']:<8} | {r['Task']:<10} | {r['Eval']:<5} | {r['Model']:<5} | {r['Accuracy']:.4f}     | {desc}")
    
    print("="*100)
    
    # Print summary by best results
    print("\n" + "-"*100)
    print(" TOP 10 MEJORES RESULTADOS ".center(100, "-"))
    print("-"*100)
    df_top = df.sort_values('Accuracy', ascending=False).head(10)
    print(f"{'Rank':<6} | {'Config':<8} | {'Tarea':<10} | {'Eval':<5} | {'Modelo':<5} | {'Accuracy':<10}")
    print("-" * 100)
    for i, (_, r) in enumerate(df_top.iterrows(), 1):
        print(f"{i:<6} | {r['Config']:<8} | {r['Task']:<10} | {r['Eval']:<5} | {r['Model']:<5} | {r['Accuracy']:.4f}")
    print("-"*100)
    
    # Print summary statistics
    print("\n" + "-"*100)
    print(" ESTADÍSTICAS POR GRUPO ".center(100, "-"))
    print("-"*100)
    
    # By Config
    print("\nPor Configuración (promedio):")
    config_stats = df.groupby('Config')['Accuracy'].agg(['mean', 'std', 'max']).round(4)
    print(config_stats.to_string())
    
    # By Task
    print("\nPor Tarea (promedio):")
    task_stats = df.groupby('Task')['Accuracy'].agg(['mean', 'std', 'max']).round(4)
    print(task_stats.to_string())
    
    # By Eval
    print("\nPor Modo de Evaluación (promedio):")
    eval_stats = df.groupby('Eval')['Accuracy'].agg(['mean', 'std', 'max']).round(4)
    print(eval_stats.to_string())
    
    # By Model
    print("\nPor Modelo (promedio):")
    model_stats = df.groupby('Model')['Accuracy'].agg(['mean', 'std', 'max']).round(4)
    print(model_stats.to_string())
    
    print("\n" + "="*100)
    print(f" RESUMEN EJECUTIVO: {combo_name} ".center(100, "="))
    print("="*100)
    print(f"• Mejor accuracy: {df['Accuracy'].max():.4f}")
    print(f"• Accuracy promedio: {df['Accuracy'].mean():.4f}")
    print(f"• Accuracy mínima: {df['Accuracy'].min():.4f}")
    print(f"• Total de experimentos: {len(df)}")
    print("="*100)

if __name__ == "__main__":
    main()

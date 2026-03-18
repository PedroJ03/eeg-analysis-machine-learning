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
        
    # Print Table
    print("\n" + "="*76)
    print(f" TABLA DE RESULTADOS OBTENIDOS (Features: {args.features})".center(76))
    print("="*76)
    print(f"{'Configuración':<15} | {'Tarea':<10} | {'Evaluación':<10} | {'Modelo':<8} | {'Exactitud (Acc)':<15}")
    print("-" * 76)
    
    # Sort results for better display (by configuration, then by best accuracy)
    results.sort(key=lambda x: (x['Config'], -x['Accuracy']))
    
    for r in results:
        print(f"{r['Config']:<15} | {r['Task']:<10} | {r['Eval']:<10} | {r['Model']:<8} | {r['Accuracy']:.4f}")
    
    print("="*76)

if __name__ == "__main__":
    main()

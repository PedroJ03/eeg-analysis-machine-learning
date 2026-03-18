import argparse
import glob
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GroupKFold, StratifiedGroupKFold, GridSearchCV, LeaveOneGroupOut
from src.config import DATA_PATH
from src.data_loader import load_mat_data, get_trial_segments, get_window_epochs
from src.features import extract_features_vector
from src.models import get_classifier, evaluate_model, format_results

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

def run_ws(paths, args):
    all_acc = []
    all_y_true = []
    all_y_pred = []
    
    print(f"Running Within-Subject evaluation for {len(paths)} subjects...")
    
    for path in paths:
        data, exp = load_mat_data(path)
        
        if args.config == "trial":
            segments, y, g = get_trial_segments(data, exp)
        else:
            segments, y, g = get_window_epochs(data, exp)
            
        if len(y) == 0: continue
        
        # Mapping
        y_final, mask = map_labels(y, args.task)
        if mask is not None:
            segments = [segments[idx_mask] for idx_mask in range(len(mask)) if mask[idx_mask]]
            y_final = y_final[mask]
            g = g[mask]
            
        if len(np.unique(y_final)) < 2: continue
        
        X = np.array([extract_features_vector(s, groups=args.features) for s in segments])
        
        if args.config == "window":
            n_groups = len(np.unique(g))
            kf = GroupKFold(n_splits=min(4, n_groups))
            split_iterator = kf.split(X, y_final, groups=g)
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            split_iterator = kf.split(X)
            
        from sklearn.preprocessing import StandardScaler
        
        subject_fold_acc = []
        for train_idx, test_idx in split_iterator:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_final[train_idx], y_final[test_idx]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            clf = get_classifier(args.model)
            
            # GridSearchCV needs groups to be passed to fit if using StratifiedGroupKFold
            if isinstance(clf, GridSearchCV):
                clf.fit(X_train, y_train, groups=g[train_idx])
            else:
                clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            
            subject_fold_acc.append(accuracy_score(y_test, y_pred))
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            
        mean_acc = np.mean(subject_fold_acc)
        all_acc.append(mean_acc)
        print(f"Subject {os.path.basename(path)}: {mean_acc:.4f}")
        
    global_results = evaluate_model(all_y_true, all_y_pred)
    format_results(global_results, title=f"WS Global Results ({args.task}, {args.model}, feats={args.features})")
    print(f"Mean WS Accuracy: {np.mean(all_acc):.4f}")

from sklearn.metrics import accuracy_score

def run_group(paths, args):
    all_X = []
    all_y = []
    all_groups = []
    
    print(f"Pooling data from {len(paths)} subjects for Group analysis...")
    
    from sklearn.preprocessing import StandardScaler
    
    for i, path in enumerate(paths):
        data, exp = load_mat_data(path)
        if args.config == "trial":
            segments, y, g = get_trial_segments(data, exp)
        else:
            segments, y, g = get_window_epochs(data, exp)
            
        if len(y) == 0: continue
        
        y_final, mask = map_labels(y, args.task)
        if mask is not None:
            segments = [segments[idx_mask] for idx_mask in range(len(mask)) if mask[idx_mask]]
            y_final = y_final[mask]
            
        X = np.array([extract_features_vector(s, groups=args.features) for s in segments])
        
        # --- Modificacion 1: Normalización Intra-Sujeto ---
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        all_X.append(X)
        all_y.append(y_final)
        # Keep track of which windows belong to which subject
        all_groups.append(np.full(len(y_final), i))
        
    X_pool = np.concatenate(all_X)
    y_pool = np.concatenate(all_y)
    g_pool = np.concatenate(all_groups)
    
    logo = LeaveOneGroupOut()
    
    all_acc = []
    all_y_true = []
    all_y_pred = []
    
    for train_idx, test_idx in logo.split(X_pool, y_pool, g_pool):
        X_train, X_test = X_pool[train_idx], X_pool[test_idx]
        y_train, y_test = y_pool[train_idx], y_pool[test_idx]
        g_train, g_test = g_pool[train_idx], g_pool[test_idx]
        
        clf = get_classifier(args.model)
        if isinstance(clf, GridSearchCV):
            clf.fit(X_train, y_train, groups=g_train)
        else:
            clf.fit(X_train, y_train)
            
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        all_acc.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        print(f"Subject Fold {g_test[0]}: {acc:.4f}")
        
    global_results = evaluate_model(all_y_true, all_y_pred)
    format_results(global_results, title=f"Group Analysis Results ({args.task}, {args.model}, feats={args.features})")
    print(f"Mean LOG Accuracy: {np.mean(all_acc):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Unified N-Back Model Training")
    parser.add_argument("--config", type=str, choices=["trial", "window"], default="trial",
                        help="Processing mode: 'trial' (segment per experiment trial) or 'window' (sliding windows via MNE)")
    parser.add_argument("--model", type=str, choices=['rf', 'svm'], default='rf', help="Classifier model")
    parser.add_argument("--task", type=str, choices=['4level', 'binary', 'extremes'], default='binary', help="Classification task")
    parser.add_argument("--eval", type=str, choices=['ws', 'group'], default='ws', help="Evaluation mode (Within-Subject or Group)")
    parser.add_argument("--features", type=str, default='BEW', help="Feature groups: B (bands), Bg (bands+gamma), E (stats), W (wavelet). Ex: 'BEW'")
    
    args = parser.parse_args()
    
    subject_paths = sorted(glob.glob(DATA_PATH))
    if not subject_paths:
        print(f"Error: No .mat files found in {DATA_PATH}")
        return
        
    if args.eval == 'ws':
        run_ws(subject_paths, args)
    else:
        run_group(subject_paths, args)

if __name__ == "__main__":
    main()

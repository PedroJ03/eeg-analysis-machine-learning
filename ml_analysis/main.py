import argparse
import glob
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from .src.config import DATA_PATH
from .src.data_loader import load_mat_data, get_trial_segments, get_window_epochs
from .src.features import extract_features_vector
from .src.models import get_classifier, evaluate_model, format_results

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
            segments = [segments[i] for i in range(len(mask)) if mask[i]]
            y_final = y_final[mask]
            g = g[mask]
            
        if len(np.unique(y_final)) < 2: continue
        
        X = np.array([extract_features_vector(s, groups=args.features) for s in segments])
        
        if args.config == "window":
            n_groups = len(np.unique(g))
            # Use GroupKFold to prevent leakage from temporal proximity
            kf = GroupKFold(n_splits=min(5, n_groups))
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
    
    print(f"Pooling data from {len(paths)} subjects for Group analysis...")
    
    for path in paths:
        data, exp = load_mat_data(path)
        if args.config == "trial":
            segments, y, g = get_trial_segments(data, exp)
        else:
            segments, y, g = get_window_epochs(data, exp)
            
        if len(y) == 0: continue
        
        y_final, mask = map_labels(y, args.task)
        if mask is not None:
            segments = [segments[i] for i in range(len(mask)) if mask[i]]
            y_final = y_final[mask]
            
        X = np.array([extract_features_vector(s, groups=args.features) for s in segments])
        all_X.append(X)
        all_y.append(y_final)
        
    X_pool = np.concatenate(all_X)
    y_pool = np.concatenate(all_y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_pool, y_pool, test_size=0.2, stratify=y_pool, random_state=42
    )
    
    clf = get_classifier(args.model)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    results = evaluate_model(y_test, y_pred)
    format_results(results, title=f"Group Analysis Results ({args.task}, {args.model}, feats={args.features})")

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

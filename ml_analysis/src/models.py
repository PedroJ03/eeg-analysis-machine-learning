from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold # IMPORTAR GroupKFold y Stratified
import pandas as pd
import numpy as np

def get_classifier(model_type, seed=42):
    if model_type == 'rf':
        return RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=seed)
    elif model_type == 'svm':
        base_svm = SVC(kernel='rbf', class_weight='balanced', random_state=seed)
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'],
            'kernel': ['rbf']
        }
        
        # Use simple cv=3 for inner loop to match legacy performance/speed tradeoff
        clf = GridSearchCV(
            estimator=base_svm, 
            param_grid=param_grid, 
            cv=3,
            n_jobs=-1,
            scoring='accuracy'
        )
        return clf
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def evaluate_model(y_true, y_pred, labels=None):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        'accuracy': acc,
        'report': pd.DataFrame(report).transpose(),
        'confusion_matrix': cm
    }

def format_results(results, title="Model Results"):
    print(f"\n--- {title} ---")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['report'].round(2))
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])

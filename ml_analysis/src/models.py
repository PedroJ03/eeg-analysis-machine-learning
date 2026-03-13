from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

def get_classifier(model_type, seed=42):
    if model_type == 'rf':
        return RandomForestClassifier(n_estimators=200, random_state=seed)
    elif model_type == 'svm':
        return SVC(kernel='rbf', random_state=seed)
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

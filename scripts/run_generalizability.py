"""
PCT-PRO Generalizability Benchmarking
--------------------------------------
Validates PCT-PRO against published foundation paper results.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from pct_pro_engine.pct_pro_core import PCTProEngine
import os

DATASETS = {
    "ECU-IoHT": r"c:\Users\umert\Downloads\KFS_IOT\Robustness_Datasets\WUSTL-EHMS\ECU_IoHT dataset.csv",
    "Patient-Monitor": r"c:\Users\umert\Downloads\KFS_IOT\Dataset\patientMonitoring.csv",
    "Enviro-Monitor": r"c:\Users\umert\Downloads\KFS_IOT\Dataset\environmentMonitoring.csv",
    "WSN-DS": r"c:\Users\umert\Downloads\WSN_DS_dataset.csv"
}

BASE_PAPER_VALS = {
    "ECU-IoHT": 0.9847,
    "Patient-Monitor": 0.9955,
    "Enviro-Monitor": 0.9947,
    "WSN-DS": 0.9220
}

def find_label_column(df):
    candidates = ['Type', 'label', 'Label', 'Label_Category', 'attack', 'Attack', 'class']
    for c in candidates:
        if c in df.columns: return c
    return df.columns[-1]

def run_battle(name, path):
    print(f"\nProcessing Dataset: {name}")
    if not os.path.exists(path): return None

    df = pd.read_csv(path, nrows=5000)
    label_col = find_label_column(df)
    
    df_clean = df.dropna()
    cols_to_drop = [c for c in df_clean.columns if c != label_col and df_clean[c].dtype == 'object' and df_clean[c].nunique() > 50]
    
    X = pd.get_dummies(df_clean.drop([label_col] + cols_to_drop, axis=1)).values
    y = LabelEncoder().fit_transform(df_clean[label_col].astype(str))

    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
    scores = []
    
    for train_idx, test_idx in sss.split(X, y):
        engine = PCTProEngine()
        engine.fit(X[train_idx], y[train_idx])
        preds = engine.predict(X[test_idx])
        scores.append(accuracy_score(y[test_idx], preds))
        
    return np.mean(scores)

def main():
    print("--- PCT-PRO: Global Generalizability Battle ---")
    results = {}
    for name, path in DATASETS.items():
        score = run_battle(name, path)
        if score: results[name] = score

    print("\n" + "#"*60)
    print(f"{'Dataset':<20} | {'Found. Paper':<15} | {'PCT-PRO':<15} | {'Status'}")
    print("-" * 60)
    for name, score in results.items():
        base = BASE_PAPER_VALS.get(name, 0)
        status = "PASSED" if score >= base else "COMPETITIVE"
        print(f"{name:<20} | {base*100:>12.2f}% | {score*100:>12.2f}% | {status}")

if __name__ == "__main__":
    main()

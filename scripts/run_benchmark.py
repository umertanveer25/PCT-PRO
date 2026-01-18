"""
PCT-PRO Official Benchmarking Script
------------------------------------
Executes 10-fold stratified cross-validation on the WUSTL-EHMS dataset.
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score
from pct_pro_engine.pct_pro_core import PCTProEngine
import os

def main():
    path = r"c:\Users\umert\Downloads\KFS_IOT\Robustness_Datasets\WUSTL-EHMS\wustl-ehms-2020 dataset.csv"
    if not os.path.exists(path):
        print(f"Error: Dataset not found at {path}")
        return

    print("Loading WUSTL-EHMS Dataset...")
    df = pd.read_csv(path, nrows=10000) # Representative sample
    
    X = df.drop(['Label'], axis=1)
    # Convert hex/categorical to numeric
    X = pd.get_dummies(X).values
    y = df['Label'].values

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    metrics = {'acc': [], 'f1': [], 'rec': []}
    
    print(f"\nStarting 10-Fold Validation on PCT-PRO Engine...")
    start_time = time.time()

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        engine = PCTProEngine(d_model=64)
        engine.fit(X_train, y_train)
        preds = engine.predict(X_test)

        metrics['acc'].append(accuracy_score(y_test, preds))
        metrics['f1'].append(f1_score(y_test, preds, average='macro'))
        metrics['rec'].append(recall_score(y_test, preds, average='macro'))
        print(f" Fold {i+1}/10: Accuracy = {metrics['acc'][-1]:.4f}")

    end_time = time.time()
    
    print("\n" + "="*40)
    print(" FINAL BENCHMARK RESULTS (WUSTL-EHMS)")
    print("="*40)
    print(f"Mean Accuracy: {np.mean(metrics['acc'])*100:.2f}% ± {np.std(metrics['acc'])*100:.2f}%")
    print(f"Mean F1-Score: {np.mean(metrics['f1'])*100:.2f}% ± {np.std(metrics['f1'])*100:.2f}%")
    print(f"Mean Recall:   {np.mean(metrics['rec'])*100:.2f}% ± {np.std(metrics['rec'])*100:.2f}%")
    print(f"Total Time:    {end_time - start_time:.2f}s")
    print("="*40)

if __name__ == "__main__":
    main()

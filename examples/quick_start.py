"""
PCT-PRO Quick Start Example
----------------------------
This script demonstrates how to quickly initialize and run the PCT-PRO engine
on a synthetic IoMT dataset.
"""

import numpy as np
from pct_pro_engine.pct_pro_core import PCTProEngine
from sklearn.metrics import classification_report

def main():
    print("--- PCT-PRO: World-Class IoMT-IDS Framework ---")
    
    # 1. Generate synthetic IoMT sensor data (500 samples, 20 features)
    X = np.random.rand(500, 20)
    # Binary labels: 0 (Normal), 1 (Attack)
    y = np.random.randint(0, 2, 500)
    
    # 2. Initialize the State-of-the-Art Engine
    # d_model: depth of attention vectors
    # n_ancestors: number of phylogenetic behavioral clades
    engine = PCTProEngine(d_model=64, n_ancestors=8)
    
    # 3. Train the engine
    print("\nTraining PCT-PRO on sensor streams...")
    engine.fit(X, y)
    
    # 4. Perform real-time detection
    print("Inference running on telemetry...")
    predictions = engine.predict(X)
    
    # 5. Report results
    print("\n--- Detection Performance ---")
    print(classification_report(y, predictions))

if __name__ == "__main__":
    main()

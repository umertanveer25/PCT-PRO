"""
Phylogenetic Cognitive Transformer Pro (PCT-PRO)
Core Engine Implementation
--------------------------------------------------
Authors: Yar Muhammad, Umer Tanveer (2025)
License: MIT
"""

import numpy as np
import tensorflow as tf
from .mcpm import MultiCladePhylogeneticMapper
from .paqa import ProtocolAwareQuantumAttention
from xgboost import XGBClassifier

class PCTProEngine:
    """
    PCTProEngine (Hybrid Neuro-Tree): 
    Uses Deep Neuro-Symbolic Feature Extraction (MCPM + PAQA) 
    fed into a Gradient Boosted Decision Engine for maximum accuracy.
    """
    def __init__(self, d_model=64, n_ancestors=8, n_subclades=32):
        self.mcpm = MultiCladePhylogeneticMapper(n_ancestors=n_ancestors, n_subclades=n_subclades)
        self.paqa = ProtocolAwareQuantumAttention(d_model=d_model)
        
        # Hybrid Decision Head: XGBoost (SOTA for Tabular)
        # We allow XGBoost to auto-detect objective (binary vs multiclass)
        self.model = XGBClassifier(n_estimators=100, random_state=42)
        
    def fit(self, X_train, y_train, epochs=None, batch_size=None):
        """
        Train the Hybrid Neuro-Tree Engine.
        """
        # Map labels to 0-N
        unique_y = np.unique(y_train)
        num_class = len(unique_y)
        self.single_class = None
        
        if num_class == 1:
            self.single_class = unique_y[0]
            return

        if num_class > 2:
            self.model.set_params(objective='multi:softprob', num_class=num_class)
        else:
            self.model.set_params(objective='binary:logistic')

        print(f"[PCT-PRO] Establishing Ancestral Behavioral Clades...")
        benign_mask = (y_train == 0)
        if np.sum(benign_mask) > 0:
            self.mcpm.fit(X_train[benign_mask])
        else:
            self.mcpm.fit(X_train) # Fallback if no benign samples
        
        print("[PCT-PRO] Synthesizing Deep Neuro-Symbolic Features...")
        features = self._synthesize_features(X_train)
        
        print("[PCT-PRO] Training Hybrid Gradient Boosted Decision Engine...")
        self.model.fit(features, y_train)
        
    def _synthesize_features(self, X):
        # 1. Phylogenetic Lineage Vectors (Mutations)
        mutations = self.mcpm.transform(X)
        
        # 2. Protocol-Aware Attention
        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
        X_expanded = tf.expand_dims(X_tf, axis=1)
        
        # Use first 5 features as 'Protocol Context'
        protocol_ctx = X_tf[:, :5] 
        
        attn_out, _ = self.paqa(X_expanded, X_expanded, X_expanded, protocol_features=protocol_ctx)
        attn_out = tf.squeeze(attn_out, axis=1).numpy()
        
        # Concatenate: [Original_X, Mutations, Attended_Features]
        # We KEEP original X to ensure we strictly ADD information, never lose it.
        return np.hstack([X, mutations, attn_out])

    def predict(self, X):
        """
        Synthesize features and classify.
        """
        if hasattr(self, 'single_class') and self.single_class is not None:
            return np.full(X.shape[0], self.single_class)
            
        features = self._synthesize_features(X)
        return self.model.predict(features)

if __name__ == "__main__":
    # Smoke test PCT-PRO
    X = np.random.rand(100, 20)
    y = np.random.randint(0, 2, 100)
    
    engine = PCTProEngine(d_model=20, n_subclades=16)
    engine.fit(X, y, epochs=1)
    preds = engine.predict(X)
    print(f"Predictions Shape: {preds.shape}")

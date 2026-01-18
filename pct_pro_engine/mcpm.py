"""
Multi-Clade Phylogenetic Mapping (MCPM)
---------------------------------------
A behavioral lineage tracking system and feature synthesizer.
Authors: Yar Muhammad, Umer Tanveer (2025)
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.preprocessing import StandardScaler

class MultiCladePhylogeneticMapper:
    """
    MCPM implements hierarchical clustering to establish 'Ancestral Lineages'
    of normal IoMT behavior across multiple clinical states.
    """
    def __init__(self, n_ancestors=8, n_subclades=32):
        self.n_ancestors = n_ancestors
        self.n_subclades = n_subclades
        self.scaler = StandardScaler()
        # BIRCH is efficient for hierarchical clustering on streaming/large data
        self.clade_model = Birch(n_clusters=n_subclades)
        self.ancestor_centers = None
        self.is_fitted = False

    def fit(self, X):
        """
        Fit on normal behavior to map the 'Healthy Genome' of the device.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.clade_model.fit(X_scaled)
        
        # Extract centroids as 'Ancestral Nodes'
        self.ancestor_centers = self.clade_model.subcluster_centers_
        self.is_fitted = True
        return self

    def transform(self, X):
        """
        Returns 'Genetic Dissimilarity' vectors.
        Each feature represents the distance to a specific behavioral clade.
        """
        if not self.is_fitted:
            raise ValueError("MCPM must be fitted before transformation.")
            
        X_scaled = self.scaler.transform(X)
        
        # Calculate distances to all ancestor nodes
        # Use vectorized operations for speed
        diff = X_scaled[:, np.newaxis, :] - self.ancestor_centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Convert distance to 'Evolutionary Proximity' (using RBF kernel approach)
        proximity = np.exp(-dist)
        # Normalize
        proximity /= (np.sum(proximity, axis=1, keepdims=True) + 1e-9)
        
        return proximity

if __name__ == "__main__":
    # Test MCPM
    data = np.random.rand(200, 15)
    mapper = MultiCladePhylogeneticMapper(n_ancestors=5, n_subclades=10)
    mapper.fit(data)
    lineage_vectors = mapper.transform(data)
    print(f"Ancestral Nodes Found: {mapper.ancestor_centers.shape[0]}")
    print(f"Lineage Vector Shape: {lineage_vectors.shape}")
    print(f"Sample Vector: {lineage_vectors[0]}")

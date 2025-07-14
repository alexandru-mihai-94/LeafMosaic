"""clustering.py
Cluster UMAP embeddings using HDBSCAN.
"""

from pathlib import Path
import numpy as np
import hdbscan

def cluster_embeddings(embeddings, min_cluster_size=10, save_path=None):
    """
    Fit HDBSCAN to embeddings (array shape [N, dim]) and return cluster labels.
    Optionally save labels as a .npy file.
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(embeddings)
    if save_path:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        np.save(sp, labels)
    return labels

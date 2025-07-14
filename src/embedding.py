"""embedding.py
Compute UMAP embeddings (2-D or 3-D) from feature vectors.
"""

from pathlib import Path
import numpy as np
import umap

def umap_embed(features, n_components=2, random_state=42, save_path=None):
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    emb = reducer.fit_transform(features)
    if save_path:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        np.save(sp, emb)
    return emb

"""feature_extraction.py
Extract CNN features for a list of tiles using a pretrained backbone.
"""

from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications import DenseNet121, VGG16
from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

_BACKBONES = {
    "densenet": (DenseNet121, densenet_pre, (224, 224)),
    "vgg":      (VGG16,      vgg_pre,      (224, 224)),
}

def load_backbone(name="densenet"):
    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone {name}. Options: {list(_BACKBONES)}")
    cls, pre_fun, target_size = _BACKBONES[name]
    base = cls(weights="imagenet", include_top=False, pooling="avg")
    model = Model(inputs=base.inputs, outputs=base.outputs[0])
    return model, pre_fun, target_size

def features_from_tiles_svd(tiles, n_components=100, target_size=(224, 224), save_path=None):
    """
    Extract features from tiles using Singular Value Decomposition (SVD).

    For each tile, SVD is applied to each color channel separately, and the top
    singular values are concatenated to form a feature vector. This provides a
    compressed representation that captures the most significant patterns in the image.

    Args:
        tiles: List of PIL Image objects
        n_components: Number of singular values to keep per channel (default: 100)
                     Higher values preserve more information but increase dimensionality
        target_size: Tuple (height, width) to resize tiles to (default: (224, 224))
        save_path: Optional path to save features as pickle file

    Returns:
        numpy array of shape (N, D) where N is number of tiles and
        D = n_components * num_channels (typically n_components * 3 for RGB)
    """
    feature_list = []

    for tile in tqdm(tiles, desc="Extracting SVD features"):
        # Resize tile to target size
        tile_resized = tile.resize(target_size)

        # Convert to numpy array and normalize to [0, 1]
        tile_array = img_to_array(tile_resized).astype(np.float32) / 255.0

        # Get dimensions
        height, width, channels = tile_array.shape

        # Extract features from each channel using SVD
        channel_features = []
        for c in range(channels):
            # Get single channel as 2D array
            channel = tile_array[:, :, c]

            # Apply SVD (full_matrices=False for efficiency)
            # Returns: u (height x k), s (k,), v (k x width)
            # where k = min(height, width)
            u, s, v = np.linalg.svd(channel, full_matrices=False)

            # Keep top n_components singular values
            # Singular values are already sorted in descending order
            n_keep = min(n_components, len(s))
            top_singular_values = s[:n_keep]

            # Pad with zeros if we have fewer components than requested
            if n_keep < n_components:
                top_singular_values = np.pad(
                    top_singular_values,
                    (0, n_components - n_keep),
                    mode='constant'
                )

            channel_features.append(top_singular_values)

        # Concatenate features from all channels
        feature_vector = np.concatenate(channel_features)
        feature_list.append(feature_vector)

    # Stack into (N, D) array
    feats = np.stack(feature_list).astype(np.float32)

    # Optionally save to pickle file
    if save_path:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, "wb") as f:
            pickle.dump(feats, f)

    return feats


def features_from_tiles(tiles, backbone="densenet", batch_size=32, save_path=None):
    """Return (N, D) feature array; optionally save as pickle."""
    model, preprocess, target_size = load_backbone(backbone)
    arrs = []
    for tile in tiles:
        tile = tile.resize(target_size)
        arrs.append(img_to_array(tile))
    X = np.stack(arrs).astype(np.float32)
    X = preprocess(X)
    feats = model.predict(X, batch_size=batch_size, verbose=0)
    if save_path:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, "wb") as f:
            pickle.dump(feats, f)
    return feats

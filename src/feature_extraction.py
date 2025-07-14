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

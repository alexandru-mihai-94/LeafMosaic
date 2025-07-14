"""classification.py
Compute coverage and predict species for each tile using a pretrained Keras model.
Writes out a CSV of tile_index, coverage_percent, and predicted probabilities.
"""

import csv
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Full set of classes in your model’s output layer
SPECIES_CLASSES = [
    "Lemna", "Spirodela", "Salvinia_dark", "Salvinia_light",
    "Azolla_green", "Azolla_red", "Water", "water_light"
]

# Mapping from class name to model output index
# Adjust these indices to match your model’s training
species_dict: Dict[str, int] = {'Spirodela': 0, 
                                'Salvinia_dark': 1, 
                                'Salvinia_light': 2, 
                                'Azolla_green': 3, 
                                'Azolla_red': 4, 
                                'Water': 5, 
                                'water_light': 6}


def calculate_coverage_percent(image_bgr: np.ndarray) -> float:
    """Return percent of pixels classified as 'plant' via HSV/LAB masks."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    lightness_mask = cv2.inRange(l_channel, 35, 255)

    green_mask = cv2.inRange(hsv, (30, 60, 60), (85, 255, 255))
    yellow_mask = cv2.inRange(hsv, (18, 80, 100), (35, 255, 255))
    brown_mask = cv2.inRange(hsv, (5, 100, 50), (18, 255, 180))

    plant_mask = cv2.bitwise_or(green_mask, yellow_mask)
    plant_mask = cv2.bitwise_or(plant_mask, brown_mask)
    plant_mask = cv2.bitwise_and(plant_mask, lightness_mask)

    kernel = np.ones((3, 3), np.uint8)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)

    plant_pixels = cv2.countNonZero(plant_mask)
    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    return round((plant_pixels / total_pixels) * 100, 2)


def load_tf_model(model_path: str):
    """Load and return a Keras model (suppresses TF logs)."""
    tf.get_logger().setLevel("ERROR")
    print(f"Loading model from: {model_path}")
    return load_model(model_path)


def predict_species_tf(model, image_bgr: np.ndarray) -> Dict[str, float]:
    """
    Run a single BGR tile through the model.
    Returns a dict of {class_name: probability}.
    """
    img = tf.cast(image_bgr, tf.float32) / 255.0
    img = tf.expand_dims(img, 0)  # batch dimension

    preds = model(img, training=False).numpy()[0]
    output: Dict[str, float] = {cls: 0.0 for cls in SPECIES_CLASSES}
    for cls, idx in species_dict.items():
        if idx < len(preds):
            output[cls] = round(float(preds[idx]), 5)
    return output


def classify_tiles(
    tiles: List[Image.Image],
    model_path: str,
    out_csv: str
) -> None:
    """
    For each tile:
      - compute coverage_percent
      - predict species probabilities
      - write one row in CSV: tile_index, coverage_percent, *SPECIES_CLASSES

    Parameters
    ----------
    tiles : list of PIL.Image
        The list of 100×100 tiles to classify.
    model_path : str
        Path to pretrained Keras .h5 / .keras model.
    out_csv : str
        Path to write the results CSV.
    """
    model = load_tf_model(model_path)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["tile_index", "coverage_percent"] + SPECIES_CLASSES
    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, tile in enumerate(tiles):
            # Convert tile to BGR numpy array
            rgb = np.array(tile)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            coverage = calculate_coverage_percent(bgr)
            preds = predict_species_tf(model, bgr)

            row = {"tile_index": idx, "coverage_percent": coverage}
            row.update(preds)
            writer.writerow(row)
    print(f"Classification results written to: {out_csv}")
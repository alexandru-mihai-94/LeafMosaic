"""run_demo.py
Pipeline: tile → filter → features → UMAP → cluster → visualize → classify.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from tiling import tile_image, filter_tiles
from feature_extraction import features_from_tiles, load_backbone
from embedding import umap_embed
from clustering import cluster_embeddings
from visualization import (
    plot_umap_2d,
    plot_umap_2d_with_tiles,
    plot_umap_3d,
    overlay_clusters,
    overlay_predictions,
    overlay_cluster_outlines
)
from classification import SPECIES_CLASSES
from classification import classify_tiles

def main():
    parser = argparse.ArgumentParser(description="LeafMosaic demo pipeline")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--model", required=True, help="Pretrained Keras model")
    parser.add_argument("--outdir", default="data", help="Output root directory")
    parser.add_argument("--backbone", default="densenet", choices=["densenet", "vgg"],
                        help="CNN backbone")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    tiles_dir = outdir / "tiles"
    feats_path = outdir / "features" / "features.pkl"
    emb2_path = outdir / "embeddings" / "umap2.npy"
    emb3_path = outdir / "embeddings" / "umap3.npy"
    clusters_path = outdir / "embeddings" / "clusters.npy"
    preds_path = outdir / "predictions" / "preds.npy"

    # # 1. Tile & save
    # tiles = tile_image(args.image, tile_size=(100, 100), save_dir=tiles_dir)
    # print(f"Tiled into {len(tiles)} tiles")

    # # 2. Filter empty/white
    # tiles = filter_tiles(tiles, white_threshold=250, max_empty_frac=0.5)
    # print(f"Filtered to {len(tiles)} non-empty tiles")

    # 1) Tile + positions
    tiles, positions = tile_image(args.image, tile_size=(100, 100), save_dir=tiles_dir)
    print(f"Tiled into {len(tiles)} tiles")

    # 2) Filter empty
    tiles, positions = filter_tiles(tiles, positions,
                                    white_threshold=250,
                                    max_empty_frac=0.5)
    print(f"Filtered to {len(tiles)} non-empty tiles")

    np.save(outdir/"embeddings"/"positions.npy", np.array(positions))

    # 3. Feature extraction
    feats = features_from_tiles(tiles, backbone=args.backbone, save_path=feats_path)
    print(f"Features shape: {feats.shape}")

    # 4. UMAP embeddings
    emb2 = umap_embed(feats, n_components=2, save_path=emb2_path)
    emb3 = umap_embed(feats, n_components=3, save_path=emb3_path)

    # 5. Clustering
    clusters = cluster_embeddings(emb2, min_cluster_size=5, save_path=clusters_path)
    unique = np.unique(clusters)
    print(f"Found clusters: {unique.tolist()}")

    # Overlay cluster outlines on the original image
    overlay_path = outdir / "embeddings" / "cluster_outline.png"
    overlay_cluster_outlines(
        original_image_path=args.image,
        cluster_labels=clusters.tolist(),
        positions=positions,
        tile_size=(100, 100),
        output_path=str(overlay_path),
        outline_width=6,
        cmap_name="Set3"
    )
    print(f"Saved cluster outline overlay at {overlay_path}")

    # 6. Visualize
    plot_umap_2d(emb2, outdir/"embeddings"/"umap2d.png", labels=clusters)
    plot_umap_2d_with_tiles(
        emb2, tiles, outdir/"embeddings"/"umap2d_tiles.png", labels=clusters
    )
    plot_umap_3d(emb3, outdir/"embeddings"/"umap3d.html", labels=clusters)
    print("Saved 2-D, thumbnail-overlay, and 3-D plots")

    # 7. Classification
    output_csv = outdir / "predictions" / "tile_predictions.csv"
    classify_tiles(tiles, args.model, str(output_csv))

    # _, preprocess, target_size = load_backbone(args.backbone)
    # preds = classify_tiles(
    #     tiles, args.model, preprocess,
    #     target_size=target_size, save_path=preds_path
    # )
    # print(f"Saved predictions (shape {preds.shape})")
    # preds: shape (N, C) from classify_tiles
    # read CSV
    df = pd.read_csv(output_csv)
    # find which species column has the max value per row
    species_cols = SPECIES_CLASSES
    pred_indices = df[species_cols].values.argmax(axis=1)

    # overlay predictions as outlines
    pred_overlay = outdir / "embeddings" / "prediction_overlay.png"
    pred_overlay2 = outdir / "embeddings" / "prediction_overlay_highres.png"
    overlay_predictions(
        original_image_path=args.image,
        predicted_indices=pred_indices,
        species_list=SPECIES_CLASSES,
        positions=positions,
        tile_size=(100, 100),
        output_path=str(pred_overlay),
        output_path2=str(pred_overlay2),
        outline_width=6,
        cmap_name="Set2",
        legend_font_size=24
    )
    print(f"Saved prediction overlay: {pred_overlay}")

    print("Pipeline complete")

if __name__ == "__main__":
    main()

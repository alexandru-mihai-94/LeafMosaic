# LeafMosaic

LeafMosaic is a lightweight, end-to-end pipeline for exploring community‐level plant classification from ecological images. It tiles a high-resolution image into small patches, extracts deep features via pretrained CNNs, projects those features with UMAP, discovers unsupervised clusters, and finally classifies each tile with a pretrained model.  The result is both unsupervised cluster outlines and supervised species predictions overlaid back on your original image.

<table>
  <tr>
    <td align="center">
      <strong>Clusters</strong><br>
      <img src="data/raw/20230627_bucket34.jpeg" width="400"/>
    </td>
    <td align="center">
      <strong>Clusters</strong><br>
      <img src="data/embeddings/cluster_outline.png" width="400"/>
    </td>
    <td align="center">
      <strong>Predictions</strong><br>
      <img src="data/embeddings/prediction_overlay.png" width="400"/>
    </td>
  </tr>
</table>


## Table of Contents

[Features](#features)  
[Prerequisites](#prerequisites)  
[Installation](#installation)  
[Running the Demo Pipeline](#running-the-demo-pipeline)  



## Features

- **Tiling**: break a large image into 100×100 px patches  
- **Filtering**: discard mostly‐empty/white tiles (> 50% background)  
- **Feature Extraction**: use DenseNet or VGG backbones (ImageNet weights)  
- **Dimensionality Reduction**: UMAP in 2D and 3D for verification  
- **Clustering**: HDBSCAN on 2-D embeddings to find visual groups  
- **Classification**: Keras model predicts dominant species per tile (proprietary training set and transfer learning enabled)
- **Overlays**: draw colored outlines for clusters **and** top predictions back onto the original image, with a clear legend  

---

## Prerequisites

- Python 3.8+  
- `venv`, or a similar virtual-environment tool  
- Git  

---

## Installation


# 1. Clone or copy LeafMosaic/
```bash
cd LeafMosaic
```

# 2. Create and activate venv
```
python3 -m venv venv
source venv/bin/activate
```
# 3. Install Python dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
## Running the Demo Pipeline

```bash
python src/run_demo.py \
  --image data/raw/your_image.jpg \
  --model models/your_classifier.keras \
  --outdir data
```

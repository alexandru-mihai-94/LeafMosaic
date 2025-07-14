# LeafMosaic

LeafMosaic is a demonstration pipeline that

1. Tiles high-resolution ecological images.
2. Extracts deep features with pretrained CNNs.
3. Projects those features with 2-D and 3-D UMAP.
4. Classifies each tile using an existing model.

Run the full pipeline on a single image:

\`\`\`bash
python src/run_demo.py --image data/raw/example.jpg --model models/classifier.h5
\`\`\`

Intermediate data are saved to disk so each step can be inspected.

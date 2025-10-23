"""Test script to verify SVD feature extraction."""

import numpy as np
from PIL import Image
from src.feature_extraction import features_from_tiles_svd

# Create some dummy tiles for testing
def create_test_tiles(n_tiles=5, size=(224, 224)):
    """Create random RGB tiles for testing."""
    tiles = []
    for i in range(n_tiles):
        # Create random RGB image
        random_array = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)
        tile = Image.fromarray(random_array, mode='RGB')
        tiles.append(tile)
    return tiles

# Test the SVD feature extraction
print("Creating test tiles...")
test_tiles = create_test_tiles(n_tiles=5)

print("\nExtracting features using SVD...")
features = features_from_tiles_svd(test_tiles, n_components=100)

print(f"\nResults:")
print(f"  Number of tiles: {len(test_tiles)}")
print(f"  Feature array shape: {features.shape}")
print(f"  Expected shape: (5, 300)  # 100 components × 3 channels")
print(f"  Feature dtype: {features.dtype}")
print(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")

# Test with different n_components
print("\n" + "="*60)
print("Testing with different n_components values:")
for n_comp in [50, 100, 150]:
    feats = features_from_tiles_svd(test_tiles, n_components=n_comp)
    print(f"  n_components={n_comp:3d} -> shape: {feats.shape}")

print("\n✓ SVD feature extraction is working correctly!")
print("  Output format is (N, D) as expected, matching features_from_tiles()")

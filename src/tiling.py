"""tiling.py
Split a large image into fixed-size tiles, optionally save them,
and filter out tiles that are >50% white/empty.
"""

from pathlib import Path
from PIL import Image
import numpy as np

def tile_image(image_path, tile_size=(100, 100), save_dir=None):
    """
    Returns
    -------
    tiles : list[PIL.Image]
    positions : list[tuple(int,int)]
        Top‐left (x,y) of each tile in the original image.
    """
    image_path = Path(image_path)
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    tw, th = tile_size

    tiles = []
    positions = []
    idx = 0

    for y in range(0, H - th + 1, th):
        for x in range(0, W - tw + 1, tw):
            tile = im.crop((x, y, x + tw, y + th))
            tiles.append(tile)
            positions.append((x, y))
            if save_dir:
                sd = Path(save_dir)
                sd.mkdir(parents=True, exist_ok=True)
                tile.save(sd / f"tile_{idx:05d}.png")
            idx += 1

    return tiles, positions


def filter_tiles(tiles, positions, white_threshold=250, max_empty_frac=0.5):
    """
    Filters out tiles where more than max_empty_frac of pixels exceed white_threshold.
    Returns filtered lists of tiles and their positions.
    """
    filtered_tiles = []
    filtered_positions = []

    for tile, (x, y) in zip(tiles, positions):
        gray = np.array(tile.convert("L"))
        empty_frac = np.mean(gray > white_threshold)
        if empty_frac <= max_empty_frac:
            filtered_tiles.append(tile)
            filtered_positions.append((x, y))

    return filtered_tiles, filtered_positions
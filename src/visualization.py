"""visualization.py
2-D/3-D UMAP plots, plus a 2-D plot that overlays tile thumbnails.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import plotly.express as px
from PIL import Image, ImageDraw
import matplotlib.cm as cm

def save_with_resolution(img, path, scale=1, dpi=None):
    """
    Resize by `scale` (integer) and optionally embed `dpi` metadata.
    """
    from PIL import Image
    if scale != 1:
        img = img.resize((img.width*scale, img.height*scale), Image.BICUBIC)
    save_kwargs = {}
    if dpi is not None:
        save_kwargs["dpi"] = (dpi, dpi)
    img.save(path, **save_kwargs)

def overlay_cluster_outlines(
    original_image_path: str,
    cluster_labels: list[int],
    positions: list[tuple[int,int]],
    tile_size: tuple[int,int] = (100, 100),
    output_path: str = "cluster_outline.png",
    output_path2: str = "cluster_outline_highres.png",
    outline_width: int = 8,
    cmap_name: str = "Set3"
):
    """
    Draws a colored border around each tile position according to its cluster label.
    """
    im = Image.open(original_image_path).convert("RGBA")
    draw = ImageDraw.Draw(im)
    tw, th = tile_size

    # build discrete colormap
    n_clusters = max(cluster_labels) + 1
    cmap = cm.get_cmap(cmap_name, n_clusters)

    for label, (x, y) in zip(cluster_labels, positions):
        color = cmap(label)
        # convert to 0–255 RGBA
        rgb = tuple(int(255 * c) for c in color[:3])
        draw.rectangle(
            [x, y, x + tw, y + th],
            outline=rgb,
            width=outline_width
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    save_with_resolution(im, Path(output_path2), scale=2, dpi=300)

def overlay_clusters(
    original_image_path: str,
    cluster_labels: np.ndarray,
    tile_size: tuple[int, int] = (100, 100),
    output_path: str = "cluster_overlay.png",
    output_path2: str = "cluster_overlay_highres.png",
    alpha: float = 0.4,
    cmap_name: str = "tab10",
):
    """
    Draw semi-transparent colored rectangles over each tile on the original image,
    where the color corresponds to the cluster label.

    Parameters
    ----------
    original_image_path : str
        Path to the large source image.
    cluster_labels : (N,) array of ints
        Cluster label for each tile, in the same order that tile_image() produced them.
    tile_size : (width, height)
        Size of each tile in pixels.
    output_path : str
        Where to save the resulting overlay image.
    alpha : float
        Opacity of the colored overlay (0.0 transparent → 1.0 opaque).
    cmap_name : str
        Matplotlib colormap name (must support integer indexing).
    """
    # Load base image
    im = Image.open(original_image_path).convert("RGBA")
    W, H = im.size
    tw, th = tile_size

    # Create transparent overlay
    overlay = Image.new("RGBA", im.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Build a discrete colormap
    n_clusters = int(cluster_labels.max()) + 1
    cmap = cm.get_cmap(cmap_name, n_clusters)

    # Compute number of columns (tiles per row)
    n_cols = W // tw

    for idx, label in enumerate(cluster_labels):
        row = idx // n_cols
        col = idx % n_cols
        x0, y0 = col * tw, row * th
        x1, y1 = x0 + tw, y0 + th

        # RGBA color from colormap
        color = cmap(label)
        fill = (
            int(color[0] * 255),
            int(color[1] * 255),
            int(color[2] * 255),
            int(alpha * 255),
        )
        draw.rectangle([x0, y0, x1, y1], fill=fill)

    # Composite and save
    result = Image.alpha_composite(im, overlay)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(out)

def plot_umap_2d(emb, out_png, labels=None, point_size=6):
    plt.figure(figsize=(6, 6))
    if labels is None:
        plt.scatter(emb[:, 0], emb[:, 1], s=point_size)
    else:
        plt.scatter(emb[:, 0], emb[:, 1], c=labels, s=point_size, cmap="tab10")
    plt.axis("off")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_umap_2d_with_tiles(emb, tiles, out_png, labels=None, zoom=0.5):
    """
    Scatter plot with tile thumbnails overlayed at their UMAP coordinates.
    emb: [N,2] array, tiles: list of PIL.Images of length N.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    if labels is None:
        ax.scatter(emb[:, 0], emb[:, 1], s=10, alpha=0.3)
    else:
        ax.scatter(emb[:, 0], emb[:, 1], c=labels, s=10, cmap="tab10", alpha=0.3)

    for (x0, y0), tile in zip(emb, tiles):
        thumbnail = tile.resize((20, 20))
        img = OffsetImage(thumbnail, zoom=zoom)
        ab = AnnotationBbox(img, (x0, y0), frameon=False)
        ax.add_artist(ab)

    ax.axis("off")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_umap_3d(emb, out_html, labels=None):
    if emb.shape[1] != 3:
        raise ValueError("3-D embedding required")
    fig = px.scatter_3d(
        x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
        color=labels, opacity=0.7
    )
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html)

from PIL import Image, ImageDraw, ImageFont

def overlay_predictions(
    original_image_path: str,
    predicted_indices: np.ndarray,
    species_list: list[str],
    positions: list[tuple[int,int]],
    tile_size: tuple[int,int] = (100, 100),
    output_path: str = "prediction_overlay.png",
    output_path2: str = "prediction_overlay_highres.png",
    outline_width: int = 10,
    cmap_name: str = "Set3",
    legend_font_size: int = 24,
    swatch_size: int = 20,
    legend_padding: int = 10
):
    """
    Draws a colored border around each tile position according to its
    top‐prediction, then appends a legend.

    Parameters
    ----------
    original_image_path : str
    predicted_indices    : (N,) int array
    species_list         : list of class names, length C
    positions            : list of (x,y) tile top-left coords, length N
    tile_size            : (w,h)
    output_path          : where to save the PNG
    outline_width        : border thickness
    cmap_name            : matplotlib colormap
    legend_font_size     : font size for legend
    """
    # Load base image
    im = Image.open(original_image_path).convert("RGBA")
    W, H = im.size
    tw, th = tile_size

    # Prepare drawing
    overlay = Image.new("RGBA", im.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)

    # Colormap
    n = len(species_list)
    cmap = cm.get_cmap(cmap_name, n)

    # Build legend entries
    legend = []
    for i, name in enumerate(species_list):
        col = cmap(i)
        rgba = (
            int(col[0]*255),
            int(col[1]*255),
            int(col[2]*255),
            255
        )
        legend.append((rgba, name))

    # Draw outlines
    for idx, label in enumerate(predicted_indices):
        x, y = positions[idx]
        rgb, _ = legend[label]
        draw.rectangle(
            [x, y, x+tw, y+th],
            outline=rgb,
            width=outline_width
        )

    # Composite overlay
    result = Image.alpha_composite(im, overlay)

    # prepare canvas to hold legend on the right
    font = ImageFont.load_default()
    legend_width = swatch_size + legend_padding + 200
    canvas = Image.new("RGBA", (W + legend_width, H), (255,255,255,255))
    canvas.paste(result, (0, 0))

    ldraw = ImageDraw.Draw(canvas)
    y0 = legend_padding
    for rgba, name in legend:
        # draw swatch
        x0 = W + legend_padding
        ldraw.rectangle(
            [x0, y0, x0 + swatch_size, y0 + swatch_size],
            fill=rgba
        )
        # draw text next to swatch
        ldraw.text(
            (x0 + swatch_size + legend_padding, y0),
            name,
            fill=(0,0,0,255),
            font=font
        )
        y0 += max(swatch_size, legend_font_size) + legend_padding

    # # Create canvas wide enough for legend
    # font = ImageFont.load_default()
    # pad = 10
    # legend_w = 200
    # canvas = Image.new("RGBA", (W+legend_w, H), (255,255,255,255))
    # canvas.paste(result, (0,0))
    # ld = ImageDraw.Draw(canvas)

    # # Draw legend boxes + labels
    # for i, (rgba, name) in enumerate(legend):
    #     y0 = pad + i*(legend_font_size+6)
    #     # colored box
    #     ld.rectangle(
    #         [W+pad, y0, W+pad+legend_font_size, y0+legend_font_size],
    #         fill=rgba
    #     )
    #     # text
    #     ld.text(
    #         (W+pad+legend_font_size+5, y0),
    #         name,
    #         fill=(0,0,0,255),
    #         font=font
    #     )

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    save_with_resolution(canvas, Path(output_path2), scale=2, dpi=300)

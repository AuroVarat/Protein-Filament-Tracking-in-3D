import argparse
import glob
from pathlib import Path

import numpy as np
import tifffile


LABEL_COLORS = [
    "transparent",
    "cyan",
    "yellow",
    "magenta",
    "lime",
    "orange",
    "red",
    "blue",
    "green",
    "violet",
]


def load_mask_tiff(mask_path: Path):
    mask = tifffile.imread(mask_path)
    if mask.ndim == 5:
        return mask[:, :, 0, :, :]
    return mask


def load_mask_series(mask_series_base: str):
    pattern = f"{mask_series_base}_t*.npy"
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No mask series files found for pattern: {pattern}")
    masks = [np.load(path) for path in files]
    return np.stack(masks, axis=0)


def build_label_color_map(mask_data: np.ndarray):
    label_ids = sorted(int(x) for x in np.unique(mask_data) if int(x) > 0)
    color_map = {0: "transparent"}
    for index, label_id in enumerate(label_ids):
        color_map[label_id] = LABEL_COLORS[(index % (len(LABEL_COLORS) - 1)) + 1]
    return color_map


def main():
    parser = argparse.ArgumentParser(description="Open a crop movie and optional mask movie in napari.")
    parser.add_argument("--image", required=True, help="Path to the crop TIFF")
    parser.add_argument("--mask-tiff", help="Path to a mask TIFF/TIFF movie")
    parser.add_argument("--mask-series-base", help="Base path used to assemble mask_tXXXX.npy files")
    parser.add_argument("--channel", type=int, default=1, help="Channel index to open from a 5D crop")
    args = parser.parse_args()

    import napari

    image_path = Path(args.image).resolve()
    image = tifffile.imread(image_path)

    if image.ndim == 5:
        channel_idx = max(0, min(args.channel, image.shape[2] - 1))
        movie = image[:, :, channel_idx, :, :]
        image_name = f"{image_path.stem} [c={channel_idx}]"
    elif image.ndim == 4:
        movie = image
        image_name = image_path.stem
    elif image.ndim == 3:
        movie = image[np.newaxis, ...]
        image_name = image_path.stem
    else:
        raise ValueError(f"Expected a 3D, 4D, or 5D image, found shape {image.shape}")

    viewer = napari.Viewer(title="AISLOP Filament Viewer", ndisplay=3)
    viewer.add_image(movie, name=image_name, rendering="mip", colormap="gray")

    if args.mask_tiff:
        mask_tiff_path = Path(args.mask_tiff).resolve()
        mask_movie = load_mask_tiff(mask_tiff_path)
        if mask_movie.ndim == 3:
            mask_movie = mask_movie[np.newaxis, ...]
        labels_layer = viewer.add_labels(
            mask_movie.astype(np.int32),
            name=mask_tiff_path.stem,
            opacity=0.55,
        )
        labels_layer.color = build_label_color_map(mask_movie)

    if args.mask_series_base:
        mask_movie = load_mask_series(args.mask_series_base)
        labels_layer = viewer.add_labels(
            mask_movie.astype(np.int32),
            name=f"{Path(args.mask_series_base).stem}_mask_series",
            opacity=0.55,
        )
        labels_layer.color = build_label_color_map(mask_movie)

    napari.run()


if __name__ == "__main__":
    main()

"""
Auto Preprocessing: Segmentation
=================================

Combines artefact and source-flux segmentation outputs into a single 5-panel
diagnostic PNG per lens (``segmentation.png`` in the lens directory).

Panel order (single row):
  1. RGB image
  2. VIS image
  3. source_flux.fits  (with inferred multiple-image positions overlaid)
  4. artefact_flux.fits
  5. artefact_binary.fits

Also writes ``positions.json`` (consumed by ``util.load_vis_dataset``).

Run from the project root::

    python preprocess/segmentation.py --sample=dr1_top_500
    python preprocess/segmentation.py --sample=dr1_top_500 --dataset=100_102010484_dr1
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import autolens as al

N_POSITIONS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_local_maxima(flux: np.ndarray) -> list[tuple[float, int, int]]:
    """Return (value, row, col) for every interior local maximum, sorted descending."""
    ny, nx = flux.shape
    maxima = []
    for r in range(1, ny - 1):
        for c in range(1, nx - 1):
            v = flux[r, c]
            if (
                v > flux[r - 1, c]
                and v > flux[r + 1, c]
                and v > flux[r, c - 1]
                and v > flux[r, c + 1]
            ):
                maxima.append((float(v), r, c))
    maxima.sort(reverse=True)
    return maxima


def pixel_to_arcsec(
    row: int, col: int, ny: int, nx: int, pixel_scale: float
) -> list[float]:
    """Convert (row, col) to AutoLens [y, x] arcsec using half-pixel offset convention."""
    y = (ny / 2 - 0.5 - row) * pixel_scale
    x = (col - nx / 2 + 0.5) * pixel_scale
    return [y, x]


def load_vis_image(dataset_path, fits_name):
    with fits.open(dataset_path / fits_name) as hdul:
        for hdu in hdul:
            name = str(hdu.name).upper()
            if "VIS" in name and ("BGSUB" in name or "FLUX" in name):
                return hdu.data
        return hdul[1].data


def load_vis_noise_map(dataset_path, fits_name):
    with fits.open(dataset_path / fits_name) as hdul:
        for hdu in hdul:
            name = str(hdu.name).upper()
            if "VIS" in name and "RMS" in name:
                return hdu.data
        return hdul[3].data


def arcsinh_norm(image: np.ndarray, stretch_factor: float = 10.0) -> np.ndarray:
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros_like(image)
    vmin = np.nanpercentile(image[finite], 1)
    vmax_abs = np.nanpercentile(np.abs(image[finite]), 99)
    stretch = vmax_abs / stretch_factor if vmax_abs > 0 else 1.0
    return np.arcsinh((image - vmin) / stretch)


def make_extent(image: np.ndarray, pixel_scale: float) -> list[float]:
    ny, nx = image.shape[:2]
    hw_x = nx / 2 * pixel_scale
    hw_y = ny / 2 * pixel_scale
    return [-hw_x, hw_x, -hw_y, hw_y]


# --- tick helpers (mirrors autoarray.plot.utils) ----------------------------


def _inward_ticks(lo: float, hi: float, factor: float = 0.75, n: int = 3) -> np.ndarray:
    centre = (lo + hi) / 2.0
    return np.linspace(
        centre + (lo - centre) * factor,
        centre + (hi - centre) * factor,
        n,
    )


def _round_ticks(values: np.ndarray, sig: int = 2) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        nonzero = np.where(values != 0, np.abs(values), 1.0)
        mags = np.where(values != 0, 10 ** (sig - 1 - np.floor(np.log10(nonzero))), 1.0)
    return np.round(values * mags) / mags


def _arcsec_labels(ticks: np.ndarray) -> list[str]:
    labels = [f"{v:g}" for v in ticks]
    if all(label.endswith(".0") for label in labels):
        labels = [label[:-2] for label in labels]
    return [f'{label}"' for label in labels]


def apply_arcsec_ticks(ax: plt.Axes, extent: list[float]) -> None:
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    xticks = _round_ticks(_inward_ticks(xmin, xmax))
    yticks = _round_ticks(_inward_ticks(ymin, ymax))
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(_arcsec_labels(xticks))
    ax.set_yticklabels(_arcsec_labels(yticks), rotation=90, va="center")
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


def compute_positions(
    flux: np.ndarray,
    snr_map: np.ndarray | None,
    ny: int,
    nx: int,
    pixel_scale: float,
) -> list[list[float]]:
    """Return up to N_POSITIONS arcsec positions from the brightest local maxima."""
    SNR_THRESHOLD = 3.0
    SNR_STEP = 0.1
    all_maxima = find_local_maxima(flux)
    maxima = [
        (v, r, c)
        for v, r, c in all_maxima
        if snr_map is None or snr_map[r, c] > SNR_THRESHOLD
    ]

    if not maxima:
        return []

    selected = maxima[:N_POSITIONS]
    positions = [pixel_to_arcsec(r, c, ny, nx, pixel_scale) for _, r, c in selected]

    # If no counter-image found, lower the SNR threshold one step at a time
    has_counter = any(
        p[0] * positions[0][0] < 0 or p[1] * positions[0][1] < 0 for p in positions[1:]
    )
    if not has_counter and snr_map is not None:
        threshold = SNR_THRESHOLD - SNR_STEP
        while threshold >= 0:
            lower_maxima = sorted(
                [(v, r, c) for v, r, c in all_maxima if snr_map[r, c] > threshold],
                reverse=True,
            )
            for v, r, c in lower_maxima:
                candidate = pixel_to_arcsec(r, c, ny, nx, pixel_scale)
                if candidate not in positions and (
                    candidate[0] * positions[0][0] < 0
                    or candidate[1] * positions[0][1] < 0
                ):
                    positions[-1] = candidate
                    break
            else:
                threshold -= SNR_STEP
                continue
            break

    return positions


# ---------------------------------------------------------------------------
# Diagnostic PNG
# ---------------------------------------------------------------------------


def _draw_mask_circle(
    ax: plt.Axes, mask_centre: list[float], mask_radius: float
) -> None:
    """Overlay the circular mask boundary on *ax* (arcsec coordinates)."""
    cy, cx = mask_centre
    circle = mpatches.Circle(
        (cx, cy),
        mask_radius,
        edgecolor="white",
        facecolor="none",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(circle)


def save_diagnostic_png(
    rgb_image: np.ndarray,
    vis_image: np.ndarray,
    source_flux: np.ndarray,
    artefact_flux: np.ndarray,
    artefact_binary: np.ndarray,
    lens_flux: np.ndarray,
    lens_binary: np.ndarray,
    pixel_scale: float,
    mask_centre: list[float],
    mask_radius: float,
    position_list: list[list[float]],
    output_path: Path,
) -> None:
    """Six-panel 2×3 PNG.

    Row 0: RGB              | VIS image   | source_flux (+ positions)
    Row 1: artefact_flux    | artefact_binary | VIS × artefact mask
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    cmap = "viridis"
    vis_extent = make_extent(vis_image, pixel_scale)

    # --- Row 0 ----------------------------------------------------------------

    # (0,0) RGB — spatial extent so we can zoom to the mask circle
    rgb_extent = make_extent(rgb_image, pixel_scale)
    axes[0, 0].imshow(rgb_image, origin="upper", extent=rgb_extent, aspect="auto")
    cy, cx = mask_centre
    axes[0, 0].set_xlim(cx - mask_radius, cx + mask_radius)
    axes[0, 0].set_ylim(cy - mask_radius, cy + mask_radius)
    axes[0, 0].set_title("RGB", fontsize=20)
    apply_arcsec_ticks(
        axes[0, 0],
        [cx - mask_radius, cx + mask_radius, cy - mask_radius, cy + mask_radius],
    )

    # (0,1) VIS image
    axes[0, 1].imshow(
        arcsinh_norm(vis_image),
        origin="upper",
        extent=vis_extent,
        cmap="gray",
        aspect="auto",
    )
    axes[0, 1].set_title("VIS image", fontsize=20)
    apply_arcsec_ticks(axes[0, 1], vis_extent)
    _draw_mask_circle(axes[0, 1], mask_centre, mask_radius)

    # (0,2) source_flux with positions
    src_extent = make_extent(source_flux, pixel_scale)
    axes[0, 2].imshow(
        arcsinh_norm(source_flux),
        origin="upper",
        extent=src_extent,
        cmap=cmap,
        aspect="auto",
    )
    for y, x in position_list:
        axes[0, 2].plot(x, y, "k+", markersize=14, markeredgewidth=2)
    axes[0, 2].set_title(f"source_flux ({len(position_list)} positions)", fontsize=20)
    apply_arcsec_ticks(axes[0, 2], src_extent)
    _draw_mask_circle(axes[0, 2], mask_centre, mask_radius)

    # --- Row 1 ----------------------------------------------------------------

    # (1,0) artefact_flux
    art_flux_extent = make_extent(artefact_flux, pixel_scale)
    axes[1, 0].imshow(
        arcsinh_norm(artefact_flux),
        origin="upper",
        extent=art_flux_extent,
        cmap=cmap,
        aspect="auto",
    )
    axes[1, 0].set_title("artefact_flux", fontsize=20)
    apply_arcsec_ticks(axes[1, 0], art_flux_extent)
    _draw_mask_circle(axes[1, 0], mask_centre, mask_radius)

    # (1,1) artefact_binary
    art_bin_extent = make_extent(artefact_binary, pixel_scale)
    axes[1, 1].imshow(
        artefact_binary,
        origin="upper",
        extent=art_bin_extent,
        cmap=cmap,
        interpolation="nearest",
        aspect="auto",
    )
    axes[1, 1].set_title("artefact_binary", fontsize=20)
    apply_arcsec_ticks(axes[1, 1], art_bin_extent)
    _draw_mask_circle(axes[1, 1], mask_centre, mask_radius)

    # (1,2) Input image: VIS - lens_flux, lens-binary region zeroed, artefact pixels masked out
    artefact_mask = np.isfinite(artefact_binary) & (artefact_binary > 0)
    lens_mask = np.isfinite(lens_binary) & (lens_binary > 0)
    subtracted = vis_image.astype(np.float32) - lens_flux
    subtracted = np.where(lens_mask, np.nan, subtracted)
    input_image = np.where(artefact_mask, np.nan, subtracted)
    cmap_black_nan = plt.get_cmap(cmap).copy()
    cmap_black_nan.set_bad("black")
    axes[1, 2].imshow(
        arcsinh_norm(input_image, stretch_factor=50.0),
        origin="upper",
        extent=vis_extent,
        cmap=cmap_black_nan,
        aspect="auto",
    )
    axes[1, 2].set_title("Input Image (with lens subtracted)", fontsize=20)
    apply_arcsec_ticks(axes[1, 2], vis_extent)
    _draw_mask_circle(axes[1, 2], mask_centre, mask_radius)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-lens processing
# ---------------------------------------------------------------------------


def process_lens(lens_dir: Path, overview_dir: Path) -> None:
    lens_name = lens_dir.name
    seg_dir = lens_dir / "segmentation"

    source_flux_path = seg_dir / "source_flux.fits"
    artefact_flux_path = seg_dir / "artefact_flux.fits"
    artefact_binary_path = seg_dir / "artefact_binary.fits"

    # Require at least one segmentation file to be present
    if not any(p.exists() for p in (source_flux_path, artefact_binary_path)):
        print(f"  [skip] {lens_name}: no segmentation files found")
        return

    # Load info.json (create defaults if missing)
    info_path = lens_dir / "info.json"
    try:
        with open(info_path) as f:
            info = json.load(f)
    except FileNotFoundError:
        print(f"  [warn] {lens_name}: info.json not found, creating with defaults")
        info = {"pixel_scale": 0.1, "mask_radius": 3.5, "mask_centre": [0.0, 0.0]}
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

    pixel_scale = float(info["pixel_scale"])
    mask_radius = float(info.get("mask_radius", 3.5))
    mask_centre = list(info.get("mask_centre", [0.0, 0.0]))

    # RGB
    rgb_candidates = [
        lens_dir / f"{lens_name}.png",
        lens_dir / "rgb_0.png",
    ]
    for rgb_path in rgb_candidates:
        try:
            rgb_image = plt.imread(rgb_path)
            break
        except Exception:
            rgb_image = None
    if rgb_image is None:
        print(f"  [warn] {lens_name}: could not load {lens_name}.png or rgb_0.png")
        rgb_image = np.zeros((10, 10, 3))

    fits_name = lens_name + ".fits"

    # Source flux + positions
    position_list: list[list[float]] = []
    if source_flux_path.exists():
        source_flux = fits.getdata(source_flux_path).astype(np.float32)
        ny, nx = source_flux.shape

        try:
            noise_map = load_vis_noise_map(lens_dir, fits_name).astype(np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                snr_map = np.where(noise_map > 0, source_flux / noise_map, 0.0)
        except Exception as exc:
            print(f"  [warn] {lens_name}: could not load noise map: {exc}")
            snr_map = None

        position_list = compute_positions(source_flux, snr_map, ny, nx, pixel_scale)

        if not position_list:
            print(f"  [warn] {lens_name}: no valid source positions found")
        else:
            if len(position_list) < N_POSITIONS:
                print(
                    f"  [warn] {lens_name}: only {len(position_list)} position(s) found (need {N_POSITIONS})"
                )
            positions = al.Grid2DIrregular(values=position_list)
            al.output_to_json(obj=positions, file_path=lens_dir / "positions.json")
            print(f"  [ok]   {lens_name}: {len(position_list)} positions written")
    else:
        print(f"  [warn] {lens_name}: source_flux.fits not found, skipping positions")
        source_flux = np.zeros((10, 10), dtype=np.float32)

    # Artefact flux
    if artefact_flux_path.exists():
        artefact_flux = fits.getdata(artefact_flux_path).astype(np.float32)
    else:
        print(f"  [warn] {lens_name}: artefact_flux.fits not found")
        artefact_flux = np.zeros_like(source_flux)

    # Artefact binary
    if artefact_binary_path.exists():
        artefact_binary = fits.getdata(artefact_binary_path).astype(np.float32)
    else:
        print(f"  [warn] {lens_name}: artefact_binary.fits not found")
        artefact_binary = np.zeros_like(source_flux)

    # VIS image
    try:
        vis_image = load_vis_image(lens_dir, fits_name)
    except Exception as exc:
        print(f"  [warn] {lens_name}: could not load VIS image: {exc}")
        vis_image = np.zeros_like(source_flux)
    # Lens flux
    lens_flux_path = seg_dir / "lens_flux.fits"
    if lens_flux_path.exists():
        lens_flux = fits.getdata(lens_flux_path).astype(np.float32)
    else:
        print(f"  [warn] {lens_name}: lens_flux.fits not found, using zeros")
        lens_flux = np.zeros_like(source_flux)

    # Lens binary
    lens_binary_path = seg_dir / "lens_binary.fits"
    if lens_binary_path.exists():
        lens_binary = fits.getdata(lens_binary_path).astype(np.float32)
    else:
        print(f"  [warn] {lens_name}: lens_binary.fits not found, using zeros")
        lens_binary = np.zeros_like(source_flux)

    # Save combined diagnostic PNG in the lens directory and the sample overview folder
    for output_path in (
        lens_dir / "segmentation.png",
        overview_dir / f"{lens_name}.png",
    ):
        try:
            save_diagnostic_png(
                rgb_image=rgb_image,
                vis_image=vis_image,
                source_flux=source_flux,
                artefact_flux=artefact_flux,
                artefact_binary=artefact_binary,
                lens_flux=lens_flux,
                lens_binary=lens_binary,
                pixel_scale=pixel_scale,
                mask_centre=mask_centre,
                mask_radius=mask_radius,
                position_list=position_list,
                output_path=output_path,
            )
        except Exception as exc:
            print(f"  [warn] {lens_name}: could not save {output_path.name}: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        metavar="name",
        required=True,
        help="Sample subdirectory inside dataset/ (e.g. dr1_top_500).",
    )
    parser.add_argument(
        "--dataset",
        metavar="name",
        required=False,
        default=None,
        help="Optional single lens directory name under the dataset root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = Path("dataset") / args.sample

    if args.dataset:
        lens_dirs = [dataset_root / args.dataset]
        if not lens_dirs[0].is_dir():
            print(f"Lens directory not found: {lens_dirs[0]}")
            return
    else:
        lens_dirs = sorted(d for d in dataset_root.iterdir() if d.is_dir())

    if not lens_dirs:
        print(f"No lens directories found under {dataset_root}")
        return

    overview_dir = dataset_root / "segmentation"
    overview_dir.mkdir(exist_ok=True)

    print(f"Processing {len(lens_dirs)} lenses...")
    for lens_dir in lens_dirs:
        process_lens(lens_dir, overview_dir)

    print("Done.")


if __name__ == "__main__":
    main()

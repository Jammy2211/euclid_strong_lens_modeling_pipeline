"""
Simulator: Euclid VIS Imaging Dataset (for profiling)
=====================================================

Simulates a single Euclid-like imaging dataset at 0.1"/pixel with a
Sersic lens light + Isothermal mass + ExternalShear lens and a cored
Sersic source. Used by ``profiling/delaunay.py`` as the input dataset
for per-step JIT profiling of the Delaunay imaging likelihood.

Run from the repository root::

    python profiling/simulator.py

Output layout::

    profiling/dataset/euclid/
        data.fits
        psf.fits
        noise_map.fits
        tracer.json

The simulator is seeded (``noise_seed=1``) so downstream correctness
checks in the profiling script are deterministic.
"""

import argparse
import numpy as np
from pathlib import Path

import autolens as al
import autolens.plot as aplt


PIXEL_SCALE = 0.1
PSF_SHAPE = (21, 21)
PSF_SIGMA = 0.1


def simulate(mask_radius: float = 3.5):
    """Simulate the Euclid profiling dataset.

    Parameters
    ----------
    mask_radius
        Mask radius in arcseconds. Determines the image shape so the
        eventual circular mask fits within the simulated frame.
    """
    dataset_path = Path("profiling") / "dataset" / "euclid"

    shape_pixels = int(np.ceil(2 * mask_radius / PIXEL_SCALE))
    if shape_pixels % 2 == 0:
        shape_pixels += 1

    grid = al.Grid2D.uniform(
        shape_native=(shape_pixels, shape_pixels),
        pixel_scales=PIXEL_SCALE,
    )

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=grid,
        sub_size_list=[32, 8, 2],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )
    grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

    psf = al.Convolver.from_gaussian(
        shape_native=PSF_SHAPE,
        sigma=PSF_SIGMA,
        pixel_scales=grid.pixel_scales,
    )

    simulator = al.SimulatorImaging(
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_poisson_noise_to_data=True,
        noise_seed=1,
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
            intensity=2.0,
            effective_radius=0.6,
            sersic_index=3.0,
        ),
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.6,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        ),
        shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.SersicCore(
            centre=(0.0, 0.0),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            intensity=4.0,
            effective_radius=0.1,
            sersic_index=1.0,
        ),
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    aplt.fits_imaging(
        dataset=dataset,
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        overwrite=True,
    )

    al.output_to_json(
        obj=tracer,
        file_path=dataset_path / "tracer.json",
    )

    print(f"  Dataset simulated: {dataset_path}")
    print(f"    Pixel scale:  {PIXEL_SCALE} arcsec/pixel")
    print(f"    Grid shape:   {shape_pixels} x {shape_pixels}")
    print(f"    PSF shape:    {PSF_SHAPE[0]} x {PSF_SHAPE[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate the Euclid profiling imaging dataset."
    )
    parser.add_argument(
        "--mask-radius",
        type=float,
        default=3.5,
        help="Mask radius in arcseconds (default: 3.5)",
    )
    args = parser.parse_args()
    simulate(args.mask_radius)

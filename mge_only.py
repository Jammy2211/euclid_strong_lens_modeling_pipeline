"""
Modeling Features: Multi Gaussian Expansion
===========================================

A multi Gaussian expansion (MGE) decomposes the lens light into ~15-100 Gaussians, where the `intensity` of every
Gaussian is solved for via a linear algebra using a process called an "inversion" (see the `light_parametric_linear.py`
feature for a full description of this).

This script fits the MGE to a strong lens image, but does not fit the mass model or source light. Analysis of JWST
data has shown this is a relatively quick way to get a clean lens light subtraction, which can be used to visually
inspect the lensed source emission.

A full description of MGE can be found at the following link:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/features/multi_gaussian_expansion.ipynb
"""


def fit(
    dataset_name: str,
    mask_radius: float = 3.0,
    number_of_cores: int = 1,
    iterations_per_update: int = 5000,
):
    import numpy as np
    import os
    import sys
    from os import path
    import autofit as af
    import autolens as al
    import autolens.plot as aplt

    sys.path.insert(0, os.getcwd())
    import slam

    """
    __Dataset__ 

    Load, plot and mask the `Imaging` data.
    """
    dataset_waveband = "vis"
    dataset_path = path.join("dataset", dataset_name, dataset_waveband)

    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        pixel_scales=0.1,
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )

    dataset = dataset.apply_mask(mask=mask)

    dataset = dataset.apply_over_sampling(
        over_sampling=al.OverSamplingDataset(
            uniform=al.OverSamplingUniform.from_radial_bins(
                grid=dataset.grid,
                sub_size_list=[4, 2, 1],
                radial_list=[0.1, 0.3],
                centre_list=[(0.0, 0.0)],
            )
        )
    )

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    """
    __Settings AutoFit__

    The settings of autofit, which controls the output paths, parallelization, database use, etc.
    """
    settings_search = af.SettingsSearch(
        path_prefix=path.join("euclid_pipeline"),
        unique_tag=dataset_name,
        info=None,
        number_of_cores=number_of_cores,
        session=None,
    )

    """
    __Redshifts__

    The redshifts of the lens and source galaxies.
    """
    redshift_lens = 0.5

    """
    __HPC Mode__

    When running in parallel via Python `multiprocessing`, display issues with the `matplotlib` backend can arise
    and cause the code to crash.

    HPC mode sets the backend to mitigate this issue and is set to run throughout the entire pipeline below.

    The `iterations_per_update` below specifies the number of iterations performed by the non-linear search between
    output, where visuals of the maximum log likelihood model, lens model parameter estimates and other information
    are output to hard-disk.
    """
    from autoconf import conf

    conf.instance["general"]["hpc"]["hpc_mode"] = True
    conf.instance["general"]["hpc"]["iterations_per_update"] = iterations_per_update

    """
    __SOURCE LP PIPELINE__

    The SOURCE LP PIPELINE uses one search to initialize a robust model for the source galaxy's light, which in 
    this example:

     - Uses a multi Gaussian expansion with 2 sets of 20 Gaussians for the lens galaxy's light.

     - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.

     - Uses a multi Gaussian expansion with 1 set of 20 Gaussians for the source galaxy's light.

     __Settings__:

     - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS TOTAL PIPELINE).
    """
    analysis = al.AnalysisImaging(dataset=dataset)

    # Lens Light

    centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    total_gaussians = 20
    gaussian_per_basis = 2

    log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

    bulge_gaussian_list = []

    for j in range(gaussian_per_basis):
        gaussian_list = af.Collection(
            af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
        )

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0
            gaussian.centre.centre_1 = centre_1
            gaussian.ell_comps = gaussian_list[0].ell_comps
            gaussian.sigma = 10 ** log10_sigma_list[i]

        bulge_gaussian_list += gaussian_list

    lens_bulge = af.Model(
        al.lp_basis.Basis,
        profile_list=bulge_gaussian_list,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
            ),
        ),
    )

    search = af.Nautilus(
        name="mge_only",
        **settings_search.search_dict,
        n_live=75,
        iterations_per_update=2000
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    slam.slam_util.output_subplot_mge_only_png(
        output_path=dataset_path,
        result_list=[result],
        filename="mge_only",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lens Model Inputs")
    parser.add_argument(
        "--dataset", metavar="path", required=True, help="the path to the dataset"
    )

    parser.add_argument(
        "--mask_radius",
        metavar="float",
        required=False,
        help="The Circular Radius of the Mask",
    )

    parser.add_argument(
        "--number_of_cores",
        metavar="int",
        required=False,
        help="The number of cores to parallelize the fit",
    )

    parser.add_argument(
        "--iterations_per_update",
        metavar="int",
        required=False,
        help="The number of iterations between each update",
    )

    args = parser.parse_args()

    fit(
        dataset_name=args.dataset,
        mask_radius=float(args.mask_radius),
        number_of_cores=int(args.number_of_cores),
        iterations_per_update=int(args.iterations_per_update),
    )

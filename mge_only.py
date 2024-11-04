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


def fit():
    # %matplotlib inline
    # from pyprojroot import here
    # workspace_path = str(here())
    # %cd $workspace_path
    # print(f"Working Directory has been set to `{workspace_path}`")

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
    dataset_name = str(sys.argv[1])
    dataset_waveband = "vis"
    dataset_path = path.join("dataset", dataset_name, dataset_waveband)

    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        pixel_scales=0.1,
    )

    mask_radius = 1.6

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
        number_of_cores=4,
        session=None,
    )

    """
    __Redshifts__

    The redshifts of the lens and source galaxies.
    """
    redshift_lens = 0.5
    redshift_source = 1.0

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
    fit()

"""
Euclid Pipeline: Start Here
===========================

This scripts allows you to fit an simple and initial lens model using the Euclid lens modeling pipeline locally
on your computer. It comes with an example Euclid strong lens dataset, which is fitted using the pipeline, and using
a GPU takes ~ 10 minutes to run.

The script itself is running **PyAutoLens**, which will require installation first via the instructions on the
GitHub page:

https://github.com/Jammy2211/euclid_strong_lens_modeling_pipeline

The pipeline can be run as a "black box", whereby you pass it the dataset you want it to fit and it automatically
fits it without understanding how the pipeline works.

The pipeline automatically outputs results and visualization to hard-disk in the `output` folder, which if you are
running it as a black box is likely all you will look at. This includes a summary of the fit, the lens model,
parameter inferences and errors, and much more. For small lens samples, manually navigating the results in the `output`
folder is sufficient to do science.

For large samples this becomes slow and cumbersome. The folder `workflow` contains example scripts that show how to
use database functionality to load, query, and output .csv, .png and .fits files summarzing fits to large samples of
lenses. This includes .fits images of the deblendeded lens and source images, source-plane source econstructions and
a .csv file containing lens model quantities like the Einstein Radius.

Please contact James Nightingale on the Euclid Consortium SLACK with any questions or if you would like other
information on the pipeline.

__Initial Lens Model__

This script fits the initial lens model, the `start_here` lens model, from which all lens modeling variations
and extensions are built upon. If you are new to lens modeling or the Euclid modeling pipeline, this is
a great starting point.

The script uses a Multi Gaussian Expansion (MGE) to model the lens light and source light of the strong lens system.
It uses a Singular Isothermal Ellipsoid (SIE) plus shear mass model for the lens mass. An MGE decomposes the light of
a galaxy into ~15-100 Gaussians, where the `intensity` of every Gaussian is solved for via a linear algebra using
a process called an "inversion". It essentially forms a relatively simple parameter space to sample which is
signficantly accelerated when modeling is performed using GPUs via JAX.
"""

from start_here import fit
from pipelines.lens_model_waveband import fit_waveband


def fit_sersic(
    dataset_name: str,
    vis_result,
    mask_radius: float = 3.0,
    iterations_per_quick_update: int = 5000,
):
    import util
    import json
    import numpy as np
    from pathlib import Path

    import autofit as af
    import autolens as al

    """
    __Dataset__
    """
    dataset_main_path = Path("dataset") / dataset_name
    dataset_fits_name = f"{dataset_name}.fits"

    dataset_index_dict = util.dataset_instrument_hdu_dict_via_fits_from(
        dataset_path=dataset_main_path,
        dataset_fits_name=dataset_fits_name,
        image_tag="_BGSUB",  # Depends on how Euclid cutout was made
    )

    dataset_waveband = "vis"

    vis_index = dataset_index_dict[dataset_waveband]

    dataset = al.Imaging.from_fits(
        data_path=dataset_main_path / dataset_fits_name,
        data_hdu=vis_index * 3 + 1,
        noise_map_path=dataset_main_path / dataset_fits_name,
        noise_map_hdu=vis_index * 3 + 3,
        psf_path=dataset_main_path / dataset_fits_name,
        psf_hdu=vis_index * 3 + 2,
        pixel_scales=0.1,
        check_noise_map=False,
    )

    dataset_centre = dataset.data.brightest_sub_pixel_coordinate_in_region_from(
        region=(-0.3, 0.3, -0.3, 0.3), box_size=2
    )

    """
    __Info__
    """
    try:
        with open(dataset_main_path / "info.json") as json_file:
            info = json.load(json_file)
            json_file.close()
    except FileNotFoundError:
        info = {}

    """
    __Header__
    """
    try:
        header = al.header_obj_from(
            file_path=dataset_main_path / dataset_fits_name,
            hdu=vis_index * 3 + 1,
        )
        magzero = header["MAGZERO"]
    except FileNotFoundError:
        magzero = None

    """
    __Extra Galaxy Removal__
    """
    try:
        mask_extra_galaxies = al.Mask2D.from_fits(
            file_path=dataset_main_path / "mask_extra_galaxies.fits",
            pixel_scales=0.1,
            invert=True,
        )

        dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)
    except FileNotFoundError:
        pass

    """
    __Mask__
    """
    if mask_radius is None:
        mask_radius = info.get("mask_radius") or 3.0
    mask_centre = info.get("mask_centre") or (0.0, 0.0)

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
        centre=mask_centre,
    )

    dataset = dataset.apply_mask(mask=mask)

    """
    __Over Sampling__
    
    The Sersic profile diverges at the centre of the profile, which can cause inaccuracies if
    the central pixel is not sampled enough. We therefore increase the over sampling in regions
    of the source and lens galaxy centres adaptively.
    """
    tracer = vis_result.max_log_likelihood_tracer

    traced_grid = tracer.traced_grid_2d_list_from(
        grid=dataset.grid,
    )[-1]

    source_centre = tracer.galaxies[1].bulge.profile_list[0].centre

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=traced_grid,
        sub_size_list=[16, 4, 2],
        radial_list=[0.1, 0.3],
        centre_list=[source_centre],
    )

    over_sample_size_lens = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[16, 4, 1],
        radial_list=[0.1, 0.3],
        centre_list=[dataset_centre],
    )

    over_sample_size = np.where(
        over_sample_size > over_sample_size_lens,
        over_sample_size,
        over_sample_size_lens,
    )
    over_sample_size = al.Array2D(values=over_sample_size, mask=mask)

    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

    """
    __Settings AutoFit__

    The settings of autofit, which controls the output paths, parallelization, database use, etc.
    """
    dataset_waveband = "vis"

    settings_search = af.SettingsSearch(
        path_prefix=Path(dataset_name),
        unique_tag="sersic_lens_model",
        info={"magzero": magzero},
        session=None,
    )

    """
    __Redshifts__
    """
    redshift_lens = 0.5
    redshift_source = 1.0

    """
    __Model: Sersic Lens Model__

    This pipeline fits the lens and source galaxy light together, both as Sersics, using a fixed lens mass model
    from the `initial_lens_model` pipeline:

     - The lens light is a linear Sersic profile [6 non-linear parameters].  

     - The source light is represented by a linear Sersic profile [6 non-linear parameters].  

     - The lens mass is modeled as an Isothermal Ellipsoid (SIE) with a centre fixed to the brighest pixel and an
        External Shear, both fixed to the result of the `initial_lens_model` pipeline [0 non-linear parameters].

    Overall the model has 12 non-linear parameters.    
    """
    # Lens:

    lens_bulge = af.Model(al.lp_linear.Sersic)
    lens_bulge.centre.centre_0 = (
        vis_result.model_centred.galaxies.lens.bulge.profile_list[0].centre.centre_0
    )
    lens_bulge.centre.centre_1 = (
        vis_result.model_centred.galaxies.lens.bulge.profile_list[0].centre.centre_1
    )

    mass = vis_result.instance.galaxies.lens.mass
    shear = vis_result.instance.galaxies.lens.shear

    lens = af.Model(
        al.Galaxy, redshift=redshift_lens, bulge=lens_bulge, mass=mass, shear=shear
    )

    # Source:

    source_bulge = af.Model(al.lp_linear.Sersic)
    source_bulge.centre.centre_0 = (
        vis_result.model_centred.galaxies.source.bulge.profile_list[0].centre.centre_0
    )
    source_bulge.centre.centre_1 = (
        vis_result.model_centred.galaxies.source.bulge.profile_list[0].centre.centre_1
    )

    source = af.Model(al.Galaxy, redshift=redshift_source, bulge=source_bulge)

    # Overall Lens Model:

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    """
    __Model Fit__
    """
    analysis = util.AnalysisImaging(
        dataset=dataset,
        dataset_main_path=dataset_main_path,
        use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
        title_prefix=dataset_waveband.upper(),
        **settings_search.info,
    )

    search = af.Nautilus(
        name=dataset_waveband,  # The name of the fit and folder results are output to.
        **settings_search.search_dict,
        n_live=100,  # The number of Nautilus "live" points, increase for more complex models.
        batch_size=50,  # For fast GPU fitting lens model fits are batched and run simultaneously.
        iterations_per_quick_update=iterations_per_quick_update,
        # Every N iterations the max likelihood model is visualized in the Jupter Notebook and output to hard-disk.
        n_like_max=100000,
        # The maximum number of likelihood evaluations, models typically take < 30000 samples so this stops runaway fits.
    )

    print(
        """
        The non-linear search has begun running.

        This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!

        On-the-fly updates every iterations_per_quick_update are printed to the notebook.
        """
    )
    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


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
        default=None,
    )

    parser.add_argument(
        "--iterations_per_quick_update",
        metavar="int",
        required=False,
        help="The number of iterations between each update",
        default=5000,
    )

    args = parser.parse_args()

    mask_radius = float(args.mask_radius) if args.mask_radius is not None else None

    iterations_per_quick_update = (
        int(args.iterations_per_quick_update)
        if args.iterations_per_quick_update is not None
        else 5000
    )

    """
    __Convert__

    Convert from command line inputs of strings to correct types depending on if command line inputs are given.

    If the mask radius input is not given, it is loaded from the dataset's info.json file in the `fit` function
    or uses the default value of 3.0" if this is not available.
    """
    vis_result = fit(
        dataset_name=args.dataset,
        mask_radius=mask_radius,
        iterations_per_quick_update=iterations_per_quick_update,
    )

    sersic_result = fit_sersic(
        dataset_name=args.dataset,
        vis_result=vis_result,
        mask_radius=mask_radius,
        iterations_per_quick_update=iterations_per_quick_update,
    )

    fit_waveband(
        dataset_name=args.dataset,
        unique_tag="sersic_lens_model",
        vis_result=sersic_result,
        use_sersic_over_sampling=True,
        mask_radius=mask_radius,
        iterations_per_quick_update=iterations_per_quick_update,
    )

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


def fit(
    dataset_name: str,
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

    Define the dataset path, assumed to follow:

        dataset/<dataset_name>/<dataset_name>.fits

    That is, the dataset lives in a subfolder named after it, and the main FITS file shares the same name.
    """
    dataset_main_path = Path("dataset") / dataset_name
    dataset_fits_name = f"{dataset_name}.fits"

    """
    __Dataset Wavebands__

    The following dictionary gives the names of the wavebands we are going to fit and maps them to their
    hdu in the FITS file. 

    It is created by inspecing the .fits headers of every hdu and extracting the waveband name from the header,
    mapping it to the HDU index.

    In this pipeline, we fit the VIS waveband, which is the highest quality data and therefore best suited to
    initializing a robust lens model.
    """
    dataset_index_dict = util.dataset_instrument_hdu_dict_via_fits_from(
        dataset_path=dataset_main_path,
        dataset_fits_name=dataset_fits_name,
        image_tag="_BGSUB",  # Depends on how Euclid cutout was made
    )

    dataset_waveband = "vis"

    vis_index = dataset_index_dict[dataset_waveband]

    """
    __Dataset__
        
    We begin by loading the dataset. Three ingredients are needed for lens modeling:
    
    1. The image itself (CCD counts).
    2. A noise-map (per-pixel RMS noise).
    3. The PSF (Point Spread Function).
    
    The `pixel_scales` value converts pixel units into arcseconds, which for Euclid VIS is 0.1" per pixel.
    """
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

    """
    Extract the brightest central pixel of the lens light from the data, which is used to 
    initializes priors on the lens model centre and other aspects of the fit,
    """
    dataset_centre = dataset.data.brightest_sub_pixel_coordinate_in_region_from(
        region=(-0.3, 0.3, -0.3, 0.3), box_size=2
    )

    """
    __Info__
    
    An `info.json` file can be used to store metadata about the dataset, which is passed
    through the pipeline and can be used in subsequent interpretation.
    """
    try:
        with open(dataset_main_path / "info.json") as json_file:
            info = json.load(json_file)
            json_file.close()
    except FileNotFoundError:
        info = {}

    """
    __Header__
    
    Load the .fits header of the Euclid VIS data, which will contain many useful pieces of metadata,
    
    We specifically extract the magnitude zero-point, which is used to convert between flux and magnitudes.
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
    __WCS__
    
    The World Coordinate System (WCS) provides a standard framework for converting between pixel coordinates in image 
    data and sky coordinates (e.g. right ascension and declination). Because WCS uses a unified coordinate convention 
    adopted by all major telescopes and instruments, it enables results derived from one dataset to be compared 
    directly with, or followed up by, observations from other facilities.
    
    The image header contains the metadata required to map pixel coordinates to sky coordinates via the 
    WCS transformation. This metadata is configured below.
    
    WCS information is propagated through the lens-modeling pipeline so that inferred quantities—such as the centre 
    of the lens light model—can be converted from AutoLens arcsecond-based coordinates into absolute sky coordinates 
    once modeling is complete. The lens model is also used to determine the sky coordinates of each multiple image of
    the lensed source.
    
    All WCS information is written to the lens `output` directory as `wcs.json` and can be further inspected or 
    manipulated using the catalogue-generation tools provided in the `workflow` module.
    """
    from astropy.wcs import WCS

    pixel_wcs = WCS(header).celestial

    """
    __Extra Galaxy Removal__
    
    There may be regions of an image that have signal near the lens and source that is from other galaxies not associated
    with the strong lens we are studying. The emission from these images will impact our model fitting and needs to be
    removed from the analysis.
    
    This `mask_extra_galaxies` is used to prevent them from impacting a fit by scaling the RMS noise map values to
    large values. This mask may also include emission from objects which are not technically galaxies,
    but blend with the galaxy we are studying in a similar way. Common examples of such objects are foreground stars
    or emission due to the data reduction process.
    
    In this example, the noise is scaled over all regions of the image, even those quite far away from the strong lens
    in the centre. We are next going to apply a 2.5" circular mask which means we only analyse the central region of
    the image. It only in these central regions where for the actual lens analysis it matters that we scaled the noise.
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
    
    Lens modeling does not need to fit the entire image, only the region containing lens and
    source light. We therefore define a circular mask around the lens.
    
    - Make sure the mask fully encloses the lensed arcs and the lens galaxy.
    - Avoid masking too much empty sky, as this slows fitting without adding information.
    
    We’ll also oversample the central pixels, which improves modeling accuracy without adding
    unnecessary cost far from the lens.
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

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.1, 0.3],
        centre_list=[dataset_centre],
    )

    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

    """
    __Positions__
    
    We can optionally add a penalty term ot the likelihood function, which penalizes models where the brightest 
    multiple images of the lensed source galaxy do not trace close to one another in the source plane. This 
    removes "demagnified source solutions" from the source pixelization, which one is likely to infer without this 
    penalty.
    
    A comprehensive description of why we do this is given at the following readthedocs page. I strongly recommend you 
    read this page in full if you are not familiar with the positions likelihood penalty and demagnified source 
    reconstructions:
    
     https://pyautolens.readthedocs.io/en/latest/general/demagnified_solutions.html
    """
    try:
        positions = al.Grid2DIrregular(
            values=al.from_json(file_path=dataset_main_path / "positions.json")
        )
        positions_likelihood_list = [al.PositionsLH(threshold=0.1, positions=positions)]
    except FileNotFoundError:
        positions_likelihood_list = None

    """
    __Lowest Resolution PSF__

    To perform aperture photometry for photometric redshift estimation, all imaging must be
    matched to the lowest-resolution PSF across the MER bands for a given strong lens.

    The strong-lens FITS cutout pipeline stores in its header the name of the worst band,
    for example "DES_G". The code below finds this header entry, extracts the name of 
    the worst band, and loads the PSF for this band from the .fits file. This PSF is 
    passed to the `AnalysisImaging` object. 

    To perform aperture photometry, the images of the lens and source are convolved with 
    this PSF. The flux within circular apertures is then comupted. The radius of these
    apertures is set as mutiples of the FWHM of the worst-seeing PSF loaded. This is also
    stored in the .fits header, loaded, and passed to the `AnalysisImaging` object. 

    The fluxes computed after lens modeling via aperture photometry are in microJanskys and are suitable 
    for direct input into an SED fitting code. This uses lens models sampled from the posterior 
    distribution and output as latent variables. As a result, the fluxes provided to the SED fitting code
    fully propagate uncertainties in the lens model.
    """
    header_primary = al.header_obj_from(
        file_path=dataset_main_path / dataset_fits_name,
        hdu=0,
    )

    lowest_resolution_waveband = header_primary.get("WORST_BAND", None).lower()

    lowest_resolution_waveband_index = dataset_index_dict.get(lowest_resolution_waveband, None)

    psf_lowest_resolution = al.Kernel2D.from_fits(
        file_path=dataset_main_path / dataset_fits_name,
        hdu=lowest_resolution_waveband_index * 3 + 2,
        pixel_scales=0.1,
        normalize=True
    )

    # Use OU-MER worst PSF FWHM if available, but if its -99 meaning the OU-MER pipeline failed used a
    # fall back value computed during lens cutout creation.
    psf_lowest_resolution_fwhm = float(header_primary.get("WORST_PSF_MER", None))

    if psf_lowest_resolution_fwhm is None or psf_lowest_resolution_fwhm < -98:
        psf_lowest_resolution_fwhm = float(header_primary.get("WORST_PSF_FWHM", None))

    """
    __Settings AutoFit__

    The settings of autofit, which controls the output paths, parallelization, database use, etc.
    """
    dataset_waveband = "vis"

    settings_search = af.SettingsSearch(
        path_prefix=Path(dataset_name),
        unique_tag="initial_lens_model",
        info={"magzero": magzero},
        session=None,
    )

    """
    __Redshifts__

    The redshifts of the lens and source galaxies.
    
    For a single plane strong lens, PyAutoLens units are dimensionless, meaning redshifts do not change lens
    modeling results. Therefore, the values input below will not change the lens model results at all.
    
    These effectively act as placeholders for now, with redshift information in Euclid coming after lens modeling
    via photometric redshhift fitting of the lens and source galaxies.
    """
    redshift_lens = 0.5
    redshift_source = 1.0

    """
    __Model: MGE Lens Model__

    This pipeline fits the lens and source galaxy light together with the lens mass distribution:

     - The lens light is represented 20 Gaussians [4 non-linear parameters].  
       Each Gaussian’s intensity is solved via linear inversion [60 linear parameters].  

     - The source light is represented by 20 Gaussians with [4 non-linear parameters].  
       Each Gaussian’s intensity is solved via linear inversion [20 linear parameters].  

     - The lens mass is modeled as an Isothermal Ellipsoid (SIE) with a cente fixed to the brighest pixel and an
        External Shear [5 non-linear parameters].

    Overall the model has 15 non-linear parameters, while most parameters are linear and solved efficiently at every 
    likelihood evaluation. This keeps the parameter space low-dimensional and well-conditioned, enabling efficient 
    sampling with a high probability of finding the global maximum.      
    """
    # Lens:

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
        centre=dataset_centre,
    )

    mass = af.Model(al.mp.Isothermal)

    mass.centre.centre_0 = dataset_centre[0]
    mass.centre.centre_1 = dataset_centre[1]

    shear = af.Model(al.mp.ExternalShear)

    lens = af.Model(
        al.Galaxy, redshift=redshift_lens, bulge=lens_bulge, mass=mass, shear=shear
    )

    # Source:

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    source = af.Model(al.Galaxy, redshift=redshift_source, bulge=source_bulge)

    # Overall Lens Model:

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    """
    __Model Fit__

    We now fit the data with the lens model using the non-linear fitting method and nested sampling algorithm Nautilus.

    This requires an `AnalysisImaging` object, which defines the `log_likelihood_function` used by Nautilus to fit
    the model to the imaging data.

    __JAX__

    PyAutoLens uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU
    support, your fits will run much faster (around 10 minutes instead of an hour). If only a CPU is available,
    JAX will still provide a speed up via multithreading, with fits taking around 20-30 minutes.

    If you don’t have a GPU locally, consider Google Colab which provides free GPUs, so your modeling runs are much faster.
    """
    analysis = util.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=positions_likelihood_list,
        use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
        dataset_main_path=dataset_main_path,
        title_prefix=dataset_waveband.upper(),
        plot_rgb=True,
        psf_lowest_resolution=psf_lowest_resolution,
        psf_lowest_resolution_fwhm=psf_lowest_resolution_fwhm,
        pixel_wcs=pixel_wcs,
        **settings_search.info,
    )

    search = af.Nautilus(
        name=dataset_waveband,  # The name of the fit and folder results are output to.
        **settings_search.search_dict,
        n_live=100,  # The number of Nautilus "live" points, increase for more complex models.
        batch_size=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
        iterations_per_quick_update=iterations_per_quick_update,  # Every N iterations the max likelihood model is visualized in the Jupter Notebook and output to hard-disk.
        n_like_max=100000,  # The maximum number of likelihood evaluations, models typically take < 30000 samples so this stops runaway fits.
    )

    """
    __Model-Fit__

    We can now begin the model-fit by passing the model and analysis object to the search, which performs the 
    Nautilus non-linear search in order to find which models fit the data with the highest likelihood.
    
    **Run Time Error:** On certain operating systems (e.g. Windows, Linux) and Python versions, the code below may produce 
    an error. If this occurs, see the `autolens_workspace/guides/modeling/bug_fix` example for a fix.
    """
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

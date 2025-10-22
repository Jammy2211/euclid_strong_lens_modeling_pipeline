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
        dataset_path=dataset_main_path, dataset_fits_name=dataset_fits_name
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

    try:
        with open(dataset_main_path / "info.json") as json_file:
            info = json.load(json_file)
            json_file.close()
    except FileNotFoundError:
        info = {}

    try:
        mask_extra_galaxies = al.Mask2D.from_fits(
            file_path=f"{dataset_main_path}/mask_extra_galaxies.fits",
            pixel_scales=0.1,
            invert=True,
        )

        dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)
    except FileNotFoundError:
        pass

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
    """
    try:
        positions = al.Grid2DIrregular(
            values=al.from_json(file_path=dataset_main_path / "positions.json")
        )
        positions_likelihood_list = [al.PositionsLH(threshold=0.1, positions=positions)]
    except FileNotFoundError:
        positions_likelihood_list = None

    """
    __Settings AutoFit__

    The settings of autofit, which controls the output paths, parallelization, database use, etc.
    """
    dataset_waveband = "vis"

    settings_search = af.SettingsSearch(
        path_prefix=Path("mge_lens_model") / dataset_name,
        unique_tag=dataset_waveband,
        info=None,
        session=None,
    )

    """
    __Redshifts__

    The redshifts of the lens and source galaxies.
    """
    redshift_lens = 0.5
    redshift_source = 1.0

    """
    __Model: MGE Lens Model__

    This pipeline fits the lens and source galaxy light together with the lens mass distribution:

     - The lens light is represented by 2 sets of 30 Gaussians (60 total) [6 non-linear parameters].  
       Each Gaussian’s intensity is solved via linear inversion [60 linear parameters].  

     - The source light is represented by 1 set of 20 Gaussians with [4 non-linear parameters].  
       Each Gaussian’s intensity is solved via linear inversion [20 linear parameters].  

     - The lens mass is modeled as an Isothermal Ellipsoid (SIE) with an External Shear [7 non-linear parameters].

    Overall the model has 17 non-linear parameters, while most parameters are linear and solved efficiently at every 
    likelihood evaluation. This keeps the parameter space low-dimensional and well-conditioned, enabling efficient 
    sampling with a high probability of finding the global maximum.  
    """
    analysis = util.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=positions_likelihood_list,
        dataset_main_path=dataset_main_path,
    )

    # Lens:

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=dataset_centre,
    )

    mass = af.Model(al.mp.Isothermal)

    mass.centre.centre_0 =  af.UniformPrior(
        lower_limit=dataset_centre[0] - 0.05, upper_limit=dataset_centre[0] + 0.05
    )
    mass.centre.centre_1 =  af.UniformPrior(
        lower_limit=dataset_centre[1] - 0.05, upper_limit=dataset_centre[1] + 0.05
    )

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

    search = af.Nautilus(
        name="mge_lens_model",
        **settings_search.search_dict,
        n_live=100,
        batch_size=50,
        iterations_per_quick_update=iterations_per_quick_update,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def fit_waveband(
    dataset_name: str,
    vis_result,
    mask_radius: float = 3.0,
    iterations_per_quick_update: int = 5000,
):
    """
    The function below fits the same lens system as above, but using lower resolution data from a different
    waveband (e.g. NISP near-infrared imaging data or EXT ground based imaging data from DES).

    The mass model is fixed to the result of the high resolution VIS fit, and the lens light and source are
    fitted using the same approach as above.

    This script is therefore a demonstration of how to fit multi-wavelength data using the SLaM pipelines.
    """

    import util
    import json
    from pathlib import Path

    import autofit as af
    import autolens as al

    """
    __Configs__
    """
    from autoconf import conf

    conf.instance["visualize"]["general"]["units"][
        "cb_unit"
    ] = r"$\,\,\mathrm{e^{-}}\,\mathrm{s^{-1}}$"

    """
    __Dataset__ 

    Load, plot and mask the `Imaging` data.
    """
    dataset_main_path = Path("dataset") / dataset_name
    dataset_fits_name = f"{dataset_name}.fits"

    try:
        with open(dataset_main_path / "info.json") as json_file:
            info = json.load(json_file)
            json_file.close()
    except FileNotFoundError:
        info = {}

    """
    __Dataset Wavebands__

    The following dictionary gives the names of the wavebands we are going to fit and maps them to their
    hdu in the FITS file. 

    It is created by inspecing the .fits headers of every hdu and extracting the waveband name from the header,
    mapping it to the HDU index.

    The pixel scale of each waveband is assumed to be 0.1" as EXT data is sampler to the same resolution as VIS,
    if this is not true this will need to be updated.
    """
    dataset_index_dict = util.dataset_instrument_hdu_dict_via_fits_from(
        dataset_path=dataset_main_path, dataset_fits_name=dataset_fits_name
    )

    """
    __Dataset Model__

    For each fit, the (y,x) offset of the secondary data from the primary data is a free parameter. 

    This is achieved by setting up a `DatasetModel` for each waveband, which extends the model with components
    including the grid offset.

    This ensures that if the datasets are offset with respect to one another, the model can correct for this,
    with sub-pixel offsets often being important in lens modeling as the precision of a lens model can often be
    less than the requirements on astrometry.
    """
    dataset_model = af.Model(al.DatasetModel)

    dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(
        lower_limit=-0.2, upper_limit=0.2
    )
    dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(
        lower_limit=-0.2, upper_limit=0.2
    )

    for i in range(len(dataset_index_dict.keys())):

        dataset_waveband = list(dataset_index_dict.keys())[i]
        dataset_path = dataset_main_path
        dataset_fits_name = f"{dataset_name}.fits"

        if dataset_waveband == "vis":
            continue

        dataset_index = dataset_index_dict[dataset_waveband]

        dataset = al.Imaging.from_fits(
            data_path=dataset_path / dataset_fits_name,
            data_hdu=dataset_index * 3 + 1,
            noise_map_path=dataset_path / dataset_fits_name,
            noise_map_hdu=dataset_index * 3 + 3,
            psf_path=dataset_path / dataset_fits_name,
            psf_hdu=dataset_index * 3 + 2,
            pixel_scales=0.1,
            check_noise_map=False,
        )

        dataset_centre = dataset.data.brightest_sub_pixel_coordinate_in_region_from(
            region=(-0.3, 0.3, -0.3, 0.3), box_size=2
        )

        try:
            mask_extra_galaxies = al.Mask2D.from_fits(
                file_path=dataset_main_path / "mask_extra_galaxies.fits",
                pixel_scales=0.1,
                invert=True,
            )

            dataset = dataset.apply_noise_scaling(
                mask=mask_extra_galaxies,
            )
        except FileNotFoundError:
            pass

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
        __Settings AutoFit__
        """
        settings_search = af.SettingsSearch(
            path_prefix=Path("mge_lens_model") / dataset_name,
            unique_tag=dataset_waveband,
            info=None,
            session=None,
        )

        """
        __SOURCE LP PIPELINE (with lens light)__

        The SOURCE LP PIPELINE (with lens light) uses three searches to initialize a robust model for the 
        source galaxy's light, which in this example:

         - Uses a parametric `Sersic` bulge and `Exponential` disk with centres aligned for the lens
         galaxy's light.

         - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.

         __Settings__:

         - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS TOTAL PIPELINE).
        """
        analysis = util.AnalysisImaging(
            dataset=dataset,
            dataset_main_path=dataset_main_path,
        )

        """
        __Dataset Model__

        For each fit, the (y,x) offset of the secondary data from the primary data is a free parameter. 

        This is achieved by setting up a `DatasetModel` for each waveband, which extends the model with components
        including the grid offset.

        This ensures that if the datasets are offset with respect to one another, the model can correct for this,
        with sub-pixel offsets often being important in lens modeling as the precision of a lens model can often be
        less than the requirements on astrometry.
        """
        dataset_model = af.Model(al.DatasetModel)

        dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(
            lower_limit=-0.2, upper_limit=0.2
        )
        dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(
            lower_limit=-0.2, upper_limit=0.2
        )

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=vis_result.instance.galaxies.lens.redshift,
                    bulge=vis_result.instance.galaxies.lens.bulge,
                    mass=vis_result.instance.galaxies.lens.mass,
                    shear=vis_result.instance.galaxies.lens.shear,
                ),
                source=af.Model(
                    al.Galaxy,
                    redshift=vis_result.instance.galaxies.source.redshift,
                    bulge=vis_result.instance.galaxies.source.bulge,
                ),
            ),
            dataset_model=dataset_model,
        )

        """
        __Search__
        """
        search = af.Nautilus(
            name="mge_lens_model",
            **settings_search.search_dict,
            n_live=75,
            batch_size=50,
            iterations_per_quick_update=iterations_per_quick_update,
        )

        search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


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

    fit_waveband(
        dataset_name=args.dataset,
        vis_result=vis_result,
        mask_radius=mask_radius,
        iterations_per_quick_update=iterations_per_quick_update,
    )

"""
Modeling Features: Multi Gaussian Expansion
===========================================

A multi Gaussian expansion (MGE) decomposes the lens light into ~15-100 Gaussians, where the `intensity` of every
Gaussian is solved for via a linear algebra using a process called an "inversion" (see the `linear_light_profiles.py`
feature for a full description of this).

This script fits the MGE to a strong lens image, but does not fit the mass model or source light. Analysis of JWST
data has shown this is a relatively quick way to get a clean lens light subtraction, which can be used to visually
inspect the lensed source emission.

A full description of MGE can be found at the following link:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/features/multi_gaussian_expansion.ipynb
"""

from start_here import fit


def fit_waveband(
    dataset_name: str,
    unique_tag,
    vis_result,
    use_sersic_over_sampling: bool = False,
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

    import numpy as np
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
        dataset_path=dataset_main_path,
        dataset_fits_name=dataset_fits_name,
        image_tag="_BGSUB",  # Depends on how Euclid cutout was made
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

        """
        __Dataset__
        """
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
                hdu=dataset_index * 3 + 1,
            )
            magzero = header["MAGZERO"]
        except FileNotFoundError:
            magzero = None

        """
        __WCS__
        """
        from astropy.wcs import WCS

        pixel_wcs = WCS(header).celestial

        """
        __Extra Galaxy Removal__
        """
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
        """
        if not use_sersic_over_sampling:

            over_sample_size = (
                al.util.over_sample.over_sample_size_via_radial_bins_from(
                    grid=dataset.grid,
                    sub_size_list=[4, 2, 1],
                    radial_list=[0.1, 0.3],
                    centre_list=[dataset_centre],
                )
            )

            dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

        else:

            tracer = vis_result.max_log_likelihood_tracer

            traced_grid = tracer.traced_grid_2d_list_from(
                grid=dataset.grid,
            )[-1]

            source_centre = tracer.galaxies[1].bulge.centre

            over_sample_size = (
                al.util.over_sample.over_sample_size_via_radial_bins_from(
                    grid=traced_grid,
                    sub_size_list=[16, 4, 2],
                    radial_list=[0.1, 0.3],
                    centre_list=[source_centre],
                )
            )

            over_sample_size_lens = (
                al.util.over_sample.over_sample_size_via_radial_bins_from(
                    grid=dataset.grid,
                    sub_size_list=[16, 4, 1],
                    radial_list=[0.1, 0.3],
                    centre_list=[dataset_centre],
                )
            )

            over_sample_size = np.where(
                over_sample_size > over_sample_size_lens,
                over_sample_size,
                over_sample_size_lens,
            )
            over_sample_size = al.Array2D(values=over_sample_size, mask=mask)

            dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

        """
        __Lowest Resolution PSF__
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
        """
        settings_search = af.SettingsSearch(
            path_prefix=Path(dataset_name),
            unique_tag=unique_tag,
            info={"magzero": magzero},
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
            use_jax=True,
            title_prefix=dataset_waveband.upper(),
            **settings_search.info,
            dataset_main_path=dataset_main_path,
            skip_rgb_plot=True,
            psf_lowest_resolution=psf_lowest_resolution,
            psf_lowest_resolution_fwhm=psf_lowest_resolution_fwhm,
            pixel_wcs=pixel_wcs,
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
            name=dataset_waveband,  # The name of the fit and folder results are output to.
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
        unique_tag="initial_lens_model",
        vis_result=vis_result,
        mask_radius=mask_radius,
        iterations_per_quick_update=iterations_per_quick_update,
    )

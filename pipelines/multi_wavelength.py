"""
Euclid Pipeline: Multi Wavelength
=================================

This example shows how to use the Euclid pipeline to fit a lens with the high resolution VIS optical imaging data,
and then fit lower resolution data from different wavelengths (e.g. NISP near-infrared imaging data and EXT
ground based imaging data for example from DES.

The multi wavelength data is lower resolution and quality than the high resolution data, therefore the model-fit
fits the VIS imaging data with full complexity and then fits each other dataset with the following approach:

- The mass model (e.g. SIE +Shear) is fixed to the result of the VIS fit.

- The lens light (Multi Gaussian Expansion) has the `intensity` values of the Gaussians updated using linear algebra.
  to capture changes in the lens light over wavelength, but it does not update the Gaussian parameters (e.g. `centre`,
 `elliptical_comps`, `sigma`) themselves due to the lower resolution of the data.

- The source reconstruction (Delaunay adaptive mesh) is updated using linear algebra to reconstruct the source, but again fixes
  the source pixelization parameters themselves.

- Sub-pixel offsets between the datasets are fully modeled as free parameters, because the precision of a lens model
can often be less than the requirements on astrometry.

The first fit, performed to the VIS data, is identical to the `start_here.py` script, you should therefore familiarize
yourself with that script before reading this one.

The subsequent fits to the lower resolution data use a reduced and simplified SLaM pipeline with the mass model
fixed to the result of the VIS fit.

__This Script__

Using a SOURCE LP PIPELINE, SOURCE PIX PIPELINE, LIGHT LP PIPELINE and TOTAL MASS PIPELINE this SLaM modeling
script  fits `Imaging` dataset  of a strong lens system where in the final model:

 - The lens galaxy's light is a bulge with Multiple Gaussian Expansion (MGE) light profile.
 - The lens galaxy's total mass distribution is an `PowerLaw` plus an `ExternalShear`.
 - The source galaxy's light is a `Pixelization`.

This modeling script uses the SLaM pipelines:

 `source_lp`
 `source_pix`
 `light_lp`
 `mass_total`

__Start Here Notebook__

If any code in this script is unclear, refer to the `chaining/start_here.ipynb` notebook.
"""

"""
Everything below is identical to `start_here.py` and thus not commented, as it is the same code.
"""

def fit(
        dataset_name: str,
        mask_radius: float = 3.0,
        iterations_per_quick_update: int = 5000,
):

    import util
    import os
    from pathlib import Path
    import json
    import numpy as np
    import sys

    import autofit as af
    import autolens as al

    sys.path.insert(0, os.getcwd())
    import slam

    """
    __Dataset__ 

    Load, plot and mask the `Imaging` data.

    __VIS Index__

    The `vis_index` parameter is key to ensuring the VIS dataset is fitted in the `fit` pipeline.

    It corresponds to the hdu index of the VIS imaging data in your .fits dataset, but is also used to load
    the PSF and noise-map data from the dataset folder of the lens you're modeling, as seen for
    the `Imaging.from_fits` method below.

    For the majority of strong lens MER cutouts, the vis_index will be 0 because the image is in hdu 1, the PSF in hdu 2
    and the noise-map in hdu 3.

    MER cutouts including EXT data may not conform to this convention, however, so always be sure to check the
    .fits files of the dataset you're using to make sure the vis_index is correct!
    """
    dataset_main_path = Path("dataset") / dataset_name
    dataset_fits_name = f"{dataset_name}.fits"

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
        header = al.header_obj_from(
            file_path=dataset_main_path / dataset_fits_name,
            hdu=vis_index * 3 + 1,
        )
        zero_point = header["MAGZERO"]
    except FileNotFoundError:
        zero_point = None

    try:
        mask_extra_galaxies = al.Mask2D.from_fits(
            file_path=dataset_main_path / "mask_extra_galaxies.fits",
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
        path_prefix=Path("slam") / dataset_name,
        unique_tag=dataset_waveband,
        info={"zero_point": zero_point},
        session=None,
    )

    """
    __Redshifts__

    The redshifts of the lens and source galaxies.

    These are placeholders for now given we probably don't know the redshifts of the lens and source galaxies,
    but manually input via command line can be added.
    """
    redshift_lens = 0.5
    redshift_source = 1.0

    """
    __SOURCE LP PIPELINE__

    The SOURCE LP PIPELINE uses one search to initialize a robust model for the source galaxy's light, which in 
    this example:

     - Uses a multi Gaussian expansion with 2 sets of 30 Gaussians for the lens galaxy's light.

     - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.

     - Uses a multi Gaussian expansion with 1 set of 20 Gaussians for the source galaxy's light.

     __Settings__:

     - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS TOTAL PIPELINE).
    """
    analysis = util.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=positions_likelihood_list,
        dataset_main_path=dataset_main_path,
    )

    # Lens Light

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=dataset_centre,
    )

    mass = af.Model(al.mp.Isothermal)

    mass.centre.centre_0 = af.UniformPrior(
        lower_limit=dataset_centre[0] - 0.05, upper_limit=dataset_centre[0] + 0.05
    )
    mass.centre.centre_1 = af.UniformPrior(
        lower_limit=dataset_centre[1] - 0.05, upper_limit=dataset_centre[1] + 0.05
    )

    # Source:

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    source_lp_result = slam.source_lp.run(
        settings_search=settings_search,
        analysis=analysis,
        lens_bulge=lens_bulge,
        lens_disk=None,
        mass=mass,
        shear=af.Model(al.mp.ExternalShear),
        source_bulge=source_bulge,
        mass_centre=dataset_centre,
        redshift_lens=redshift_lens,
        redshift_source=redshift_source,
    )

    """
    __JAX & Preloads__

    In JAX, calculations must use static shaped arrays with known and fixed indexes. For certain calculations in the
    pixelization, this information has to be passed in before the pixelization is performed. Below, we do this for 3
    inputs:

    - `total_linear_light_profiles`: The number of linear light profiles in the model. This is 0 because we are not
      fitting any linear light profiles to the data, primarily because the lens light is omitted.

    - `total_mapper_pixels`: The number of source pixels in the rectangular pixelization mesh. This is required to set up 
      the arrays that perform the linear algebra of the pixelization.

    - `source_pixel_zeroed_indices`: The indices of source pixels on its edge, which when the source is reconstructed 
      are forced to values of zero, a technique tests have shown are required to give accruate lens models.

    The `image_mesh` can be ignored, it is legacy API from previous versions which may or may not be reintegrated in future
    versions.
    """
    image_mesh = None
    mesh_shape = (20, 20)
    total_mapper_pixels = mesh_shape[0] * mesh_shape[1]

    total_linear_light_profiles = 20

    preloads = al.Preloads(
        mapper_indices=al.mapper_indices_from(
            total_linear_light_profiles=total_linear_light_profiles,
            total_mapper_pixels=total_mapper_pixels
        ),
        source_pixel_zeroed_indices=al.util.mesh.rectangular_edge_pixel_list_from(
            total_linear_light_profiles=total_linear_light_profiles,
            shape_native=mesh_shape,
        ),
    )

    """
    __SOURCE PIX PIPELINE__

    The SOURCE PIX PIPELINE uses two searches to initialize a robust model for the `Pixelization` that
    reconstructs the source galaxy's light. 

    This pixelization adapts its source pixels to the morphology of the source, placing more pixels in its 
    brightest regions. To do this, an "adapt image" is required, which is the lens light subtracted image meaning
    only the lensed source emission is present.

    The SOURCE LP Pipeline result is not good enough quality to set up this adapt image (e.g. the source
    may be more complex than a simple light profile). The first step of the SOURCE PIX PIPELINE therefore fits a new
    model using a pixelization to create this adapt image.

    The first search, which is an initialization search, fits an `Overlay` image-mesh, `Delaunay` mesh 
    and `AdaptiveBrightnessSplit` regularization.

    __Adapt Images / Image Mesh Settings__

    If you are unclear what the `adapt_images` and `SettingsInversion` inputs are doing below, refer to the 
    `autolens_workspace/*/imaging/advanced/chaining/pix_adapt/start_here.py` example script.

    __Settings__:

     - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
     in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
    """
    analysis = util.AnalysisImaging(
        dataset=dataset,
        adapt_image_maker=al.AdaptImageMaker(result=source_lp_result),
        positions_likelihood_list=[
            source_lp_result.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
        preloads=preloads,
        dataset_main_path=dataset_main_path,
    )

    source_pix_result_1 = slam.source_pix.run_1(
        settings_search=settings_search,
        analysis=analysis,
        source_lp_result=source_lp_result,
        mesh_init=af.Model(al.mesh.Rectangular, shape=mesh_shape),
        regularization_init=af.Model(al.reg.Constant),
        image_mesh_init=image_mesh,
    )

    """
    __SOURCE PIX PIPELINE 2 (with lens light)__

    The second search, which uses the mesh and regularization used throughout the remainder of the SLaM pipelines,
    fits the following model:

    - Uses a `Hilbert` image-mesh. 

    - Uses a `Delaunay` mesh.

     - Uses an `AdaptiveBrightnessSplit` regularization.

     - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the
     SOURCE PIX PIPELINE.

    The `Hilbert` image-mesh and `AdaptiveBrightness` regularization adapt the source pixels and regularization weights
    to the source's morphology.

    Below, we therefore set up the adapt image using this result.
    """
    analysis = util.AnalysisImaging(
        dataset=dataset,
        adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
        preloads=preloads,
        dataset_main_path=dataset_main_path,
    )

    source_pix_result_2 = slam.source_pix.run_2(
        settings_search=settings_search,
        analysis=analysis,
        source_lp_result=source_lp_result,
        source_pix_result_1=source_pix_result_1,
        image_mesh=image_mesh,
        mesh=af.Model(al.mesh.Rectangular, shape=mesh_shape),
        regularization=af.Model(al.reg.Constant),
    )

    """
    __LIGHT LP PIPELINE__

    The LIGHT LP PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
    lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE LP PIPELINE.
    In this example it:

     - Uses a multi Gaussian expansion with 2 sets of 30 Gaussians for the lens galaxy's light. [6 Free Parameters].

     - Uses an `Isothermal` mass model with `ExternalShear` for the lens's total mass distribution [fixed from SOURCE PIX PIPELINE].

     - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].

     - Carries the lens redshift and source redshift of the SOURCE PIPELINE through to the MASS PIPELINE [fixed values].   
    """
    analysis = util.AnalysisImaging(
        dataset=dataset,
        adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
        preloads=preloads,
        dataset_main_path=dataset_main_path,
    )

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=dataset_centre,
    )

    light_result = slam.light_lp.run(
        settings_search=settings_search,
        analysis=analysis,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        lens_bulge=lens_bulge,
        lens_disk=None,
    )

    """
    __MASS TOTAL PIPELINE__

    The MASS TOTAL PIPELINE uses one search to fits a complex lens mass model to a high level of accuracy, 
    using the lens mass model and source model of the SOURCE PIX PIPELINE to initialize the model priors and the lens 
    light model of the LIGHT LP PIPELINE. 

    In this example it:

     - Uses a linear Multi Gaussian Expansion bulge [fixed from LIGHT LP PIPELINE].

     - Uses an `PowerLaw` model for the lens's total mass distribution [priors initialized from SOURCE 
     PARAMETRIC PIPELINE + centre unfixed from (0.0, 0.0)].

     - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].

     - Carries the lens redshift and source redshift of the SOURCE PIPELINE through to the MASS TOTAL PIPELINE.

    __Settings__:

     - adapt: We may be using adapt features and therefore pass the result of the SOURCE PIX PIPELINE to use as the
     hyper dataset if required.

     - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
     in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
    """
    analysis = util.AnalysisImaging(
        dataset=dataset,
        adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
        positions_likelihood=source_pix_result_2.positions_likelihood_from(
            factor=3.0, minimum_threshold=0.2
        ),
        preloads=preloads,
        dataset_main_path=dataset_main_path,
    )

    mass_result = slam.mass_total.run(
        settings_search=settings_search,
        analysis=analysis,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        light_result=light_result,
        mass=af.Model(al.mp.Isothermal),
        reset_shear_prior=True,
    )

    return source_lp_result, mass_result


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
    import numpy as np
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
            header = al.header_obj_from(
                file_path=dataset_main_path / dataset_fits_name,
                hdu=dataset_index * 3 + 1,
            )
            zero_point = header["MAGZERO"]
        except FileNotFoundError:
            zero_point = None

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
            path_prefix=Path("slam") / dataset_name,
            unique_tag=dataset_waveband,
            info={"zero_point": zero_point},
            session=None,
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
        )

        source_bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
        )

        source_lp_result = slam.source_lp.run(
            settings_search=settings_search,
            analysis=analysis,
            lens_bulge=mass_result.instance.galaxies.lens.bulge,
            lens_disk=None,
            lens_point=mass_result.instance.galaxies.lens.point,
            mass=mass_result.instance.galaxies.lens.mass,
            shear=mass_result.instance.galaxies.lens.shear,
            source_bulge=source_bulge,
            redshift_lens=0.5,
            redshift_source=1.0,
            dataset_model=dataset_model,
        )

        """
        __SOURCE PIX PIPELINE (with lens light)__

        The SOURCE PIX PIPELINE (with lens light) uses four searches to initialize a robust model for the `Inversion` 
        that reconstructs the source galaxy's light. It begins by fitting a `VoronoiMagnification` pixelization with `Constant` 
        regularization, to set up the model and hyper images, and then:

         - Uses a `VoronoiBrightnessImage` pixelization.
         - Uses an `AdaptiveBrightness` regularization.
         - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the
         SOURCE PIX PIPELINE.
        """
        analysis = util.AnalysisImaging(
            dataset=dataset,
            adapt_image_maker=al.AdaptImageMaker(result=source_lp_result),
            raise_inversion_positions_likelihood_exception=False,
        )

        dataset_model.grid_offset.grid_offset_0 = (
            source_lp_result.instance.dataset_model.grid_offset[0]
        )
        dataset_model.grid_offset.grid_offset_1 = (
            source_lp_result.instance.dataset_model.grid_offset[1]
        )

        source_pix_result_1 = slam.source_pix.run_1(
            settings_search=settings_search,
            analysis=analysis,
            source_lp_result=source_lp_result,
            mesh_init=al.mesh.Delaunay,
            image_mesh_init_shape=(30, 30),
            dataset_model=dataset_model,
            fixed_mass_model=True,
        )

        source_pix_result_1.max_log_likelihood_fit.inversion.cls_list_from(
            cls=al.AbstractMapper
        )[0].extent_from()

        analysis = util.AnalysisImaging(
            dataset=dataset,
            adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
            settings_inversion=al.SettingsInversion(
                image_mesh_min_mesh_pixels_per_pixel=3,
                image_mesh_min_mesh_number=5,
                image_mesh_adapt_background_percent_threshold=0.1,
                image_mesh_adapt_background_percent_check=0.8,
            ),
        )

        dataset_model.grid_offset.grid_offset_0 = (
            source_lp_result.instance.dataset_model.grid_offset[0]
        )
        dataset_model.grid_offset.grid_offset_1 = (
            source_lp_result.instance.dataset_model.grid_offset[1]
        )

        multi_result = slam.source_pix.run_2(
            settings_search=settings_search,
            analysis=analysis,
            source_lp_result=source_lp_result,
            source_pix_result_1=source_pix_result_1,
            image_mesh=al.image_mesh.Hilbert,
            mesh=al.mesh.Delaunay,
            regularization=al.reg.AdaptiveBrightnessSplit,
            image_mesh_pixels_fixed=500,
            dataset_model=dataset_model,
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

    """
    __Convert__

    Convert from command line inputs of strings to correct types depending on if command line inputs are given.

    If the mask radius input is not given, it is loaded from the dataset's info.json file in the `fit` function
    or uses the default value of 3.0" if this is not available.
    """
    mask_radius = float(args.mask_radius) if args.mask_radius is not None else None
    iterations_per_quick_update = (
        int(args.iterations_per_quick_update)
        if args.iterations_per_quick_update is not None
        else 5000
    )

    source_lp_result, mass_result = fit(
        dataset_name=args.dataset,
        mask_radius=mask_radius,
        iterations_per_quick_update=iterations_per_quick_update,
    )

    fit_waveband(
        mass_result=mass_result,
        dataset_name=args.dataset,
        mask_radius=mask_radius,
        iterations_per_quick_update=iterations_per_quick_update,
    )

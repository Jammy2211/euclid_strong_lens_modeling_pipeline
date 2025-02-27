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
    number_of_cores: int = 1,
    iterations_per_update: int = 5000,
):
    import json
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
    """
    dataset_waveband = "vis"
    dataset_main_path = path.join("dataset", dataset_name)
    dataset_path = path.join(dataset_main_path, dataset_waveband)

    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        pixel_scales=0.1,
    )

    dataset_centre = dataset.data.brightest_sub_pixel_coordinate_in_region_from(
        region=(-0.3, 0.3, -0.3, 0.3), box_size=2
    )

    try:
        with open(path.join(dataset_main_path, "info.json")) as json_file:
            info = json.load(json_file)
            json_file.close()
    except FileNotFoundError:
        info = {}

    if mask_radius is None:
        mask_radius = info.get("mask_radius") or 3.0

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )

    dataset = dataset.apply_mask(mask=mask)

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.1, 0.3],
        centre_list=[dataset_centre],
    )

    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    """
    __Settings AutoFit__
    """
    settings_search = af.SettingsSearch(
        path_prefix=path.join("euclid_pipeline", dataset_name),
        unique_tag=dataset_waveband,
        info=None,
        number_of_cores=number_of_cores,
        session=None,
    )

    """
    __Redshifts__
    """
    redshift_lens = 0.5
    redshift_source = 1.0

    """
    __HPC Mode__

    When running in parallel via Python `multiprocessing`, display issues with the `matplotlib` backend can arise
    and cause the code to crash.

    HPC mode sets the backend to mitigate this issue and is set to run throughout the entire pipeline below.

    The `iterations_per_update` below specifies the number of iterations performed by the non-linear search between
    output, where visuals of the maximum log likelihood model, lens model parameter estimates and other information
    are output to hard-disk.

    There are a number of environment variables which must be set to ensure parallelization is efficient, which
    are set below in this script to ensure the pipeline always runs efficiently even if you have not manually set them.
    """
    from autoconf import conf

    conf.instance["general"]["hpc"]["hpc_mode"] = True
    conf.instance["general"]["hpc"]["iterations_per_update"] = iterations_per_update

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    """
    __SOURCE LP PIPELINE__
    """
    analysis = al.AnalysisImaging(dataset=dataset)

    # Lens Light

    total_gaussians = 30
    gaussian_per_basis = 2

    log10_sigma_list = np.linspace(-3, np.log10(mask_radius), total_gaussians)

    centre_0 = af.UniformPrior(lower_limit=dataset_centre[0] - 0.05, upper_limit=dataset_centre[0] + 0.05)
    centre_1 = af.UniformPrior(lower_limit=dataset_centre[1] - 0.05, upper_limit=dataset_centre[1] + 0.05)

    bulge_gaussian_list = []

    for j in range(gaussian_per_basis):
        ell_comps_0 = af.UniformPrior(lower_limit=-0.7, upper_limit=0.7)
        ell_comps_1 = af.UniformPrior(lower_limit=-0.7, upper_limit=0.7)

        gaussian_list = af.Collection(
            af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
        )

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0
            gaussian.centre.centre_1 = centre_1
            gaussian.ell_comps.ell_comps_0 = ell_comps_0
            gaussian.ell_comps.ell_comps_1 = ell_comps_1
            gaussian.sigma = 10 ** log10_sigma_list[i]

        bulge_gaussian_list += gaussian_list

    lens_bulge = af.Model(
        al.lp_basis.Basis,
        profile_list=bulge_gaussian_list,
    )

    # Source Light

    centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)

    total_gaussians = 20
    gaussian_per_basis = 1

    log10_sigma_list = np.linspace(-3, np.log10(1.0), total_gaussians)

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

    source_bulge = af.Model(
        al.lp_basis.Basis,
        profile_list=bulge_gaussian_list,
    )

    source_lp_result = slam.source_lp.run(
        settings_search=settings_search,
        analysis=analysis,
        lens_bulge=lens_bulge,
        lens_disk=None,
        mass=af.Model(al.mp.Isothermal),
        shear=af.Model(al.mp.ExternalShear),
        source_bulge=source_bulge,
        mass_centre=dataset_centre,
        redshift_lens=redshift_lens,
        redshift_source=redshift_source,
    )

    """
    __SOURCE PIX PIPELINE__
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_image_maker=al.AdaptImageMaker(result=source_lp_result),
        positions_likelihood=source_lp_result.positions_likelihood_from(
            factor=3.0, minimum_threshold=0.2
        ),
    )

    source_pix_result_1 = slam.source_pix.run_1(
        settings_search=settings_search,
        analysis=analysis,
        source_lp_result=source_lp_result,
        mesh_init=al.mesh.Delaunay,
        image_mesh_init_shape=(30, 30),
    )

    """
    __SOURCE PIX PIPELINE 2 (with lens light)__
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
        settings_inversion=al.SettingsInversion(
            image_mesh_min_mesh_pixels_per_pixel=3,
            image_mesh_min_mesh_number=5,
            image_mesh_adapt_background_percent_threshold=0.1,
            image_mesh_adapt_background_percent_check=0.8,
        ),
    )

    source_pix_result_2 = slam.source_pix.run_2(
        settings_search=settings_search,
        analysis=analysis,
        source_lp_result=source_lp_result,
        source_pix_result_1=source_pix_result_1,
        image_mesh=al.image_mesh.Hilbert,
        mesh=al.mesh.Delaunay,
        regularization=al.reg.AdaptiveBrightnessSplit,
        image_mesh_pixels_fixed=500,
    )

    """
    __LIGHT LP PIPELINE__  
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    )

    centre_0 = af.GaussianPrior(mean=dataset_centre[0], sigma=0.1)
    centre_1 = af.GaussianPrior(mean=dataset_centre[1], sigma=0.1)

    total_gaussians = 30
    gaussian_per_basis = 2

    log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

    bulge_gaussian_list = []

    for j in range(gaussian_per_basis):
        ell_comps_0 = af.UniformPrior(lower_limit=-0.7, upper_limit=0.7)
        ell_comps_1 = af.UniformPrior(lower_limit=-0.7, upper_limit=0.7)

        gaussian_list = af.Collection(
            af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
        )

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0
            gaussian.centre.centre_1 = centre_1
            gaussian.ell_comps.ell_comps_0 = ell_comps_0
            gaussian.ell_comps.ell_comps_1 = ell_comps_1
            gaussian.sigma = 10 ** log10_sigma_list[i]

        bulge_gaussian_list += gaussian_list

    lens_bulge = af.Model(
        al.lp_basis.Basis,
        profile_list=bulge_gaussian_list,
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
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
        positions_likelihood=source_pix_result_2.positions_likelihood_from(
            factor=3.0, minimum_threshold=0.2
        ),
    )

    mass_result = slam.mass_total.run(
        settings_search=settings_search,
        analysis=analysis,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        light_result=light_result,
        mass=af.Model(al.mp.Isothermal),
        reset_shear_prior=True
    )

    """
    __Output__
    """
    slam.slam_util.update_result_json_file(
        file_path=path.join(dataset_main_path, "result.json"),
        result=mass_result,
        waveband=dataset_waveband,
        einstein_radius=True,
        fluxes=True,
    )

    slam.slam_util.output_result_to_fits(
        output_path=path.join(dataset_path, "result"),
        result=source_lp_result,
        model_lens_light=True,
        model_source_light=True,
        mge_source_reconstruction=True,
        prefix="mge_",
        tag=dataset_waveband,
        remove_fits_first=True,
    )

    slam.slam_util.output_result_to_fits(
        output_path=path.join(dataset_path, "result"),
        result=mass_result,
        model_lens_light=True,
        model_source_light=True,
        source_reconstruction=True,
        source_reconstruction_noise_map=True,
        remove_fits_first=True,
        tag=dataset_waveband,
    )

    slam.slam_util.output_model_results(
        output_path=path.join(dataset_path, "result"),
        result=mass_result,
        filename="sie_model_results.txt",
    )

    slam.slam_util.output_fit_multi_png(
        output_path=path.join(dataset_path, "result"),
        result_list=[mass_result],
        filename="sie_fit_pix",
        tag_prefix=f"{dataset_waveband}{dataset_name}"
    )

    slam.slam_util.output_source_multi_png(
        output_path=path.join(dataset_path, "result"),
        result_list=[mass_result],
        filename="source_reconstruction",
        tag_prefix=f"{dataset_waveband}{dataset_name}"
    )

    return source_lp_result, mass_result


def fit_waveband(
    dataset_name: str,
    source_lp_result,
    mass_result,
    mask_radius: float = 3.0,
    number_of_cores: int = 1,
    iterations_per_update: int = 5000,
):
    """
    The function below fits the same lens system as above, but using lower resolution data from a different
    waveband (e.g. NISP near-infrared imaging data or EXT ground based imaging data from DES).

    The mass model is fixed to the result of the high resolution VIS fit, and the lens light and source are
    fitted using the same approach as above.

    This script is therefore a demonstration of how to fit multi-wavelength data using the SLaM pipelines.
    """

    # %matplotlib inline
    # from pyprojroot import here
    # workspace_path = str(here())
    # %cd $workspace_path
    # print(f"Working Directory has been set to `{workspace_path}`")

    import json
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
    __HPC Mode__

    When running in parallel via Python `multiprocessing`, display issues with the `matplotlib` backend can arise
    and cause the code to crash.

    HPC mode sets the backend to mitigate this issue and is set to run throughout the entire pipeline below.

    The `iterations_per_update` below specifies the number of iterations performed by the non-linear search between
    output, where visuals of the maximum log likelihood model, lens model parameter estimates and other information
    are output to hard-disk.

    There are a number of environment variables which must be set to ensure parallelization is efficient, which
    are set below in this script to ensure the pipeline always runs efficiently even if you have not manually set them.
    """
    from autoconf import conf

    conf.instance["general"]["hpc"]["hpc_mode"] = True
    conf.instance["general"]["hpc"]["iterations_per_update"] = iterations_per_update

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    """
    __Configs__
    """
    from autoconf import conf

    conf.instance["visualize"]["general"]["units"][
        "cb_unit"
    ] = r"$\,\,\mathrm{e^{-}}\,\mathrm{s^{-1}}$"

    from euclid import slam

    """
    __Dataset__

    Usual API to set up dataset paths, but include its "main` path which is before the waveband folders.
    """
    dataset_waveband = "vis"
    dataset_main_path = path.join("dataset", dataset_name)
    dataset_path = path.join(dataset_main_path, dataset_waveband)

    """
    __Dataset Wavebands__

    The following dictionary gives the names of the wavebands we are going to fit and maps them to their
    hdu in the FITS file. 
    
    The standard MER cutout goes VIS, NIR_Y, NIR_J, NIR_H, and is used in this pipeline.
    
    The send dictionary, which is commented out, is an example of a dataset included EXT data from ground based
    telescopes as well, which is often used in Euclid pipelines.
    
    The data for each waveband is loaded from a folder in the dataset folder with that name, where the vis
    datasets fitted above is removed from the list.

    The pixel scale of each waveband is assumed to be 0.1" as EXT data is sampler to the same resolution as VIS,
    if this is not true this will need to be updated.
    """
    dataset_index_dict = {"vis" : 0, "nir_y" : 1, "nir_j" : 2, "nir_h" : 3}
    # dataset_index_dict = {"meg_u" : 0, "hsc_g" : 1, "meg_r" : 2, "vis" : 3, "hsc_z" : 4, "nir_y" : 5, "nir_j" : 6, "nir_h" : 7}

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
    __Result Dict__

    Visualization at the end of the pipeline will output all fits to all wavebands on a single matplotlib subplot.

    The results of each fit are stored in a dictionary, which is used to pass the results of each fit to the
    visualization functions.
    """
    multi_result_lp_dict = {"vis" : source_lp_result}
    multi_result_pix_dict = {"vis": mass_result}

    for i in range(len(dataset_index_dict.keys())):

        dataset_waveband = list(dataset_index_dict.keys())[i]
        dataset_main_path = path.join("dataset", dataset_name)
        dataset_path = path.join(dataset_main_path, dataset_waveband)
        dataset_fits_name = f"{dataset_name}.fits"

        if dataset_waveband == "vis":
            continue

        dataset_index = dataset_index_dict[dataset_waveband]

        dataset = al.Imaging.from_fits(
            data_path=path.join(dataset_path, dataset_fits_name),
            data_hdu=dataset_index*3+1,
            noise_map_path=path.join(dataset_path, dataset_fits_name),
            noise_map_hdu=dataset_index*3+3,
            psf_path=path.join(dataset_path, dataset_fits_name),
            psf_hdu=dataset_index*3+2,
            pixel_scales=0.1,
            check_noise_map=False,
        )

        dataset_centre = dataset.data.brightest_sub_pixel_coordinate_in_region_from(
            region=(-0.3, 0.3, -0.3, 0.3), box_size=2
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

        mask = al.Mask2D.circular(
            shape_native=dataset.shape_native,
            pixel_scales=dataset.pixel_scales,
            radius=mask_radius,
        )

        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=dataset.grid,
            sub_size_list=[4, 2, 1],
            radial_list=[0.1, 0.3],
            centre_list=[dataset_centre],
        )

        dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

        dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
        dataset_plotter.subplot_dataset()

        """
        __Settings AutoFit__
        """
        settings_search = af.SettingsSearch(
            path_prefix=path.join("euclid_pipeline", dataset_name),
            unique_tag=dataset_waveband,
            info=None,
            number_of_cores=number_of_cores,
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
        analysis = al.AnalysisImaging(
            dataset=dataset,
        )

        centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
        centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)

        total_gaussians = 20
        gaussian_per_basis = 1

        log10_sigma_list = np.linspace(-3, np.log10(1.0), total_gaussians)

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

        source_bulge = af.Model(
            al.lp_basis.Basis,
            profile_list=bulge_gaussian_list,
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

        multi_result_lp_dict[dataset_waveband] = source_lp_result

        slam.slam_util.output_result_to_fits(
            output_path=path.join(dataset_path, "result"),
            result=source_lp_result,
            model_lens_light=True,
            model_source_light=True,
            mge_source_reconstruction=True,
            tag=dataset_waveband,
            prefix="mge_",
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
        analysis = al.AnalysisImaging(
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

        analysis = al.AnalysisImaging(
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

        multi_result_pix_dict[dataset_waveband] = multi_result

        slam.slam_util.output_result_to_fits(
            output_path=path.join(dataset_path, "result"),
            result=multi_result,
            model_lens_light=True,
            model_source_light=True,
            source_reconstruction=True,
            source_reconstruction_noise_map=True,
            tag=dataset_waveband,
        )


    tag_list = list(multi_result_pix_dict.keys())

    slam.slam_util.output_fit_multi_png(
        output_path=path.join(dataset_path, "result"),
        result_list=[
            multi_result_lp_dict[dataset_waveband] for dataset_waveband in tag_list
        ],
        tag_list=[tag + dataset_name for tag in tag_list],
        filename="sie_fit_mge",
    )

    slam.slam_util.output_fit_multi_png(
        output_path=path.join(dataset_path, "result"),
        result_list=[
            multi_result_pix_dict[dataset_waveband] for dataset_waveband in tag_list
        ],
        tag_list=[tag + dataset_name for tag in tag_list],
        filename="sie_fit_pix",
    )

    slam.slam_util.output_source_multi_png(
        output_path=path.join(dataset_path, "result"),
        result_list=[
            multi_result_pix_dict[dataset_waveband] for dataset_waveband in tag_list
        ],
        tag_list=[tag + dataset_name for tag in tag_list],
        filename="source_reconstruction",
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
        default=None
    )

    parser.add_argument(
        "--number_of_cores",
        metavar="int",
        required=False,
        help="The number of cores to parallelize the fit",
        default=1
    )

    parser.add_argument(
        "--iterations_per_update",
        metavar="int",
        required=False,
        help="The number of iterations between each update",
        default=5000
    )

    args = parser.parse_args()

    """
    __Convert__

    Convert from command line inputs of strings to correct types depending on if command line inputs are given.

    If the mask radius input is not given, it is loaded from the dataset's info.json file in the `fit` function
    or uses the default value of 3.0" if this is not available.
    """
    mask_radius = float(args.mask_radius) if args.mask_radius is not None else None
    number_of_cores = int(args.number_of_cores) if args.number_of_cores is not None else 1
    iterations_per_update = int(args.iterations_per_update) if args.iterations_per_update is not None else 5000


    source_lp_result, mass_result = fit(
        dataset_name=args.dataset,
        mask_radius=mask_radius,
        number_of_cores=number_of_cores,
        iterations_per_update=iterations_per_update,
    )

    fit_waveband(
        source_lp_result=source_lp_result,
        mass_result=mass_result,
        dataset_name=args.dataset,
        mask_radius=mask_radius,
        number_of_cores=number_of_cores,
        iterations_per_update=iterations_per_update,
    )

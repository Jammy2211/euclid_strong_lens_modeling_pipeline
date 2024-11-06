"""
Euclid Pipeline: Start Here
===========================

This scripts allows you to run the Euclid lens modeling pipeline locally on your computer. It comes with an example
simulated Euclid strong lens dataset, which is fitted using the pipeline, and over 4 CPus should take ~3-5 hours to run.

The script itself is running **PyAutoLens**, which will require installation first via the instructions on the
GitHub page:

https://github.com/Jammy2211/euclid_strong_lens_modeling_pipeline

The pipeline can be run as a "black box", whereby you pass it the dataset you want it to fit and it automatically
fits it without understanding how the pipeline works.

The pipeline automatically outputs results and visualization to hard-disk, which if you are running it as a black
box is likely all you will look at. These include deblended lens and source images, a source-plane reconstruction and
lens model quantities like the Einstein Radius.

Please contact James Nightingale on the Euclid Consortium Slack with any questions or if you would like other
information on the pipeline.

The text below is taken from the **PyAutoLens** documentation and describes the pipeline in more detail. You do
not need to understand things like the prequisites, pipeline structure, design choices, etc in order to run
the pipeline as a black box, the text is simply there for your information.

__SLaM (Source, Light and Mass)__

This scripts gives an introduce to the Source, (lens) Light and Mass (SLaM) pipelines. These are advanced modeling
pipelines which use many aspects of core PyAutoLens functionality to automate the modeling of strong lenses.

__Preqrequisites__

Before reading this script, you should have familiarity with the following key concepts:

- **Non-linear Search Chaining:** This approach, demonstrated in `imaging/advanced/chaining`, shows the power of
  linking models together in a sequence, such as transitioning from a light profile source to a pixelized source.

- **Pixelizations:** These structures, explained in `features/pixelization.ipynb`, allow for the reconstruction of the
  source galaxy on a pixel grid.

- **Adaptive Pixelizations:** Described in `imaging/advanced/chaining/pix_adapt/start_here.ipynb`, these pixelizations
  adapt to the unlensed morphology of the source.

- **Multi Gaussian Expansions (MGE):** Introduced in `features/multi_gaussian_expansion.ipynb`, MGEs are employed to
  model the lens's light and can serve as an initialization for source galaxy light prior to using pixelization.

If any of these concepts are unfamiliar, you may still proceed with the script, but reviewing the referenced examples
later can deepen your understanding of how and why SLaM pipelines are structured as they are.

Additionally, this script allows for flexibility with model components, such as swapping out MGE models for other
light profiles (e.g., linear `Sersic` profiles). For an example, see `examples/source_light_profile.ipynb`.

__Overview__

The Source, (Lens) Light, and Mass (SLaM) pipelines strategically chain together up to five sequential searches,
carefully designed to maximize the advantages of search chaining. This setup provides a fully automated framework for
fitting large samples of strong lenses with complex models, making SLaM the default choice for many **PyAutoLens**
research publications.

__Pipeline Structure__

Each pipeline in the SLaM sequence targets a specific aspect of the strong lens model:

- **Source Pipeline**: The first step focuses on establishing a robust source model. For pixelized sources,
  this includes obtaining accurate values for mesh and regularization parameters. For sources modeled with light
  profiles, the focus is on determining initial parameter estimates.

- **Light Pipeline**: This stage focuses on modeling the lens light, using source and mass models fixed from previous pipelines.

- **Mass Pipeline**: The final stage develops a detailed mass model, potentially of high complexity, leveraging source
  and lens light models initialized from earlier stages.

Models set up in earlier pipelines guide those used in later ones. For instance, if the Source Pipeline uses a
pixelized `Delaunay` mesh for the source, that mesh type will carry through to the Mass Total Pipeline that follows.

__Design Choices__

There are many design choices that go into the SLaM pipelines, which we discuss now.

The SLaM pipelines are designed around pixelixed source modeling. Pixelized sources are necessary for fitting complex
mass models, which the SLaM pipelines automates the fitting of. However, the SLaM pipelines support fitting of
light profile sources, and using the SLaM pipelines in this way will still provide automated and robust lens modeling.

We now list the design considerations which dictate the ordering of the SLaM pipelines, which were driven by the use
of pixelized source modeling:

The SLaM pipelines involve several design choices to support the complexities of pixelized source modeling, which is
crucial for robustly fitting complex mass models. Here’s an overview of these key considerations and their influence
on the pipeline sequence:

- **Source First**: The pipeline starts with the Source Pipeline, as complex mass models (e.g., `PowerLaw` or
composite models with stars and dark matter) require pixelized source modeling rather than simple light profiles.
This step establishes a robust pixelized source model using a simpler mass model (like `Isothermal` with `Shear`).

- **Image Positions**: For pixelized source modeling, specifying the positions of the multiple images of the
lensed source(s) is crucial to prevent unphysical reconstructions. The SLaM pipelines can estimate these positions
automatically from the SOURCE LP PIPELINE's mass and source results.

- **Adapt Images**: Advanced pixelized source models use "adapt images" to optimize the mesh and regularization
  weights according to the source's morphology. The SLaM pipelines set the adapt-images once a good model for the source is
  available, enabling the best adaptation to the source structure.

- **Lens Light Before Mass**: Modeling the lens light accurately requires deblending the lens and source emissions,
  which a robust pixelized source model facilitates. This deblending, essential for certain mass models with both
  stellar and dark matter components, benefits from a simpler mass model during the lens light fitting stage.

- **Mass Model Last**: The most complex mass model fitting is saved for last. This final stage benefits from the
  prior refinement of the source and lens light models, ensuring accurate reconstructions and parameter estimations.

These design choices enable the SLaM pipelines to deliver precise and automated lens modeling while optimizing each
stage for robustness and efficiency.

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


def fit(dataset_name: str, mask_radius: float = 3.0, number_of_cores: int = 1):

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
    dataset_name = dataset_name
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
    
    These are placeholders for now given we probably don't know the redshifts of the lens and source galaxies,
    but amnually input via command line can be added.
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
        mass_centre=(0.0, 0.0),
        redshift_lens=redshift_lens,
        redshift_source=redshift_source,
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
        image_mesh_init_shape=(20, 20),
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
    
    The LIGHT LP PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
    lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE LP PIPELINE.
    In this example it:
    
     - Uses a multi Gaussian expansion with 2 sets of 20 Gaussians for the lens galaxy's light. [6 Free Parameters].
    
     - Uses an `Isothermal` mass model with `ExternalShear` for the lens's total mass distribution [fixed from SOURCE PIX PIPELINE].
    
     - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].
    
     - Carries the lens redshift and source redshift of the SOURCE PIPELINE through to the MASS PIPELINE [fixed values].   
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    )

    total_gaussians = 20
    gaussian_per_basis = 2

    log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

    bulge_gaussian_list = []

    for j in range(gaussian_per_basis):
        gaussian_list = af.Collection(
            af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
        )

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = gaussian_list[0].centre.centre_0
            gaussian.centre.centre_1 = gaussian_list[0].centre.centre_1
            gaussian.ell_comps = gaussian_list[0].ell_comps
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
    )

    """
    __Output__
    
    The SLaM pipeline above outputs the model-fitting results to the `output` folder of the workspace, which includes
    the usual model results, visualization, and .json files.
    
    As described in the `autolens_workspace/*/results` package there is an API for loading these results from hard disk
    to Python, for example so they can be manipulated in a Juypter notebook.
    
    However, it is also often useful to output the results to the dataset folder of each lens in standard formats, for
    example images of the lens and lensed source in .fits or visualization outputs like .png files. This makes transferring
    the results more portable, especially if they are to be used by other people.
    
    The `slam_util` module provides convenience methods for outputting many results to the dataset folder, we
    use it below to output the following results:
    
     - Images of the model lens light, lensed source light and source reconstruction to .fits files.
     - A text `model.results` file containing the lens model parameter estimates.
     - A subplot containing the fit in one row, which is output to .png.
     - A subplot of the source reconstruction in the source plane in one row, which is output to .png.
    
    """
    slam.slam_util.output_model_to_fits(
        output_path=path.join(dataset_path, "model"),
        result=mass_result,
        model_lens_light=True,
        model_source_light=True,
        source_reconstruction=True,
    )

    slam.slam_util.output_model_results(
        output_path=path.join(dataset_path, "model"),
        result=mass_result,
        filename="model.results",
    )

    slam.slam_util.output_fit_multi_png(
        output_path=dataset_path,
        result_list=[mass_result],
        filename="sie_fit",
    )

    slam.slam_util.output_source_multi_png(
        output_path=dataset_path,
        result_list=[mass_result],
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
    )

    parser.add_argument(
        "--number_of_cores",
        metavar="int",
        required=False,
        help="The number of cores to parallelize the fit",
    )

    args = parser.parse_args()

    fit(
        dataset_name=args.dataset,
        mask_radius=float(args.mask_radius),
        number_of_cores=int(args.number_of_cores),
    )

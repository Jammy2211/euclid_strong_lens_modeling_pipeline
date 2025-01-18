"""
Euclid Pipeline: Point Source
=============================

This scripts allows you to run the Euclid point source lens modeling pipeline locally on your computer. It comes
with an example simulated Euclid strong lens dataset, which is fitted using the pipeline, and over 4 CPus should
take < 1 hour to run.

This should enable simple lens modeling of lensed quasars, supernovae or other point sources.

__Point Dataset__

Point source modeling uses a `PointDataset` object, which is a collection of positions and fluxes of the point
source (although the fluxes are by default not used in the modeling).

Here is the example the `point_dataset.json` file passed to this script on the GitHub page:

https://github.com/Jammy2211/euclid_strong_lens_modeling_pipeline/blob/main/dataset/point_example/point_dataset.json

If you want to perform point source modeling on your own dataset, you need to format it as a `.json` file using
the same format as the example above.

__Model__

This script fits a `PointDataset` data of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's total mass distribution is an `Isothermal`.
 - The source `Galaxy` is a point source `Point`.

The `ExternalShear` is also not included in the mass model, where it is for the `imaging` examples. For a quadruply
imaged point source (8 data points) there is insufficient information to fully constain a model with
an `Isothermal` and `ExternalShear` (9 parameters).
"""


def fit(
    dataset_name: str,
    use_fluxes: bool = False,
    number_of_cores: int = 1,
    iterations_per_update: int = 5000,
):
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
    __Dataset__
    
    Load the strong lens point-source dataset `simple`, which is the dataset we will use to perform point source 
    lens modeling.
    """
    dataset_path = path.join("dataset", dataset_name)

    """
    We now load the point source dataset we will fit using point source modeling. 
    
    We load this data as a `PointDataset`, which contains the positions and fluxes of every point source. 
    """
    dataset = al.from_json(
        file_path=path.join(dataset_path, "point_dataset.json"),
    )

    """
    We next load an image of the dataset. 
    
    Although we are performing point-source modeling and do not use this data in the actual modeling, it is useful to 
    load it for visualization, for example to see where the multiple images of the point source are located relative to the 
    lens galaxy.
    
    The image will also be passed to the analysis further down, meaning that visualization of the point-source model
    overlaid over the image will be output making interpretation of the results straight forward.
    
    Loading and inputting the image of the dataset in this way is entirely optional, and if you are only interested in
    performing point-source modeling you do not need to do this.
    """
    data = al.Array2D.from_fits(
        file_path=path.join(dataset_path, "data.fits"), pixel_scales=0.05
    )

    """
    __Point Solver__
    
    For point-source modeling we require a `PointSolver`, which determines the multiple-images of the mass model for a 
    point source at location (y,x) in the source plane. 
    
    It does this by ray tracing triangles from the image-plane to the source-plane and calculating if the 
    source-plane (y,x) centre is inside the triangle. The method gradually ray-traces smaller and smaller triangles so 
    that the multiple images can be determine with sub-pixel precision.
    
    The `PointSolver` requires a starting grid of (y,x) coordinates in the image-plane which defines the first set
    of triangles that are ray-traced to the source-plane. It also requires that a `pixel_scale_precision` is input, 
    which is the resolution up to which the multiple images are computed. The lower the `pixel_scale_precision`, the
    longer the calculation, with the value of 0.001 below balancing efficiency with precision.
    
    Strong lens mass models have a multiple image called the "central image". However, the image is nearly always 
    significantly demagnified, meaning that it is not observed and cannot constrain the lens model. As this image is a
    valid multiple image, the `PointSolver` will locate it irrespective of whether its so demagnified it is not observed.
    To ensure this does not occur, we set a `magnification_threshold=0.1`, which discards this image because its
    magnification will be well below this threshold.
    
    If your dataset contains a central image that is observed you should reduce to include it in
    the analysis.
    """
    grid = al.Grid2D.uniform(
        shape_native=(100, 100),
        pixel_scales=0.1,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
    )

    solver = al.PointSolver.for_grid(
        grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
    )

    """
    __Model__
    
    We compose a lens model where:
    
     - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters].
     - The source galaxy's light is a point `Point` [2 parameters].
    
    The number of free parameters and therefore the dimensionality of non-linear parameter space is N=7.
    
    __Name Pairing__
    
    Every point-source dataset in the `PointDataset` has a name, which in this example was `point_0`. This `name` pairs 
    the dataset to the `Point` in the model below. Because the name of the dataset is `point_0`, the 
    only `Point` object that is used to fit it must have the name `point_0`.
    
    If there is no point-source in the model that has the same name as a `PointDataset`, that data is not used in
    the model-fit. If a point-source is included in the model whose name has no corresponding entry in 
    the `PointDataset` it will raise an error.
    
    In this example, where there is just one source, name pairing appears unnecessary. However, point-source datasets may
    have many source galaxies in them, and name pairing is necessary to ensure every point source in the lens model is 
    fitted to its particular lensed images in the `PointDataset`.
    
    __Coordinates__
    
    The model fitting default settings assume that the lens galaxy centre is near the coordinates (0.0", 0.0"). 
    
    If for your dataset the  lens is not centred at (0.0", 0.0"), we recommend that you either: 
    
     - Reduce your data so that the centre is (`autolens_workspace/*/data_preparation`). 
     - Manually override the lens model priors (`autolens_workspace/*/modeling/imaging/customize/priors.py`).
    """
    # Lens:

    mass = af.Model(al.mp.Isothermal)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    # Source:

    if not use_fluxes:
        point_0 = af.Model(al.ps.Point)
    else:
        point_0 = af.Model(al.ps.PointFlux)

    source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

    # Overall Lens Model:

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    """
    __Search__
    
    The lens model is fitted to the data using a non-linear search. 
    
    All examples in the autolens workspace use the nested sampling algorithm 
    Nautilus (https://nautilus-sampler.readthedocs.io/en/latest/), which extensive testing has revealed gives the most 
    accurate and efficient modeling results.
    
    We make the following changes to the Nautilus settings:
    
     - Increase the number of live points, `n_live`, from the default value of 50 to 100. 
    
    These are the two main Nautilus parameters that trade-off slower run time for a more reliable and accurate fit.
    Increasing both of these parameter produces a more reliable fit at the expense of longer run-times.
    
    __Customization__
    
    The folders `autolens_workspace/*/point_source/modeling/searches` gives an overview of alternative non-linear searches,
    other than Nautilus, that can be used to fit lens models. They also provide details on how to customize the
    model-fit, for example the priors.
    
    The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  
    
     `/autolens_workspace/output/point_source/modeling/simple/mass[sie]_source[point]/unique_identifier`.
    
    __Unique Identifier__
    
    In the path above, the `unique_identifier` appears as a collection of characters, where this identifier is generated 
    based on the model, search and dataset that are used in the fit.
     
    An identical combination of model and search generates the same identifier, meaning that rerunning the script will use 
    the existing results to resume the model-fit. In contrast, if you change the model or search, a new unique identifier 
    will be generated, ensuring that the model-fit results are output into a separate folder.
    
    We additionally want the unique identifier to be specific to the dataset fitted, so that if we fit different datasets
    with the same model and search results are output to a different folder. We achieve this below by passing 
    the `dataset_name` to the search's `unique_tag`.
    
    __Number Of Cores__
    
    We include an input `number_of_cores`, which when above 1 means that Nautilus uses parallel processing to sample multiple 
    lens models at once on your CPU. When `number_of_cores=2` the search will run roughly two times as
    fast, for `number_of_cores=3` three times as fast, and so on. The downside is more cores on your CPU will be in-use
    which may hurt the general performance of your computer.
    
    You should experiment to figure out the highest value which does not give a noticeable loss in performance of your 
    computer. If you know that your processor is a quad-core processor you should be able to use `number_of_cores=4`. 
    
    Above `number_of_cores=4` the speed-up from parallelization diminishes greatly. We therefore recommend you do not
    use a value above this.
    
    For users on a Windows Operating system, using `number_of_cores>1` may lead to an error, in which case it should be 
    reduced back to 1 to fix it.
    """
    search = af.Nautilus(
        path_prefix=path.join("euclid_point_source_pipeline"),
        unique_tag=dataset_name,
        n_live=100,
        number_of_cores=number_of_cores,
    )

    """
    __Chi Squared__
    
    For point-source modeling, there are many different ways to define the likelihood function, broadly referred to a
    an `image-plane chi-squared` or `source-plane chi-squared`. This determines whether the multiple images of the point
    source are used to compute the likelihood in the source-plane or image-plane.
    
    We will use an "image-plane chi-squared", which uses the `PointSolver` to determine the multiple images of the point
    source in the image-plane for the given mass model and compares the positions of these model images to the observed
    images to compute the chi-squared and likelihood.
    
    There are still many different ways the image-plane chi-squared can be computed, for example do we allow for 
    repeat image-pairs (i.e. the same multiple image being observed multiple times)? Do we pair all possible combinations
    of multiple images to observed images? This example uses the simplest approach, which is to pair each multiple image
    with the observed image that is closest to it, allowing for repeat image pairs. 
    
    For a "source-plane chi-squared", the likelihood is computed in the source-plane. The analysis basically just ray-traces
    the multiple images back to the source-plane and defines a chi-squared metric. For example, the default implementation 
    sums the Euclidean distance between the image positions and the point source centre in the source-plane.
    
    The source-plane chi-squared is significantly faster to compute than the image-plane chi-squared, as it requires 
    only ray-tracing the ~4 observed image positions and does not require the iterative triangle ray-tracing approach
    of the image-plane chi-squared. However, the source-plane chi-squared is less robust than the image-plane chi-squared,
    and can lead to biased lens model results. If you are using the source-plane chi-squared, you should be aware of this
    and interpret the results with caution.
    
    Checkout the guide `autolens_workspace/*/guides/point_source.py` for more details and a full illustration of the
    different ways the chi-squared can be computed.
    
    __Analysis__
    
    The `AnalysisPoint` object defines the `log_likelihood_function` used by the non-linear search to fit the model 
    to the `PointDataset`.
    """
    analysis = al.AnalysisPoint(
        dataset=dataset,
        solver=solver,
        fit_positions_cls=al.FitPositionsImagePairRepeat,  # Image-plane chi-squared with repeat image pairs.
    )

    """
    __Model-Fit__
    
    We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
    for on-the-fly visualization and results).
    """
    result = search.fit(model=model, analysis=analysis)

    slam.slam_util.output_model_results(
        output_path=path.join(dataset_path, "result"),
        result=result,
        filename="model.results",
    )

    """
    Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.
    """


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lens Model Inputs")
    parser.add_argument(
        "--dataset", metavar="path", required=True, help="the path to the dataset"
    )
    parser.add_argument(
        "--use_fluxes", metavar="bool", required=False, help="Use fluxes in the fit"
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

    number_of_cores = int(args.number_of_cores) if args.number_of_cores is not None else 1
    iterations_per_update = int(args.iterations_per_update) if args.iterations_per_update is not None else 5000

    """
    __Convert__

    Convert from command line inputs of strings to correct types depending on if command line inputs are given.
    """
    fit(
        dataset_name=args.dataset,
        use_fluxes=args.use_fluxes,
        number_of_cores=args.number_of_cores,
        iterations_per_update=args.iterations_per_update,
    )

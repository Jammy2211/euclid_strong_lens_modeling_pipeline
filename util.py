"""
__Custom Visualizer & Analysis__
"""

from astropy.io import fits
import jax.numpy as jnp
import numpy as np
from PIL import Image

import autofit as af
import autolens as al
import autolens.plot as aplt


class VisualizerImaging(al.VisualizerImaging):
    """
    __RGB Visualizer__

    In built into **PyAutoLens** are `Visualizer` objects that output images of the dataset, fit, tracer and other
    quantities to hard-disk.

    These images for in the `image` folder of th modeling results. They are used for quick inspection of the fit and
    by the workflow functionality to produce new images of the results quickly.

    However, the source code visualizers cannot access quantities that are outside the inputs of the source-code,
    such as the RGB images of the dataset.

    The API below shows how a custom visualizer can be created that can access these quantities, output them to
    hard-disk in the modeling folder results and in the workflow examples are used to produce new images of the results.
    """

    @staticmethod
    def visualize_before_fit(
        analysis,
        paths: af.AbstractPaths,
        model: af.AbstractPriorModel,
    ):
        """
        PyAutoFit calls this function immediately before the non-linear search begins.

        It visualizes objects which do not change throughout the model fit like the dataset.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """
        dataset = analysis.dataset
        dataset_main_path = analysis.kwargs["dataset_main_path"]

        visualizer = al.VisualizerImaging()

        visualizer.visualize_before_fit(
            analysis=analysis,
            paths=paths,
            model=model,
        )

        # Load the images
        try:
            img0 = np.array(Image.open(dataset_main_path / "rgb_0.png"))
            img1 = np.array(Image.open(dataset_main_path / "rgb_1.png"))
        except FileNotFoundError:
            return

        mask = al.Mask2D.all_false(
            shape_native=(img0.shape[0], img0.shape[1]),
            pixel_scales=dataset.pixel_scales,
            origin=dataset.mask.origin,
        )

        img0 = al.Array2DRGB(values=img0, mask=mask)
        img1 = al.Array2DRGB(values=img1, mask=mask)

        img0_masked = al.Array2DRGB(values=img0, mask=dataset.mask)
        img1_masked = al.Array2DRGB(values=img1, mask=dataset.mask)

        mat_plot_2d = aplt.MatPlot2D(
            output=aplt.Output(
                path=paths.image_path, filename="subplot_rgb", format="png"
            ),
        )

        plotter_0 = aplt.Array2DPlotter(array=img0, mat_plot_2d=mat_plot_2d)
        plotter_1 = aplt.Array2DPlotter(array=img1, mat_plot_2d=mat_plot_2d)
        plotter_0_masked = aplt.Array2DPlotter(
            array=img0_masked, mat_plot_2d=mat_plot_2d
        )
        plotter_1_masked = aplt.Array2DPlotter(
            array=img1_masked, mat_plot_2d=mat_plot_2d
        )

        plotter_0.mat_plot_2d = plotter_1.mat_plot_2d

        plotter_0.open_subplot_figure(
            number_subplots=4,
            subplot_shape=(2, 2),
        )

        plotter_0.figure_2d()
        plotter_1.figure_2d()
        plotter_0_masked.figure_2d()
        plotter_1_masked.figure_2d()

        plotter_0.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot_rgb")
        plotter_0.close_subplot_figure()


class AnalysisImaging(al.AnalysisImaging):
    """
    Sets the custom RGB visualizer above ensuring the RGB subplot is output.
    """

    Visualizer = VisualizerImaging

    LATENT_KEYS = [
        "total_lens_flux",
        "total_lensed_source_flux",
        "total_source_flux",
    ]

    def compute_latent_variables(self, parameters, model):
        """
        A latent variable is not a model parameter but can be derived from the model. Its value and errors may be
        of interest and aid in the interpretation of a model-fit.

        This code implements a simple example of a latent variable, the magn

        By overwriting this method we can manually specify latent variables that are calculated and output to
        a `latent.csv` file, which mirrors the `samples.csv` file.

        In the example below, the `latent.csv` file will contain at least two columns with the shear magnitude and
        angle sampled by the non-linear search.

        This function is called at the end of search, following one of two schemes depending on the settings in
        `output.yaml`:

        1) Call for every search sample, which produces a complete `latent/samples.csv` which mirrors the normal
        `samples.csv` file but takes a long time to compute.

        2) Call only for N random draws from the posterior inferred at the end of the search, which only produces a
        `latent/latent_summary.json` file with the median and 1 and 3 sigma errors of the latent variables but is
        fast to compute.

        You can add your own custom latent variables here, if you have particular quantities that you
        would like to output to the `latent.csv` file.

        Parameters
        ----------
        parameters : array-like
            The parameter vector of the model sample. This will typically come from the non-linear search.
            Inside this method it is mapped back to a model instance via `model.instance_from_vector`.
        model : Model
            The model object defining how the parameter vector is mapped to an instance. Passed explicitly
            so that this function can be used inside JAX transforms (`vmap`, `jit`) with `functools.partial`.

        Returns
        -------
        A dictionary mapping every latent variable name to its value.

        """

        instance = model.instance_from_vector(vector=parameters)

        fit = self.fit_from(instance=instance)
        tracer = fit.tracer_linear_light_profiles_to_light_profiles

        try:
            total_lens_flux = np.sum(tracer.galaxies[0].image_2d_from(grid=self.dataset.grids.lp))
        except AttributeError:
            total_lens_flux = jnp.nan

        try:
            source_plane_grid = tracer.traced_grid_2d_list_from(grid=self.dataset.grids.lp)[-1]
            total_lensed_source_flux = np.sum(tracer.galaxies[-1].image_2d_from(grid=source_plane_grid))
        except AttributeError:
            total_lensed_source_flux = jnp.nan

        try:
            total_source_flux = np.sum(tracer.galaxies[-1].image_2d_from(grid=self.dataset.grids.lp))
        except AttributeError:
            total_source_flux = jnp.nan

        return (total_lens_flux, total_lensed_source_flux, total_source_flux)


def dataset_instrument_hdu_dict_via_fits_from(
    dataset_path, dataset_fits_name, image_tag: str = "_FLUX"
):
    """
    Load a dictionary mapping dataset instruments (e.g. DES_g, NIR_Y) to their index in a multi-extension
    fits file.

    Parameters
    ----------
    dataset_path
        The path where the multi-extension fits file is stored.
    dataset_fits_name
        The name of the multi-extension fits file.
    image_tag
        The tag appended to the instrument name of the image HDU, e.g. _FLUX, _IMAGE, which is used to pick
        out the image HDUs from the fits file and ignore other HDUs like noise maps or PSFs.

    Returns
    -------
    A dictionary mapping dataset names to their index in the fits file.
    """
    hdu_list = fits.open(dataset_path / dataset_fits_name)

    # Build dictionary: {name: index}
    hdu_dict = {}
    for i, hdu in enumerate(hdu_list):
        name = hdu.name if hdu.name else ("PRIMARY" if i == 0 else f"UNNAMED_{i}")
        hdu_dict[name] = i

    instrument_dict = {}
    counter = 0

    for hdu in hdu_list:
        name = hdu.name
        if name.endswith(image_tag):
            band = name.replace(image_tag, "").lower()
            instrument_dict[band] = counter
            counter += 1

    return instrument_dict

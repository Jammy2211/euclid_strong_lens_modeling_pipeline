from astropy.io import fits
import numpy as np
from PIL import Image

import autofit as af
import autolens as al
import autolens.plot as aplt


def ab_mag_via_flux_from(flux, magzero, xp=np):
    """
    Convert image flux values (ADU) into calibrated astronomical AB magnitudes.

    This uses the standard relation:
        m_AB = -2.5 log10(flux) + magzero

    `flux` and `magzero` must be in consistent units. Euclid VIS, NISP and EXT data are
    typically in different units (E.g. VIS is ADU / second, NISP and EXT are
    electrons / second). However, because `magzero` is also in these units, this
    function does not need to know the specific units of `flux`.

    Parameters
    ----------
    flux : float or xp.ndarray
        Measured flux value(s) in image units (ADU). Must be strictly positive.
    magzero : float
        Photometric zero-point defining the AB magnitude system for the image.

    Returns
    -------
    ab_mag : float or xp.ndarray
        The corresponding AB magnitude(s).
    """
    ab_mag = -2.5 * xp.log10(flux) + magzero
    return ab_mag


def flux_mujy_via_ab_mag_from(ab_mag, xp=np):
    """
    Convert AB magnitudes into flux density expressed in microJansky (µJy).

    This uses the AB definition where a source with 0 mag has a flux of 3631 Jy.

    Parameters
    ----------
    ab_mag : float or array-like
        AB magnitude value(s).

    Returns
    -------
    flux_mujy : float or xp.ndarray
        Flux densities in microJansky (µJy).
    """
    flux_mujy = 3631e6 * 10 ** (-0.4 * ab_mag)
    return flux_mujy


def aperture_flux_from(image_2d, centre, radius_pixels, xp=np):
    """
    Measure enclosed flux inside a single circular aperture on an image.

    Given an image and a central coordinate (typically the lens centre),
    compute the total pixel flux within a circular aperture of specified radius.

    Parameters
    ----------
    image_2d : 2D xp.ndarray
        Image data (NumPy or JAX array).
    centre : (float, float)
        (y, x) coordinate defining the centre of the aperture in pixel units.
    radius_pixels : float
        Aperture radius in pixel units.
    xp : array module, optional
        Array namespace (default: numpy). Can also be jax.numpy.

    Returns
    -------
    float
        Total flux inside the circular aperture.
    """
    y0, x0 = centre
    yy, xx = xp.indices(image_2d.shape)

    rr = xp.sqrt((yy - y0) ** 2 + (xx - x0) ** 2)

    # mask = 1 inside aperture, 0 outside (ensures JAX-safety)
    mask = (rr <= radius_pixels).astype(image_2d.dtype)

    # Equivalent to summing only those inside aperture, but no boolean indexing
    return xp.sum(image_2d * mask)


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
        skip_rgb_plot = analysis.kwargs.get("skip_rgb_plot", False)
        if skip_rgb_plot:
            return

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
        "latent.total_lens_flux",
        "latent.total_lens_flux_13_pix",
        "latent.total_lens_flux_26_pix",
        "latent.total_lens_flux_39_pix",
        "latent.total_lens_flux_52_pix",
        "latent.total_lensed_source_flux",
        "latent.total_source_flux",
        "latent.magnification",
    #    "latent_effective_einstein_radius"
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

        xp = self._xp

        instance = model.instance_from_vector(vector=parameters)

        fit = self.fit_from(instance=instance)
        tracer = fit.tracer_linear_light_profiles_to_light_profiles

        magzero = self.kwargs.get("magzero", None)

        if magzero is None:
            raise ValueError(
                "MAGZERO must be provided in analysis kwargs to compute latent variables."
            )

        # LENS LIGHT FLUX IN MICROJANSKY, INCLUDING APERTURE FLUXES

        try:

            image = tracer.image_2d_from(grid=self.dataset.grids.lp, xp=xp)

            total_lens_flux = xp.sum(image.array)
            lens_ab_mag = ab_mag_via_flux_from(
                flux=total_lens_flux, magzero=magzero, xp=xp
            )
            total_lens_flux_muJy = flux_mujy_via_ab_mag_from(ab_mag=lens_ab_mag, xp=xp)

            image_native = xp.zeros(image.mask.shape, dtype=image.dtype)

            if xp is np:

                image_native[image.mask.slim_to_native_tuple] = image.array

            else:

                image_native = image_native.at[image.mask.slim_to_native_tuple].set(image.array)

            flat_index = xp.argmax(image_native)
            y, x = xp.unravel_index(flat_index, image_native.shape)

            aperture_radii = [13, 26, 39, 52]  # pixels

            total_lens_flux_aperture_list = [
                aperture_flux_from(
                    image_2d=image_native,
                    centre=(y, x),
                    radius_pixels=radius_pixels,
                    xp=xp,
                )
                for radius_pixels in aperture_radii
            ]

            # convert each aperture flux to magnitude and µJy
            total_lens_mag_aperture_list = [
                ab_mag_via_flux_from(
                    flux=total_lens_flux_aperture, magzero=magzero, xp=xp
                )
                for total_lens_flux_aperture in total_lens_flux_aperture_list
            ]
            total_lens_flux_muJy_aperture_list = [
                flux_mujy_via_ab_mag_from(ab_mag=m, xp=xp)
                for m in total_lens_mag_aperture_list
            ]

        except AttributeError:
            total_lens_flux_muJy = xp.nan
            total_lens_flux_muJy_aperture_list = [xp.nan, xp.nan, xp.nan, xp.nan]

        # LENSED SOURCE FLUX IN MICROJANSKY

        try:

            source_plane_grid = tracer.traced_grid_2d_list_from(
                grid=self.dataset.grids.lp, xp=xp
            )[-1]
            lensed_source_image = tracer.galaxies[-1].image_2d_from(
                grid=source_plane_grid, xp=xp
            )
            total_lensed_source_flux = xp.sum(lensed_source_image.array)
            lensed_source_ab_mag = ab_mag_via_flux_from(
                flux=total_lensed_source_flux, magzero=magzero, xp=xp
            )
            total_lensed_source_flux_muJy = flux_mujy_via_ab_mag_from(
                ab_mag=lensed_source_ab_mag, xp=xp
            )

        except AttributeError:
            total_lensed_source_flux_muJy = xp.nan

        # SOURCE FLUX IN MICROJANSKY

        try:
            source_image = tracer.galaxies[-1].image_2d_from(
                grid=self.dataset.grids.lp, xp=xp
            )
            total_source_flux = xp.sum(source_image.array)
            source_ab_mag = ab_mag_via_flux_from(
                flux=total_source_flux, magzero=magzero, xp=xp
            )
            total_source_flux_muJy = flux_mujy_via_ab_mag_from(
                ab_mag=source_ab_mag, xp=xp
            )

        except AttributeError:
            total_source_flux_muJy = xp.nan

        # MAGNIFICATION

        try:
            magnification = total_lensed_source_flux / total_source_flux
        except AttributeError:
            magnification = xp.nan

        # EFFECTIVE EINSTEIN RADIUS

        # try:
        #     effective_einstein_radius = tracer.einstein_radius_from(
        #         grid=self.dataset.grids.lp
        #     )
        # except Exception:
        #     effective_einstein_radius = xp.nan

        return (
            total_lens_flux_muJy,
            total_lens_flux_muJy_aperture_list[0],
            total_lens_flux_muJy_aperture_list[1],
            total_lens_flux_muJy_aperture_list[2],
            total_lens_flux_muJy_aperture_list[3],
            total_lensed_source_flux_muJy,
            total_source_flux_muJy,
            magnification,
        #    effective_einstein_radius
        )

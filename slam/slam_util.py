import copy
import csv
import json
import numpy as np
from os import path
from typing import Dict, List, Optional, Union

import os
from astropy.io import fits
from os import path

from autoconf import conf

import autofit as af
import autolens as al
import autolens.plot as aplt


def update_result_json_file(
        file_path: str,
        result,
        waveband,
        einstein_radius: bool = False,
        fluxes: bool = False,
        fluxes_with_errors: bool = False,
        magnitude: bool = False,
        zero_point: float = Optional[None],
):
    """
    The `result.json` file is output to the `dataset/lens_name` folder and it contains all keys lens modeling
    results (e.g. Einstein Radii, magnitudes, magnifications) in a format that is easily readable by the user.

    This function updates the `result.json` file (or creates it if it does not exist) with the results of the lens
    modeling.

    Parameters
    ----------
    file_path
        The path to the lens dataset folder where the `result.json` file is stored.
    result
        The result object from the lens modeling which is used to update the `result.json` file.
    einstein_radius
        Whether to update the Einstein Radius in the `result.json` file.
    """
    try:
        with open(file_path, "r") as f:
            result_dict = json.load(f)
    except FileNotFoundError:
        result_dict = {}

    samples = result.samples

    if einstein_radius:
        tracer = result.max_log_likelihood_tracer
        einstein_radius = tracer.einstein_radius_from(grid=result.grids.lp)
        result_dict["einstein_radius_max_lh"] = einstein_radius

        einstein_radius_list = []

        for i in range(50):

            instance = samples.draw_randomly_via_pdf()

            tracer = result.analysis.tracer_via_instance_from(instance=instance)
            einstein_radius = tracer.einstein_radius_from(grid=result.grids.lp)

            einstein_radius_list.append(einstein_radius)

        (
            median_einstein_radius,
            lower_einstein_radius,
            upper_einstein_radius,
        ) = af.marginalize(parameter_list=einstein_radius_list, sigma=3.0)

        result_dict["einstein_radius_median_pdf"] = median_einstein_radius
        result_dict["einstein_radius_lower_3_sigma"] = lower_einstein_radius
        result_dict["einstein_radius_upper_3_sigma"] = upper_einstein_radius

    if fluxes:

        tracer = result.max_log_likelihood_tracer

        image = tracer.galaxies[0].image_2d_from(grid=result.grids.lp)
        total_lens_flux = np.sum(image)
        result_dict[f"{waveband}_total_lens_flux_max_lh"] = total_lens_flux

        inversion = result.max_log_likelihood_fit.inversion
        mapper = inversion.cls_list_from(cls=al.AbstractMapper)[0]

        mapper_valued = al.MapperValued(
            mapper=mapper,
            values=inversion.reconstruction_dict[mapper],
        )

        mapped_reconstructed_image = mapper_valued.mapped_reconstructed_image_from()
        total_lensed_source_flux = np.sum(mapped_reconstructed_image)
        result_dict[f"{waveband}_total_lensed_source_flux_max_lh"] = total_lensed_source_flux

        reconstruction = inversion.reconstruction_dict[mapper]

        total_source_flux = np.sum(reconstruction)
        result_dict[f"{waveband}_total_source_flux_max_lh"] = total_source_flux

        magnification = mapper_valued.magnification_via_interpolation_from()
        result_dict[f"{waveband}_magnification_max_lh"] = magnification

        lensed_source_image = result.max_log_likelihood_fit.model_images_of_planes_list[
            -1
        ]
        signal_to_noise_map = (
                lensed_source_image / result.max_log_likelihood_fit.dataset.noise_map
        )
        result_dict[f"{waveband}_max_lensed_source_signal_to_noise_ratio"] = np.max(
            signal_to_noise_map
        )

        if magnitude:
            lens_magnitude_ab = - 2.5 * np.log10(total_lens_flux) + zero_point
            lensed_source_magnitude_ab = - 2.5 * np.log10(total_lensed_source_flux) + zero_point
            source_magnitude_ab = - 2.5 * np.log10(total_source_flux) + zero_point

            result_dict[f"{waveband}_lens_magnitude_ab_max_lh"] = lens_magnitude_ab
            result_dict[f"{waveband}_lensed_source_magnitude_ab_max_lh"] = lensed_source_magnitude_ab
            result_dict[f"{waveband}_source_magnitude_ab_max_lh"] = source_magnitude_ab

    if fluxes_with_errors:

        total_lens_flux_list = []
        total_lensed_source_flux_list = []
        total_source_flux_list = []
        total_magnification_list = []
        lens_magnitude_ab_list = []
        lensed_source_magnitude_ab_list = []
        source_magnitude_ab_list = []

        for i in range(50):

            instance = samples.draw_randomly_via_pdf()

            fit = result.analysis.fit_from(instance=instance)
            tracer = fit.tracer_linear_light_profiles_to_light_profiles

            image = tracer.galaxies[0].image_2d_from(grid=result.grids.lp)

            total_lens_flux = np.sum(image)
            total_lens_flux_list.append(total_lens_flux)

            inversion = fit.inversion
            mapper = inversion.cls_list_from(cls=al.AbstractMapper)[0]

            mapper_valued = al.MapperValued(
                mapper=mapper,
                values=inversion.reconstruction_dict[mapper],
            )

            mapped_reconstructed_image = mapper_valued.mapped_reconstructed_image_from()
            total_lensed_source_flux = np.sum(mapped_reconstructed_image)

            total_lensed_source_flux_list.append(total_lensed_source_flux)

            reconstruction = inversion.reconstruction_dict[mapper]
            total_source_flux = np.sum(reconstruction)
            total_source_flux_list.append(total_source_flux)

            magnification = mapper_valued.magnification_via_interpolation_from(shape_native=(101, 101))
            total_magnification_list.append(magnification)

            lens_magnitude_ab = - 2.5 * np.log10(total_lens_flux) + zero_point
            lens_magnitude_ab_list.append(lens_magnitude_ab)

            lensed_source_magnitude_ab = - 2.5 * np.log10(total_lensed_source_flux) + zero_point
            lensed_source_magnitude_ab_list.append(lensed_source_magnitude_ab)

            source_magnitude_ab = - 2.5 * np.log10(total_source_flux) + zero_point
            source_magnitude_ab_list.append(source_magnitude_ab)

        (
            median_total_lens_flux,
            lower_total_lens_flux,
            upper_total_lens_flux,
        ) = af.marginalize(parameter_list=total_lens_flux_list, sigma=3.0)

        result_dict[f"{waveband}_total_lens_flux_median_pdf"] = median_total_lens_flux
        result_dict[f"{waveband}_total_lens_flux_lower_3_sigma"] = lower_total_lens_flux
        result_dict[f"{waveband}_total_lens_flux_upper_3_sigma"] = upper_total_lens_flux

        (
            median_total_lensed_source_flux,
            lower_total_lensed_source_flux,
            upper_total_lensed_source_flux,
        ) = af.marginalize(parameter_list=total_lensed_source_flux_list, sigma=3.0)

        result_dict[f"{waveband}_total_lensed_source_flux_median_pdf"] = median_total_lensed_source_flux
        result_dict[f"{waveband}_total_lensed_source_flux_lower_3_sigma"] = lower_total_lensed_source_flux
        result_dict[f"{waveband}_total_lensed_source_flux_upper_3_sigma"] = upper_total_lensed_source_flux

        (
            median_total_source_flux,
            lower_total_source_flux,
            upper_total_source_flux,
        ) = af.marginalize(parameter_list=total_source_flux_list, sigma=3.0)

        result_dict[f"{waveband}_total_source_flux_median_pdf"] = median_total_source_flux
        result_dict[f"{waveband}_total_source_flux_lower_3_sigma"] = lower_total_source_flux
        result_dict[f"{waveband}_total_source_flux_upper_3_sigma"] = upper_total_source_flux

        (
            median_total_magnification,
            lower_total_magnification,
            upper_total_magnification,
        ) = af.marginalize(parameter_list=total_magnification_list, sigma=3.0)

        result_dict[f"{waveband}_total_magnification_median_pdf"] = median_total_magnification
        result_dict[f"{waveband}_total_magnification_lower_3_sigma"] = lower_total_magnification
        result_dict[f"{waveband}_total_magnification_upper_3_sigma"] = upper_total_magnification

        if magnitude:
            (
                median_lens_magnitude_ab,
                lower_lens_magnitude_ab,
                upper_lens_magnitude_ab,
            ) = af.marginalize(parameter_list=lens_magnitude_ab_list, sigma=3.0)

            result_dict[f"{waveband}_lens_magnitude_ab_median_pdf"] = median_lens_magnitude_ab
            result_dict[f"{waveband}_lens_magnitude_ab_lower_3_sigma"] = lower_lens_magnitude_ab
            result_dict[f"{waveband}_lens_magnitude_ab_upper_3_sigma"] = upper_lens_magnitude_ab

            (
                median_lensed_source_magnitude_ab,
                lower_lensed_source_magnitude_ab,
                upper_lensed_source_magnitude_ab,
            ) = af.marginalize(parameter_list=lensed_source_magnitude_ab_list, sigma=3.0)

            result_dict[f"{waveband}_lensed_source_magnitude_ab_median_pdf"] = median_lensed_source_magnitude_ab
            result_dict[f"{waveband}_lensed_source_magnitude_ab_lower_3_sigma"] = lower_lensed_source_magnitude_ab
            result_dict[f"{waveband}_lensed_source_magnitude_ab_upper_3_sigma"] = upper_lensed_source_magnitude_ab

            (
                median_source_magnitude_ab,
                lower_source_magnitude_ab,
                upper_source_magnitude_ab,
            ) = af.marginalize(parameter_list=source_magnitude_ab_list, sigma=3.0)

            result_dict[f"{waveband}_source_magnitude_ab_median_pdf"] = median_source_magnitude_ab
            result_dict[f"{waveband}_source_magnitude_ab_lower_3_sigma"] = lower_source_magnitude_ab
            result_dict[f"{waveband}_source_magnitude_ab_upper_3_sigma"] = upper_source_magnitude_ab

    with open(file_path, "w") as f:
        json.dump(result_dict, f, indent=4)


def update_progress_csv_file(
        file_path: str,
        info : Dict,
        phase : str
):
    """
    Updates the `progress.csv` file, which lists the progress of the model fits to all lenses in a sample.

    This code operates as follows:

    1) Determine if a `progress.csv` file exists, if so load it, else create one.
    2) Append a new entry containing the lens index, name and phase of the pipeline it has just completed. If the index
    of that lens is already in the .csv file, it overwritres this, so that a lens does not get written multiple times.
    3) Order the entries by index so they are in ascending order of lens index.
    4) Write to a new `progress.csv` file.

    Parameters
    ----------
    file_path
        The path to where the `progress.csv` file is stored.
    info
        A dictionary containing information about the lens, in particular its index number used to order the ,csv
        field and its name or id string.
    phase
        The name of the phase (e.g. `source_lp[1]` in the SLaM pipeline that just completed and thus is where
        the model-fit is.
    """

    data = []

    try:
        with open(file_path, mode="r") as file:
            reader = csv.DictReader(file)
            # Ensure all rows are treated as dictionaries for consistency
            data = [row for row in reader]
    except FileNotFoundError:
        pass

    # Append new entry
    new_entry = {"": info["index"], "id_str": info["id_str"], "phase": phase}
    data = [row for row in data if row[""] != new_entry[""]]  # Remove any existing entry with the same id
    data.append(new_entry)

    # Sort data by ascending index
    sorted_data = sorted(data, key=lambda x: int(x[""]))

    # Write back to the CSV file
    fieldnames = new_entry.keys()
    with open(file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_data)


def output_result_to_fits(
    output_path: str,
    result,
    model_lens_light: bool = False,
    model_source_light: bool = False,
    mge_source_reconstruction: bool = False,
    source_reconstruction: bool = False,
    source_reconstruction_noise_map: bool = False,
    tag = None,
    remove_fits_first : bool = False,
    prefix: str = "",
):
    """
    Output modeling results from the SLAM pipeline to .fits files.

    These typically go to the path the dataset is stored in, so that a dataset can be extended with the modeling results
    easily.

    Parameters
    ----------
    output_path
        The path to the output directory where the modeling results are stored.
    result
        The result object from the SLaM pipeline used to make the modeling results, typically the MASS PIPELINE.
    model_lens_light
        When to output a 2D image of the lens light model to a .fits file.
    model_source_light
        When to output a 2D image of the source light model to a .fits file.
    source_reconstruction
        When to output a 2D image of the source reconstruction to a .fits file, where this may be interpolated from
        an irregular pixelization like a Delaunay mesh or Voronoi mesh.
    source_reconstruction_noise_map
        When to output a 2D image of the source reconstruction noise-map to a .fits file, where this may be
        interpolated from an irregular pixelization like a Delaunay mesh or Voronoi mesh.
    """
    fit = result.max_log_likelihood_fit

    def update_fits_file(arr, output_fits_file, remove_fits_first = False, tag = None):

        if remove_fits_first and os.path.exists(output_fits_file):
            os.remove(output_fits_file)

        if conf.instance["general"]["fits"]["flip_for_ds9"]:
            arr = np.flipud(arr)

        if os.path.exists(output_fits_file):

            with fits.open(output_fits_file, mode='update') as hdul:
                hdul.append(fits.ImageHDU(arr))
                if tag is not None:
                    hdul[-1].header['EXTNAME'] = tag.upper()
                hdul.flush()

        else:

            hdu = fits.PrimaryHDU(arr)
            if tag is not None:
                hdu.header['EXTNAME'] = tag.upper()
            hdul = fits.HDUList([hdu])
            hdul.writeto(output_fits_file, overwrite=True)

    if model_lens_light:

        update_fits_file(
            arr=fit.model_images_of_planes_list[0].native,
            output_fits_file=path.join(output_path, f"{prefix}lens_light.fits"),
            remove_fits_first=remove_fits_first,
            tag=tag,
        )

    if model_source_light:

        update_fits_file(
            arr=fit.model_images_of_planes_list[-1].native,
            output_fits_file=path.join(output_path, f"{prefix}source_light.fits"),
            remove_fits_first=remove_fits_first,
            tag=tag,
        )

    if mge_source_reconstruction:
        grid = al.Grid2D.from_extent(shape_native=(201, 201), extent=(-1.0, 1.0, -1.0, 1.0))

        plane_image = fit.tracer.galaxies[-1].image_2d_from(grid=grid)

        update_fits_file(
            arr=plane_image.native,
            output_fits_file=path.join(output_path, f"mge_source_reconstruction.fits"),
            remove_fits_first=remove_fits_first,
            tag=tag,
        )

    if source_reconstruction:
        inversion = fit.inversion
        mapper = inversion.cls_list_from(cls=al.AbstractMapper)[0]
        mapper_valued = al.MapperValued(
            mapper=mapper, values=inversion.reconstruction_dict[mapper]
        )

        interpolated_reconstruction = mapper_valued.interpolated_array_from(
            shape_native=(201, 201),
            extent=(-1.0, 1.0, -1.0, 1.0)
        )

        update_fits_file(
            arr=interpolated_reconstruction.native,
            output_fits_file=path.join(output_path, f"source_reconstruction.fits"),
            remove_fits_first=remove_fits_first,
            tag=tag,
        )

    if source_reconstruction_noise_map:
        inversion = fit.inversion
        mapper = inversion.cls_list_from(cls=al.AbstractMapper)[0]
        mapper_valued = al.MapperValued(
            mapper=mapper, values=inversion.reconstruction_noise_map
        )

        interpolated_reconstruction_noise_map = mapper_valued.interpolated_array_from(
            shape_native=(201, 201),
            extent=(-1.0, 1.0, -1.0, 1.0)
        )

        update_fits_file(
            arr=interpolated_reconstruction_noise_map.native,
            output_fits_file=path.join(output_path, f"source_reconstruction_noise_map.fits"),
            remove_fits_first=remove_fits_first,
            tag=tag,
        )



def output_model_results(
    output_path: str,
    result,
    filename: str = "model.results",
):
    """
    Outputs the results of a model-fit to an easily readable `model.results` file containing the model parameters and
    log likelihood of the fit.

    Parameters
    ----------
    output_path
        The path to the output directory where the modeling results are stored.
    result
        The result object from the SLaM pipeline used to make the modeling results, typically the MASS PIPELINE.
    filename
        The name of the file that the results are written to.
    """

    from autofit.text import text_util
    from autofit.tools.util import open_

    result_info = text_util.result_info_from(
        samples=result.samples,
    )

    with open_(path.join(output_path, filename), "w") as f:
        f.write(result_info)
        f.close()


def plot_fit_png_row(
    plotter_main,
    fit,
    tag,
    vmax,
    vmax_lens_light,
    vmax_convergence,
    image_plane_extent,
    visuals_2d,
    title_fontweight="normal",
):
    """
    Plots a row of a subplot which shows a fit to a list of datasets (e.g. varying across wavelengths) where each row
    corresponds to a different dataset.

    Parameters
    ----------
    plotter_main
        The main plotter object that is used to create the subplot.
    fit
        The fit to the dataset that is plotted, which corresponds to a row in the subplot.
    tag
        The tag that labels the row of the subplot.
    vmax
        The maximum pixel value of the subplot, which is chosen based on all fits in the list in order to make visual
        comparison easier.
    vmax_lens_light
        The maximum pixel value of the lens light subplot, again chosen based on all fits in the list.
    vmax_convergence
        The maximum pixel value of the convergence subplot, again chosen based on all fits in the list.
    image_plane_extent
        The extent of the image-plane grid that is plotted, chosen to be the same for all fits in the list.
    source_plane_extent
        The extent of the source-plane grid that is plotted, chosen to be the same for all fits in the list.
    visuals_2d
        The 2D visuals that are plotted on the subplot, which are chosen to be the same for all fits in the list.
    """

    plotter = aplt.FitImagingPlotter(
        fit=fit,
        include_2d=aplt.Include2D(
            light_profile_centres=False, mass_profile_centres=False
        ),
    )

    plotter_main.mat_plot_2d.axis = aplt.Axis(extent=image_plane_extent)
    plotter_main.mat_plot_2d.cmap = aplt.Cmap()
    plotter_main.mat_plot_2d.title = aplt.Title(
        fontweight=title_fontweight,
        fontsize=16
    )

    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax)

    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.mat_plot_2d.title = aplt.Title(
            label=tag,
            fontsize=11,
        )
    plotter.mat_plot_2d.use_log10 = True
    plotter.figures_2d(data=True)
    plotter.mat_plot_2d.use_log10 = False

    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.mat_plot_2d.title = aplt.Title(
            label=tag,
            fontsize=16,
        )
    plotter.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax)
    plotter.set_title(label=f"Lens Subtracted Image")
    plotter.figures_2d_of_planes(
        plane_index=1, subtracted_image=True, use_source_vmax=True
    )

    visuals_2d_original = copy.copy(plotter_main.visuals_2d)

    subplot_index = copy.copy(plotter_main.mat_plot_2d.subplot_index)

    plotter_main.visuals_2d = visuals_2d
    plotter.visuals_2d = plotter_main.visuals_2d
    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax)
    plotter.set_title(label=f"Lensed Source Model")
    plotter.figures_2d_of_planes(plane_index=1, model_image=True, use_source_vmax=True)
    plotter.visuals_2d = visuals_2d_original
    plotter_main.visuals_2d = visuals_2d_original

    plotter_main.mat_plot_2d.subplot_index = subplot_index + 1

    subplot_index = copy.copy(plotter_main.mat_plot_2d.subplot_index)
    plotter_main.mat_plot_2d.axis = aplt.Axis()
    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.set_title(label=f"Source Plane")
    plotter.figures_2d_of_planes(plane_index=1, plane_image=True, use_source_vmax=True, zoom_to_brightest=False)
    plotter_main.mat_plot_2d.subplot_index = subplot_index + 1


    subplot_index = copy.copy(plotter_main.mat_plot_2d.subplot_index)
    plotter_main.mat_plot_2d.axis = aplt.Axis(extent=image_plane_extent)
    plotter.set_title(label=f"Lens Light")
    plotter.mat_plot_2d.use_log10 = True
    tracer_plotter = plotter.tracer_plotter
    tracer_plotter.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax_lens_light)
    tracer_plotter.include_2d._light_profile_centres = False
    tracer_plotter.include_2d._mass_profile_centres = False
    tracer_plotter.include_2d._tangential_critical_curves = False
    tracer_plotter.include_2d._radial_critical_curves = False

    try:
        tracer_plotter.figures_2d_of_planes(
            plane_image=True, plane_index=0, zoom_to_brightest=False
        )
    except ValueError:
        pass
    plotter_main.mat_plot_2d.subplot_index = subplot_index + 1

    tracer_plotter = plotter.tracer_plotter
    tracer_plotter.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax_convergence)

    tracer_plotter.set_title(label=f"Convergence")
    tracer_plotter.figures_2d(convergence=True)

    tracer_plotter.include_2d._light_profile_centres = True
    tracer_plotter.include_2d._mass_profile_centres = True
    tracer_plotter.include_2d._tangential_critical_curves = True
    tracer_plotter.include_2d._radial_critical_curves = True

    plotter.mat_plot_2d.use_log10 = False


def output_fit_multi_png(
    output_path: str,
    result_list,
    tag_list=None,
    filename="fit",
    main_dataset_index: int = 1e99,
    tag_prefix: str = ""
):
    """
    Outputs a .png subplot of a fit to multiple datasets (e.g. varying across wavelengths) where each row
    corresponds to a different dataset.

    Many aspects of the plot are homogenized so that the fits can be compared easily.

    Parameters
    ----------
    output_path
        The path to the output directory where the modeling results are stored.
    result_list
        A list of results from the SLaM pipeline used to make the modeling results, typically the MASS PIPELINE.
    tag_list
        A list of tags to label each row of the subplot.
    filename
        The name of the file that the results are written to.
    """
    fit_list = [result.max_log_likelihood_fit for result in result_list]

    vmax = (
        np.max([np.max(fit.model_images_of_planes_list[1]) for fit in fit_list]) / 2.0
    )

    image_plane_extent = fit_list[0].data.extent_of_zoomed_array()

    vmax_lens_light = np.min(
        [np.max(fit.model_images_of_planes_list[0]) for fit in fit_list]
    )

    vmax_convergence = np.min(
        [
            np.max(fit.tracer.convergence_2d_from(grid=fit.dataset.grid))
            for fit in fit_list
        ]
    )

    plotter = aplt.FitImagingPlotter(
        fit=fit_list[0],
        mat_plot_2d=aplt.MatPlot2D(
            output=aplt.Output(path=output_path, filename=filename, format="png"),
        ),
    )

    plotter.open_subplot_figure(
        number_subplots=len(fit_list) * 6,
        subplot_shape=(len(fit_list), 6),
    )

    for i, fit in enumerate(fit_list):
        title_fontweight = "bold" if i == main_dataset_index else "normal"

        visuals_2d = aplt.Visuals2D(
            light_profile_centres=al.Grid2DIrregular(
                values=[fit.tracer.galaxies[0].bulge.profile_list[0].centre]
            ),
            mass_profile_centres=al.Grid2DIrregular(
                values=[fit.tracer.galaxies[0].mass.centre]
            ),
        )

        tag = tag_list[i] if tag_list is not None else ""

        plot_fit_png_row(
            plotter_main=plotter,
            fit=fit,
            tag=f"{tag_prefix}{tag}",
            vmax=vmax,
            vmax_lens_light=vmax_lens_light,
            vmax_convergence=vmax_convergence,
            image_plane_extent=image_plane_extent,
            visuals_2d=visuals_2d,
            title_fontweight=title_fontweight,
        )

    plotter.mat_plot_2d.output.subplot_to_figure(auto_filename=filename)
    plotter.close_subplot_figure()


def plot_source_png_row(
    plotter_main, fit, tag, vmax, source_plane_extent, title_fontweight="normal"
):
    """
    Plots a row of a subplot which shows a source reconstruction to a list of datasets (e.g. varying across wavelengths)
    where each row corresponds to a different dataset.

    Parameters
    ----------
    plotter_main
        The main plotter object that is used to create the subplot.
    fit
        The fit to the dataset that is plotted, which corresponds to a row in the subplot.
    tag
        The tag that labels the row of the subplot.
    vmax
        The maximum pixel value of the subplot, which is chosen based on all fits in the list in order to make visual
        comparison
    source_plane_extent
        The extent of the source-plane grid that is plotted, chosen to be the same for all fits in the list.
    """
    plotter = aplt.FitImagingPlotter(
        fit=fit,
        include_2d=aplt.Include2D(
            light_profile_centres=False, mass_profile_centres=False
        ),
    )

    plotter.mat_plot_2d = plotter_main.mat_plot_2d

    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax)
    plotter.mat_plot_2d.title = aplt.Title(
            label=f"{tag} Image",
            fontsize=11,
        )
    plotter.figures_2d_of_planes(
        plane_index=1, subtracted_image=True, use_source_vmax=True
    )

    visuals_2d_original = copy.copy(plotter_main.visuals_2d)

    subplot_index = copy.copy(plotter_main.mat_plot_2d.subplot_index)

    plotter.visuals_2d = plotter_main.visuals_2d
    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax)
    plotter.mat_plot_2d.title = aplt.Title(
            label=f"Lensed Source Model",
            fontsize=16,
        )
    plotter.figures_2d_of_planes(plane_index=1, model_image=True, use_source_vmax=True)
    plotter.visuals_2d = visuals_2d_original
    plotter_main.visuals_2d = visuals_2d_original

    plotter_main.mat_plot_2d.subplot_index = subplot_index + 1

    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax)
    plotter_main.mat_plot_2d.title = aplt.Title(
        fontweight=title_fontweight, fontsize=16
    )
    plotter.set_title(label=f"Source")
    plotter.figures_2d_of_planes(
        plane_index=1,
        plane_image=True,
        use_source_vmax=True,
        zoom_to_brightest=False,
    )

    plotter_main.mat_plot_2d.axis = aplt.Axis(extent=source_plane_extent)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.set_title(label=f"Source (Zoomed)")
    plotter.figures_2d_of_planes(plane_index=1, plane_image=True, use_source_vmax=True)

    plotter.set_title(label=f"Source S/N (Zoomed)")
    plotter.mat_plot_2d.use_log10 = False
    plotter.figures_2d_of_planes(
        plane_index=1,
        plane_signal_to_noise_map=True,
    )

    plotter_main.mat_plot_2d.axis = aplt.Axis()
    plotter.set_title(label=f"Source Interpolation")
    plotter.mat_plot_2d.use_log10 = False
    plotter.figures_2d_of_planes(
        plane_index=1,
        plane_image=True,
        use_source_vmax=True,
        interpolate_to_uniform=True,
        zoom_to_brightest=False,
    )


def output_source_multi_png(
    output_path: str,
    result_list,
    tag_list=None,
    filename="source_reconstruction",
    main_dataset_index: int = 1e99,
    tag_prefix : str = ""
):
    """
    Outputs a .png subplot of the source-plane source reconstructions to multiple datasets (e.g. varying across
    wavelengths) where each row corresponds to a different dataset.

    Many aspects of the plot are homogenized so that the fits can be compared easily.

    Parameters
    ----------
    output_path
        The path to the output directory where the modeling results are stored.
    result_list
        A list of results from the SLaM pipeline used to make the modeling results, typically the MASS PIPELINE.
    tag_list
        A list of tags to label each row of the subplot.
    filename
        The name of the file that the results are written to.
    """

    fit_list = [result.max_log_likelihood_fit for result in result_list]

    mapper_list = [
        fit.inversion.cls_list_from(cls=al.AbstractMapper)[0] for fit in fit_list
    ]
    pixel_values_list = [
        fit.inversion.reconstruction_dict[mapper]
        for fit, mapper in zip(fit_list, mapper_list)
    ]
    extent_list = [
        mapper.extent_from(values=pixel_values)
        for mapper, pixel_values in zip(mapper_list, pixel_values_list)
    ]

    source_plane_extent = [
        np.min([extent[0] for extent in extent_list]),
        np.max([extent[1] for extent in extent_list]),
        np.min([extent[2] for extent in extent_list]),
        np.max([extent[3] for extent in extent_list]),
    ]

    vmax = (
        np.max([np.max(fit.model_images_of_planes_list[1]) for fit in fit_list]) / 2.0
    )

    plotter_main = aplt.FitImagingPlotter(
        fit=fit_list[0],
        mat_plot_2d=aplt.MatPlot2D(
            output=aplt.Output(path=output_path, filename=filename, format="png"),
        ),
    )

    plotter_main.open_subplot_figure(
        number_subplots=len(fit_list) * 6,
        subplot_shape=(len(fit_list), 6),
    )

    for i, fit in enumerate(fit_list):
        tag = tag_list[i] if tag_list is not None else ""

        title_fontweight = "bold" if i == main_dataset_index else "normal"

        plot_source_png_row(
            plotter_main=plotter_main,
            fit=fit,
            tag=f"{tag_prefix}{tag}",
            vmax=vmax,
            source_plane_extent=source_plane_extent,
            title_fontweight=title_fontweight,
        )

    plotter_main.mat_plot_2d.output.subplot_to_figure(auto_filename=filename)
    plotter_main.close_subplot_figure()


def plot_mge_only_row(
    plotter_main,
    fit,
    tag,
    mask,
    vmax_data,
    vmax_mge,
):
    """
    Plots a row of a subplot which shows a MGE lens light subtraction to a list of datasets (e.g. varying across
    wavelengths) where each row corresponds to a different dataset.

    Parameters
    ----------
    plotter_main
        The main plotter object that is used to create the subplot.
    fit
        The fit to the dataset that is plotted, which corresponds to a row in the subplot.
    tag
        The tag that labels the row of the subplot.
    mask
        The mask applied to the data, which is used to plot the residual map.
    vmax_data
        The maximum pixel value of the data subplot, which is chosen based on all fits in the list in order to make
        visual comparison easier.
    vmax_mge
        The maximum pixel value of the MGE lens light subtraction subplot, chosen based on all fits in the list.
    """
    vmax_mge_2 = vmax_mge / 3.0
    vmax_mge_3 = vmax_mge / 10.0

    visuals = aplt.Visuals2D(
        mask=mask,
    )

    plotter = aplt.FitImagingPlotter(
        fit=fit, visuals_2d=visuals, mat_plot_2d=aplt.MatPlot2D(use_log10=True)
    )

    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.mat_plot_2d.use_log10 = True
    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax_data)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.set_title(label=f"Data Pre MGE {tag}")
    plotter.figures_2d(data=True)
    plotter.mat_plot_2d.use_log10 = False

    plotter = aplt.FitImagingPlotter(fit=fit, visuals_2d=visuals)

    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax_data)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.set_title(label=f"Data Pre MGE {tag}")
    plotter.figures_2d(data=True)

    plotter = aplt.FitImagingPlotter(
        fit=fit,
        mat_plot_2d=aplt.MatPlot2D(
            cmap=aplt.Cmap(vmin=0.0, vmax=vmax_mge),
        ),
        visuals_2d=visuals,
    )

    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax_mge)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.set_title(label=f"MGE Subtraction {tag}")
    plotter.figures_2d(residual_map=True)

    plotter = aplt.FitImagingPlotter(fit=fit, visuals_2d=visuals)

    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax_mge_2)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.set_title(label=f"MGE Subtraction {tag}")
    plotter.figures_2d(residual_map=True)

    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax_mge_3)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.set_title(label=f"MGE Subtraction {tag}")
    plotter.figures_2d(residual_map=True)

    plotter.mat_plot_2d.use_log10 = True
    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=1.0e-3, vmax=vmax_mge)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.set_title(label=f"MGE Subtraction {tag}")
    plotter.figures_2d(residual_map=True)
    plotter.mat_plot_2d.use_log10 = False


def output_subplot_mge_only_png(
    output_path: str, result_list, tag_list=None, filename="mge_only"
):
    """
    Outputs a .png subplot of the MGE lens light subtraction (without mass or source models) to multiple
    datasets (e.g. varying across wavelengths) where each row corresponds to a different dataset.

    Many aspects of the plot are homogenized so that the fits can be compared easily.

    Parameters
    ----------
    output_path
        The path to the output directory where the modeling results are stored.
    result_list
        A list of results from the SLaM pipeline used to make the modeling results, typically the MASS PIPELINE.
    tag_list
        A list of tags to label each row of the subplot.
    filename
        The name of the file that the results are written to.
    """
    fit_list = [result.max_log_likelihood_fit for result in result_list]

    vmax_data = np.max([np.max(fit.data) for fit in fit_list]) / 2.0

    vmax_mge_list = []

    for fit in fit_list:
        image = fit.residual_map.native
        mask = al.Mask2D.circular(
            radius=0.3,
            pixel_scales=fit.dataset.pixel_scales,
            shape_native=image.shape_native,
        )

        vmax = image[mask].max()

        vmax_mge_list.append(vmax)

    vmax_mge = np.max(vmax_mge_list)

    plotter_main = aplt.FitImagingPlotter(
        fit=fit_list[0],
        mat_plot_2d=aplt.MatPlot2D(
            output=aplt.Output(path=output_path, filename=filename, format="png"),
        ),
    )

    plotter_main.open_subplot_figure(
        number_subplots=len(fit_list) * 6,
        subplot_shape=(len(fit_list), 6),
    )

    for i, fit in enumerate(fit_list):
        tag = tag_list[i] if tag_list is not None else ""

        plot_mge_only_row(
            plotter_main=plotter_main,
            fit=fit,
            tag=tag,
            mask=fit.mask,
            vmax_data=vmax_data,
            vmax_mge=vmax_mge,
        )

    plotter_main.mat_plot_2d.output.subplot_to_figure(auto_filename=filename)
    plotter_main.close_subplot_figure()


def analysis_multi_dataset_from(
    analysis: Union[af.Analysis, af.CombinedAnalysis],
    model,
    multi_dataset_offset: bool = False,
    multi_source_regularization: bool = False,
    source_regularization_result=None,
):
    """
    Updates the `Analysis` object to include free parameters for multi-dataset modeling.

    The following updates can be made:

    - The arc-second (y,x) offset between two datasets for multi-band fitting, where a different offset is used for each
      dataset (e.g. 2 extra free parameters per dataset).

    - The regularization parameters of the pixelization used to reconstruct the source, where different regularization
      parameters are used for each dataset (e.g. 1-3 extra free parameters per dataset).

    - The regularization parameters of the pixelization used to reconstruct the source are fixed to the max log likelihood
      instance of the regularization from a previous model-fit (e.g. the SOURCE PIPELINE).

    The function is quite rigid and should not be altered to change the behavior of the multi wavelength SLaM pipelines.
    Future updates will add more flexibility, once multi wavelength modeling is better understood.

    Parameters
    ----------
    analysis
        The sum of analysis classes that are used to fit the data.
    model
        The model used to fit the data, which is extended to include the extra free parameters.
    multi_dataset_offset
        If True, a different (y,x) arc-second offset is used for each dataset.
    multi_source_regularization
        If True, a different regularization parameters are used for each dataset.
    source_regularization_result
        The result of a previous model-fit that is used to fix the regularization parameters of the source pixelization.

    Returns
    -------
    The updated analysis object that includes the extra free parameters.
    """
    if not isinstance(analysis, af.CombinedAnalysis):
        return analysis

    if multi_dataset_offset and not multi_source_regularization:
        analysis = analysis.with_free_parameters(model.dataset_model.grid_offset)
    elif not multi_dataset_offset and multi_source_regularization:
        analysis = analysis.with_free_parameters(
            model.galaxies.source.pixelization.regularization
        )
    elif multi_dataset_offset and multi_source_regularization:
        analysis = analysis.with_free_parameters(
            model.dataset_model.grid_offset,
            model.galaxies.source.pixelization.regularization,
        )

    for i in range(1, len(analysis)):
        if multi_dataset_offset:
            analysis[i][
                model.dataset_model.grid_offset.grid_offset_0
            ] = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
            analysis[i][
                model.dataset_model.grid_offset.grid_offset_1
            ] = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

    if source_regularization_result is not None:
        for i in range(len(analysis)):
            analysis[i][
                model.galaxies.source.pixelization.regularization
            ] = source_regularization_result[
                i
            ].instance.galaxies.source.pixelization.regularization

    return analysis

"""
GUI Preprocessing: Extra Galaxies Mask Dataset
==============================================

This tool allows one to mask a bespoke noise-map for a given image of a strong lens, using a GUI.

This noise-map is primarily used for increasing the variances of pixels that have non-modeled components in an image,
for example intervening line-of-sight galaxies that are near the lens, but not directly interfering with the
analysis of the lens and source galaxies.

This GUI is adapted from the following code: https://gist.github.com/brikeats/4f63f867fd8ea0f196c78e9b835150ab
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
import json
from os import path
import autolens as al
import autolens.plot as aplt
import numpy as np

"""
__Dataset__

Setup the path the datasets we'll use to illustrate preprocessing, which is the 
folder `dataset/imaging/no_lens_light/mass_sie__source_sersic`.
"""

dataset_name = "example"
dataset_path = path.join("dataset", dataset_name)

psf_full = al.Kernel2D.from_fits(
    file_path=path.join(dataset_path, "psf_full.fits"), hdu=0, pixel_scales=0.1
)

print("PSF Shape Before Trimming:")
print(psf_full.shape_native)

new_shape = (11, 11)

print()
print(f"PSF being trimmed to {new_shape}")

psf = psf_full.resized_from(new_shape=new_shape)

psf.output_to_fits(file_path=path.join(dataset_path, "psf.fits"), overwrite=True)


"""
__Png output__
"""
mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(path=dataset_path, filename="psf_full", format="png"),
    use_log10=True,
)

plotter = aplt.Array2DPlotter(
    array=psf_full,
    mat_plot_2d=mat_plot_2d,
)
plotter.figure_2d()


mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(path=dataset_path, filename="psf", format="png"), use_log10=True
)

plotter = aplt.Array2DPlotter(
    array=psf,
    mat_plot_2d=mat_plot_2d,
)
plotter.figure_2d()

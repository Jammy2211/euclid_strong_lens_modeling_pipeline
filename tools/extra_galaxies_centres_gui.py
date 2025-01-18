"""
GUI Preprocessing: Extra Galaxies Centres
=========================================

There may be extra galaxies nearby the lens and source galaxies, whose emission blends with the lens and source
and whose mass may contribute to the ray-tracing and lens model.

This script uses a GUI to mark the (y,x) arcsecond locations of these extra galaxies, each of which is then included
in the lens model as light profiles (a Multi Gasusian Expansion) and mass profiles (singular isothermal spheres) if
the `groups.py` pipeline is used (the `start_here.py` pipeline does not include extra galaxies for simplicity).

Extra galaxies require that the mask is expanded to include their light, which is done by computing the radial
distance of the furthest extra galaxy from the origin and adding a buffer. This mask is used in the `groups.py`
pipeline to mask the lens and source galaxies and the extra galaxies.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import json
import numpy as np
import os
from os import path
from matplotlib import pyplot as plt

import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The path where the extra galaxy centres are output, which is `datasetgroup`.
"""
dataset_name = "EUCLJ174907.29+645946.3"
dataset_main_path = path.join("dataset", dataset_name)
dataset_path = path.join(dataset_main_path, "vis")

"""
The pixel scale of the imaging dataset.
"""
pixel_scales = 0.1

"""
Load the image which we will use to mark the lens light centre.
"""
data = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=pixel_scales
)

"""
__Mask__

Create a 3.0" mask to plot over the image to guide where points should be marked.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=data.shape_native,
    pixel_scales=data.pixel_scales,
    radius=mask_radius
)

grid = mask.derive_grid.edge

"""
__Search Box__

When you click on a pixel to mark a position, the search box looks around this click and finds the pixel with
the highest flux to mark the position.

The `search_box_size` is the number of pixels around your click this search takes place.
"""
search_box_size = 3

"""
__Clicker__

Set up the `Clicker` object from the `clicker.py` module, which monitors your mouse clicks in order to determine
the extra galaxy centres.
"""
clicker = al.Clicker(
    image=data, pixel_scales=pixel_scales, search_box_size=search_box_size
)

"""
Set up the clicker canvas and load the GUI which you can now click on to mark the extra galaxy centres.
"""
n_y, n_x = data.shape_native
hw = int(n_x / 2) * pixel_scales
ext = [-hw, hw, -hw, hw]
fig = plt.figure(figsize=(14, 14))
cmap = aplt.Cmap(cmap="jet", norm="log", vmin=1.0e-3, vmax=np.max(data) / 3.0)
norm = cmap.norm_from(array=data, use_log10=True)
plt.imshow(data.native, cmap="jet", norm=norm, extent=ext)
plt.scatter(y=grid[:, 0], x=grid[:, 1], c="k", marker="x", s=10)
plt.colorbar()
cid = fig.canvas.mpl_connect("button_press_event", clicker.onclick)
plt.show()
fig.canvas.mpl_disconnect(cid)
plt.close(fig)

"""
Use the results of the Clicker GUI to create the list of extra galaxy centres.
"""
extra_galaxies_centres = al.Grid2DIrregular(values=clicker.click_list)

"""
__Mask Radius__

The circular mask radius is calculated as the radial distance of the furthest extra galaxy from the origin,
with a buffer of 0.2" added to this value.

If the lens has no extra galaxies, the default mask radius of 3.0" is used instead.

This will be used as the circular mask radius of any modeling pipeline if a user input value is not input
(recommended behaviour).
"""
if len(extra_galaxies_centres) > 1:
    extra_galaxies_radii = np.sqrt(extra_galaxies_centres[:, 0] ** 2 + extra_galaxies_centres[:, 1] ** 2)
    extra_galaxies_radius_buffer = 0.2
    extra_galaxies_max_radius = np.max(extra_galaxies_radii) + extra_galaxies_radius_buffer
    mask_radius = max(mask_radius, extra_galaxies_max_radius)

info = {}

if os.path.exists(path.join(dataset_main_path, "info.json")):
    try:
        with open(path.join(dataset_main_path, "info.json")) as json_file:
            info = json.load(json_file)
            json_file.close()
    except FileNotFoundError:
        info = {}

info["mask_radius"] = mask_radius

with open(path.join(dataset_main_path, "info.json"), "w") as json_file:
    json.dump(info, json_file)
    json_file.close()

"""
__Output__

Output this image of the extra galaxy centres to a .png file in the dataset folder for future reference.
"""
mask = al.Mask2D.circular(
    shape_native=data.shape_native,
    pixel_scales=data.pixel_scales,
    radius=mask_radius
)

visuals = aplt.Visuals2D(
    mask=mask,
    mass_profile_centres=extra_galaxies_centres
)

array_2d_plotter = aplt.Array2DPlotter(
    array=data,
    visuals_2d=visuals,
    mat_plot_2d=aplt.MatPlot2D(
        mass_profile_centres_scatter=aplt.MassProfileCentresScatter(c="cy"),
        output=aplt.Output(
            path=dataset_main_path, filename="extra_galaxies_centres", format="png"
        ),
        use_log10=True
    ),
)
array_2d_plotter.figure_2d()

"""
Output the extra galaxy centres to the dataset folder of the lens, so that we can load them from a .json file 
when we model them.
"""
al.output_to_json(
    obj=extra_galaxies_centres,
    file_path=path.join(dataset_main_path, "extra_galaxies_centres.json"),
)
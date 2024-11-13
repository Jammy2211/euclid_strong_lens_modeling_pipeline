Euclid Strong Lens Modeling Pipeline
====================================

This repository makes it straightforward to use the Euclid strong lens modeling pipeline on your local machine
or a supercomputer.

The pipeline uses **PyAutoLens** to perform automated strong lens modeling, with this repository making it simple
to run the pipeline as a black-box on Euclid data.

Getting Started
---------------

**PyAutoLens** supports Python 3.9 to 3.11, with **Python 3.11 recommended**.

You first may want to set up a **Python virtual environment** or **conda enviuroment** to install the pipeline
in (see https://docs.python.org/3/library/venv.html).

Next, install **PyAutoLens** via pip:

.. code-block:: bash

    pip install --upgrade pip
    pip install autolens
    pip install numba

Clone the pipeline repository:

.. code-block:: bash

    git clone https://github.com/Jammy2211/euclid_strong_lens_modeling_pipeline
    cd euclid_strong_lens_modeling_pipeline

Run the pipeline with the example dataset:

.. code-block:: bash

    python start_here.py --dataset=example --mask_radius=2.0 --number_of_cores=4 --iterations_per_update=5000

The pipeline will run on the example dataset, outputting results to the `output` folder and in the `dataset` folder,
and it can be easily modified to run on your own data.

The pipeline above is parallelized using 4 cores and results are output on-the-fly during the model fitting
procedure every 5000 iterations.

Overview
--------

The main Euclid strong lens modeling pipeline is found in the ``start_here.py`` script, which is the script you run
to perform the lens modeling.

This script can be run as a black-box, with key output being generated, including:

- A SIE plus shear lens mass model.
- Deblended images of the lens and source galaxies.
- A pixelized source reconstruction.

Here is an example of the output, which shows the lens and source galaxies debelended and a source reconstruction
in the source-plane:

.. image:: https://github.com/Jammy2211/euclid_strong_lens_modeling_pipeline/blob/main/sie_fit.png?raw=true
  :width: 900

If key output for your science case is not generated, please contact James Nightingale on the Euclid consortium
SLACK so it can be added to the pipeline and become a standard output of the Euclid strong lens modeling pipeline
and therefore data release.

Additional Pipelines
--------------------

The following additional pipelines are available in the repository:

- ``group.py``: Lens modeling of group-scale lenses which have extra nearby galaxies whose light and mass must be modeled.
- ``mge_only.py``: Perform a fast Multi-Gaussian Expansion (MGE) subtraction of the lens light, in order to better visualize the lensed source.
- ``multi_wavelength.py``: After modeling the high resolution VIS imaging, model lower resolution NIR / EXT imaging using a fixed lens model.

All pipelines are run with the same API as the `start_here.py` script, for example:

.. code-block:: bash

    python groups.py --dataset=group --mask_radius=3.0 --number_of_cores=4 --iterations_per_update=5000

.. code-block:: bash

    python multi_wavelength.py --dataset=EUCLJ174517.55+655612.5 --mask_radius=2.0 --number_of_cores=4 --iterations_per_update=5000

Documentation
-------------

The following links are useful for anyone more interested in the **PyAutoLens** software:

- `The PyAutoLens readthedocs <https://pyautolens.readthedocs.io/en/latest>`_: which includes `an overview of PyAutoLens's core features <https://pyautolens.readthedocs.io/en/latest/overview/overview_1_start_here.html>`_, `a new user starting guide <https://pyautolens.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html>`_ and `an installation guide <https://pyautolens.readthedocs.io/en/latest/installation/overview.html>`_.

- `The introduction Jupyter Notebook on Binder <https://mybinder.org/v2/gh/Jammy2211/autolens_workspace/release?filepath=start_here.ipynb>`_: try **PyAutoLens** in a web browser (without installation).

- `The autolens_workspace GitHub repository <https://github.com/Jammy2211/autolens_workspace>`_: example scripts and the HowToLens Jupyter notebook lectures.

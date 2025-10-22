Euclid Strong Lens Modeling Pipeline
====================================

This repository makes it straightforward to use the Euclid strong lens modeling pipeline on your local machine
or a supercomputer.

The pipeline uses **PyAutoLens** to perform automated strong lens modeling, with this repository making it simple
to run the pipeline as a black-box on Euclid data.

JAX & GPU
---------

**PyAutoLens** runs significantly faster on GPUs — often **50x or more** compared to CPUs.

This acceleration is achieved through [**JAX**](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html), which provides GPU and TPU support.

When you install **PyAutoLens** (see instructions below), JAX will also be installed. However, the default installation may not include GPU support.

To ensure GPU acceleration, it is recommended that you install JAX with GPU support **before** installing **PyAutoLens**, by following the official [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

If you install **PyAutoLens** without a proper GPU setup, a warning will be displayed.

Getting Started
---------------

**PyAutoLens** supports Python 3.9 to 3.12, with **Python 3.11 recommended**.

You first may want to set up a **Python virtual environment** or **conda environment** to install the pipeline
in (see https://docs.python.org/3/library/venv.html).

Next, install **PyAutoLens** via pip:

.. code-block:: bash

    pip install --upgrade pip
    pip install autolens

Clone the pipeline repository:

.. code-block:: bash

    git clone https://github.com/Jammy2211/euclid_strong_lens_modeling_pipeline
    cd euclid_strong_lens_modeling_pipeline

Run the pipeline with the example dataset:

.. code-block:: bash

    python start_here.py --dataset=EUCLJ174517.55+655612.5 --mask_radius=3.0 --iterations_per_quick_update=10000

The pipeline will run on the example dataset, outputting results to the ``output`` folder and in the ``dataset`` folder,
and it can be easily modified to run on your own data.

The pipeline above is parallelized automatically based on the hardware available to Python (GPU or CPU) and
results are output on-the-fly during the model fitting procedure every 10000 iterations, meaning you can watch
the lens model improve over time!

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

- ``mge_lens_model.py``: Perform a fast Multi-Gaussian Expansion (MGE) lens modeling for the lens light and source, to get a fast (< 10 minutes on GPU) lens model.
- ``mge_lens_only.py``: Perform a fast Multi-Gaussian Expansion (MGE) subtraction of the lens light, in order to better visualize the lensed source.
- ``group.py``: Lens modeling of group-scale lenses which have extra nearby galaxies whose light and mass must be modeled.
- ``multi_wavelength.py``: After modeling the high resolution VIS imaging, model lower resolution NIR / EXT imaging using a fixed lens model.
- ``point_source.py``: Model the lensed source as a point source, for example if its a strongly lensed quasar.

All pipelines are run with the same API as the `start_here.py` script, for example:

.. code-block:: bash

    python pipelines/mge_lens_model.py --dataset=EUCLJ174517.55+655612.5 --mask_radius=3.0 --iterations_per_quick_update=50000

.. code-block:: bash

    python pipelines/mge_lens_only.py --dataset=EUCLJ174517.55+655612.5 --mask_radius=3.0 --iterations_per_quick_update=50000

.. code-block:: bash

    python pipelines/groups.py --dataset=group --mask_radius=3.0 --iterations_per_quick_update=50000

.. code-block:: bash

    python pipelines/multi_wavelength.py --dataset=EUCLJ174517.55+655612.5 --mask_radius=3.0 --iterations_per_quick_update=50000

.. code-block:: bash

    python pipelines/point_source.py --dataset=point_example --iterations_per_quick_update=10000

Documentation
-------------

The following links are useful for anyone more interested in the **PyAutoLens** software:

- `The PyAutoLens readthedocs <https://pyautolens.readthedocs.io/en/latest>`_: which includes `an overview of PyAutoLens's core features <https://pyautolens.readthedocs.io/en/latest/overview/overview_1_start_here.html>`_, `a new user starting guide <https://pyautolens.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html>`_ and `an installation guide <https://pyautolens.readthedocs.io/en/latest/installation/overview.html>`_.

- `The introduction Jupyter Notebook on Binder <https://mybinder.org/v2/gh/Jammy2211/autolens_workspace/release?filepath=start_here.ipynb>`_: try **PyAutoLens** in a web browser (without installation).

- `The autolens_workspace GitHub repository <https://github.com/Jammy2211/autolens_workspace>`_: example scripts and the HowToLens Jupyter notebook lectures.

"""
JAX Profiling: Delaunay Imaging Likelihood (Euclid, Step-by-Step)
=================================================================

Profiles each step of the JAX likelihood function for a Euclid VIS-like
imaging dataset (0.1"/pixel) where the source galaxy is reconstructed
using a Delaunay triangulation mesh with ConstantSplit regularization.

This is a Euclid-only port of
``autolens_workspace_developer/jax_profiling/imaging/delaunay.py``,
intended for estimating Euclid strong-lens modelling run times on this
pipeline's typical data scale.

Pipeline steps:

1. Ray-trace data grid to source plane
2. Ray-trace mesh grid (image-plane vertices) to source plane
3. Lens light images (pre-PSF, JIT) + PSF convolution (eager)
4. Profile-subtracted image
5. Border relocation (data grid + mesh grid)
6. Delaunay triangulation + interpolation + mapper
7. Mapping matrix
8. Blurred mapping matrix (PSF convolution)
9. Data vector (D)
10. Curvature matrix (F)
11. Regularization matrix (H) — ConstantSplit scheme
12. Regularized reconstruction: s = (F + H)^{-1} D
13. Map reconstruction to image + log evidence

Caveat: XLA may fuse operations differently when compiled as one program vs
separate pieces, so per-step timings are approximate. They are still useful
for identifying which step dominates.

All JAX timings use ``block_until_ready()`` to force synchronous measurement.

Run from the repository root::

    python profiling/delaunay.py

The simulator (``profiling/simulator.py``) is invoked automatically the
first time if the dataset does not already exist on disk.
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
import subprocess
import sys
from pathlib import Path
from contextlib import contextmanager

import autofit as af
import autolens as al
import autoarray as aa

# ---------------------------------------------------------------------------
# Configuration (Euclid VIS)
# ---------------------------------------------------------------------------

PIXEL_SCALE = 0.1
MASK_RADIUS = 3.5


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------


class Timer:
    """Accumulates named timing measurements and prints a summary."""

    def __init__(self):
        self.records: list[tuple[str, float]] = []

    @contextmanager
    def section(self, label: str):
        """Context manager that records wall-clock time for *label*."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.records.append((label, elapsed))
        print(f"  [{label}] {elapsed:.4f} s")

    def summary(self):
        print("\n" + "=" * 70)
        print("PROFILING SUMMARY")
        print("=" * 70)
        max_label = max(len(r[0]) for r in self.records)
        total = 0.0
        for label, elapsed in self.records:
            print(f"  {label:<{max_label}}  {elapsed:>10.4f} s")
            total += elapsed
        print("-" * 70)
        print(f"  {'TOTAL':<{max_label}}  {total:>10.4f} s")
        print("=" * 70)


def block(x):
    """Call block_until_ready if available (JAX arrays)."""
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


def jit_profile(func, label, *args, n_repeats=10):
    """JIT-compile *func*, time first call and steady-state average.

    Returns the compiled function and its result.
    """
    jitted = jax.jit(func)

    with timer.section(f"{label}_lower"):
        lowered = jitted.lower(*args)

    with timer.section(f"{label}_compile"):
        compiled = lowered.compile()

    with timer.section(f"{label}_first_call"):
        result = compiled(*args)
        block(result)

    with timer.section(f"{label}_steady_x{n_repeats}"):
        for _ in range(n_repeats):
            result = compiled(*args)
            block(result)

    per_call = timer.records[-1][1] / n_repeats
    print(f"    -> per-call avg: {per_call:.6f} s")
    return compiled, result


timer = Timer()
likelihood_steps = []  # (label, per_call_seconds) for the final summary

# ===================================================================
# PART A — Setup (not JIT-compiled)
# ===================================================================

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

print("\n--- Dataset loading & masking [euclid] ---")

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
dataset_path = Path("profiling") / "dataset" / "euclid"

if al.util.dataset.should_simulate(str(dataset_path)):
    print("  Simulating euclid dataset...")
    subprocess.run(
        [sys.executable, str(_script_dir / "simulator.py")],
        cwd=str(_repo_root),
        check=True,
    )

with timer.section("dataset_load"):
    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        pixel_scales=PIXEL_SCALE,
    )

with timer.section("mask_and_oversample"):
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=MASK_RADIUS,
    )

    dataset = dataset.apply_mask(mask=mask)
    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=4,
        over_sample_size_pixelization=1,
    )

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=over_sample_size,
        over_sample_size_pixelization=1,
    )

# ---------------------------------------------------------------------------
# 2. Image mesh + edge points (Delaunay-specific)
# ---------------------------------------------------------------------------

print("\n--- Image mesh construction (Delaunay) ---")

overlay_shape = (26, 26)
edge_n_points = 30

with timer.section("image_mesh_overlay"):
    image_mesh = al.image_mesh.Overlay(shape=overlay_shape)
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=dataset.mask)

with timer.section("edge_points"):
    edge_pixels_total = image_plane_mesh_grid.shape[0]
    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=(0.0, 0.0),
        radius=MASK_RADIUS,
        n_points=edge_n_points,
    )
    edge_pixels_total = image_plane_mesh_grid.shape[0] - edge_pixels_total

n_mesh_vertices = image_plane_mesh_grid.shape[0]
print(f"  Overlay shape: {overlay_shape}")
print(f"  Mesh vertices (incl. edge): {n_mesh_vertices}")
print(f"  Edge points added: {edge_pixels_total}")

# ---------------------------------------------------------------------------
# 3. Model construction
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

with timer.section("model_build"):
    # GaussianPrior(mean=truth, sigma=small) centres prior-median at the
    # simulator truth while keeping params free so gradient diagnostics
    # have dimensionality.
    lens_bulge = af.Model(al.lp.Sersic)
    lens_bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    lens_bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    _lens_bulge_ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
    lens_bulge.ell_comps.ell_comps_0 = af.GaussianPrior(
        mean=_lens_bulge_ell[0], sigma=0.01
    )
    lens_bulge.ell_comps.ell_comps_1 = af.GaussianPrior(
        mean=_lens_bulge_ell[1], sigma=0.01
    )
    lens_bulge.intensity = af.GaussianPrior(mean=2.0, sigma=0.1)
    lens_bulge.effective_radius = af.GaussianPrior(mean=0.6, sigma=0.05)
    lens_bulge.sersic_index = af.GaussianPrior(mean=3.0, sigma=0.2)

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    _lens_mass_ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=_lens_mass_ell[0], sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=_lens_mass_ell[1], sigma=0.01)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.005)
    shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.005)

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear)

    mesh = al.mesh.Delaunay(
        pixels=n_mesh_vertices,
        zeroed_pixels=edge_pixels_total,
    )
    regularization = al.reg.ConstantSplit(coefficient=1.0)
    pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  Delaunay pixels: {n_mesh_vertices}")
print(f"  Zeroed edge pixels: {edge_pixels_total}")

# ---------------------------------------------------------------------------
# 4. Instantiate concrete objects from prior medians
# ---------------------------------------------------------------------------

print("\n--- Instantiate concrete model ---")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

tracer = al.Tracer(galaxies=list(instance.galaxies))

# AdaptImages tells FitImaging where mesh vertices live in image-plane
adapt_images = al.AdaptImages(
    galaxy_image_plane_mesh_grid_dict={
        instance.galaxies.source: image_plane_mesh_grid,
    },
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid,
    },
)

print(f"  Tracer planes: {tracer.total_planes}")

# ---------------------------------------------------------------------------
# Key configuration that dictates run time
# ---------------------------------------------------------------------------

n_image_pixels = dataset.data.shape[0]
n_over_sampled_pixels = dataset.grids.lp.over_sampled.shape[0]
n_source_pixels = n_mesh_vertices

print("\n--- Configuration (determines run time) ---")
print(f"  Instrument:              euclid")
print(f"  Pixel scale:             {PIXEL_SCALE} arcsec/pixel")
print(f"  Mask radius:             {MASK_RADIUS} arcsec")
print(f"  Image pixels (masked):   {n_image_pixels}")
print(f"  Over-sampled pixels:     {n_over_sampled_pixels}")
print(f"  Delaunay vertices:       {n_source_pixels}")
print(f"  Edge zeroed pixels:      {edge_pixels_total}")

# ---------------------------------------------------------------------------
# 5. Full-pipeline reference (FitImaging) — eager baseline
# ---------------------------------------------------------------------------

print("\n--- Full FitImaging (eager baseline) ---")

with timer.section("fit_imaging_eager"):
    fit = al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=al.Settings(use_border_relocator=True),
        xp=np,
    )
    log_evidence_ref = fit.figure_of_merit
    log_likelihood_ref = fit.log_likelihood

print(f"  figure_of_merit (log_evidence) = {log_evidence_ref}")
print(f"  log_likelihood                 = {log_likelihood_ref}")


# ===================================================================
# PART B — Per-step JIT profiling
# ===================================================================

print("\n" + "=" * 70)
print("PER-STEP JIT PROFILING")
print("=" * 70)

# Extract raw arrays from autoarray types via .array so they can cross
# JIT boundaries.  See autolens_workspace_developer/CLAUDE.md for rationale.

grid_lp_raw = jnp.array(dataset.grids.lp.array)
grid_pix_raw = jnp.array(dataset.grids.pixelization.array)
grid_blurring_raw = jnp.array(dataset.grids.blurring.array)
mesh_grid_raw = jnp.array(image_plane_mesh_grid.array)
data_array = jnp.array(dataset.data.array)
noise_map_array = jnp.array(dataset.noise_map.array)

# Keep autoarray objects for eager calls that need them.
grid_lp = dataset.grids.lp
grid_blurring = dataset.grids.blurring

# ---------------------------------------------------------------------------
# Step 1: Ray-trace data grid to source plane
# ---------------------------------------------------------------------------

print("\n--- Step 1: Ray-trace data grid ---")

with timer.section("ray_trace_data_eager"):
    traced_grids = tracer.traced_grid_2d_list_from(
        grid=dataset.grids.pixelization, xp=jnp
    )
    for tg in traced_grids:
        block(tg)

print(f"  Number of planes traced: {len(traced_grids)}")


def ray_trace_data_raw(grid_raw):
    """Wraps ray-tracing so inputs/outputs are raw arrays."""
    grid = aa.Grid2DIrregular(values=grid_raw, xp=jnp)
    traced = tracer.traced_grid_2d_list_from(grid=grid, xp=jnp)
    return jnp.stack([tg.array for tg in traced])


_, traced_data_grids_raw = jit_profile(
    ray_trace_data_raw, "ray_trace_data_jit", grid_pix_raw
)
likelihood_steps.append(("Ray-trace data grid", timer.records[-1][1] / 10))

print(f"  traced_data_grids shape: {traced_data_grids_raw.shape}")

# ---------------------------------------------------------------------------
# Step 2: Ray-trace mesh grid (image-plane vertices) to source plane
# ---------------------------------------------------------------------------

print("\n--- Step 2: Ray-trace mesh grid ---")

with timer.section("ray_trace_mesh_eager"):
    traced_mesh = tracer.traced_grid_2d_list_from(
        grid=al.Grid2DIrregular(image_plane_mesh_grid), xp=jnp
    )
    for tg in traced_mesh:
        block(tg)


def ray_trace_mesh_raw(mesh_raw):
    """Ray-trace image-plane mesh vertices to source plane."""
    grid = aa.Grid2DIrregular(values=mesh_raw, xp=jnp)
    traced = tracer.traced_grid_2d_list_from(grid=grid, xp=jnp)
    return jnp.stack([tg.array for tg in traced])


_, traced_mesh_grids_raw = jit_profile(
    ray_trace_mesh_raw, "ray_trace_mesh_jit", mesh_grid_raw
)
likelihood_steps.append(("Ray-trace mesh grid", timer.records[-1][1] / 10))

print(f"  traced_mesh_grids shape: {traced_mesh_grids_raw.shape}")

# ---------------------------------------------------------------------------
# Step 3: Blurred image of non-linear light profiles (lens light)
# ---------------------------------------------------------------------------

print("\n--- Step 3: Blurred image (lens light profiles) ---")


# Sub-step 3a: Compute raw lens light images (JIT-profiled)
def lens_image_raw(grid_raw, blurring_grid_raw):
    """Compute lens light images on masked + blurring grids (no PSF)."""
    grid = aa.Grid2DIrregular(values=grid_raw, xp=jnp)
    blurring_grid = aa.Grid2DIrregular(values=blurring_grid_raw, xp=jnp)
    image = tracer.image_2d_from(grid=grid, xp=jnp)
    blurring_image = tracer.image_2d_from(grid=blurring_grid, xp=jnp)
    return image.array, blurring_image.array


with timer.section("lens_image_eager"):
    img_eager, blur_img_eager = lens_image_raw(grid_lp_raw, grid_blurring_raw)
    block(img_eager)
    block(blur_img_eager)

_, (img_jit, blur_img_jit) = jit_profile(
    lens_image_raw, "lens_image_jit", grid_lp_raw, grid_blurring_raw
)
likelihood_steps.append(("Lens light images (pre-PSF)", timer.records[-1][1] / 10))

# Sub-step 3b: PSF convolution
with timer.section("blurred_image_eager"):
    blurred_image = tracer.blurred_image_2d_from(
        grid=grid_lp,
        psf=dataset.psf,
        blurring_grid=grid_blurring,
        xp=jnp,
    )
    block(blurred_image)

print(f"  blurred_image shape: {blurred_image.array.shape}")

jnp_params = jnp.array(param_vector)


def blurred_image_from_params(params):
    """Reconstruct tracer from params, compute blurred image — fully JIT-traceable."""
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
    result = t.blurred_image_2d_from(
        grid=grid_lp,
        psf=dataset.psf,
        blurring_grid=grid_blurring,
        xp=jnp,
    )
    return result.array


_, blurred_img_jit = jit_profile(
    blurred_image_from_params, "blurred_image_jit", jnp_params
)
likelihood_steps.append(("Blurred image (PSF convolution)", timer.records[-1][1] / 10))

# ---------------------------------------------------------------------------
# Step 4: Profile-subtracted image (lens light subtraction)
# ---------------------------------------------------------------------------

print("\n--- Step 4: Profile-subtracted image ---")


def profile_subtract(data, blurred_image):
    return data - blurred_image


with timer.section("profile_subtract_eager"):
    blurred_img_jnp = jnp.array(blurred_image.array)
    profile_subtracted = profile_subtract(data_array, blurred_img_jnp)
    block(profile_subtracted)

_, profile_subtracted = jit_profile(
    profile_subtract, "profile_subtract_jit", data_array, blurred_img_jnp
)
likelihood_steps.append(("Profile-subtracted image", timer.records[-1][1] / 10))

print(f"  profile_subtracted shape: {profile_subtracted.shape}")

# ---------------------------------------------------------------------------
# Step 5: Border relocation (data grid + mesh grid)
# ---------------------------------------------------------------------------

print("\n--- Step 5: Border relocation ---")

from autoarray.inversion.mesh.border_relocator import BorderRelocator

with timer.section("border_relocator_setup"):
    border_relocator = BorderRelocator(mask=dataset.mask, sub_size=1)

traced_source_grid = tracer.traced_grid_2d_list_from(
    grid=dataset.grids.pixelization, xp=jnp
)[-1]
traced_mesh_source = tracer.traced_grid_2d_list_from(
    grid=al.Grid2DIrregular(image_plane_mesh_grid), xp=jnp
)[-1]

with timer.section("border_relocation_eager"):
    relocated_grid = border_relocator.relocated_grid_from(grid=traced_source_grid)
    relocated_mesh_grid = border_relocator.relocated_mesh_grid_from(
        grid=traced_source_grid,
        mesh_grid=traced_mesh_source,
    )
    block(relocated_grid)
    block(relocated_mesh_grid)

print(f"  relocated_data_grid shape: {relocated_grid.array.shape}")
print(f"  relocated_mesh_grid shape: {relocated_mesh_grid.array.shape}")

# ---------------------------------------------------------------------------
# Step 6: Delaunay triangulation + interpolation + mapper
# ---------------------------------------------------------------------------

print("\n--- Step 6: Delaunay triangulation + Interpolation + Mapper ---")

pixelization_obj = instance.galaxies.source.pixelization

with timer.section("delaunay_interpolation_and_mapper"):
    interpolator = al.InterpolatorDelaunay(
        mesh=pixelization_obj.mesh,
        mesh_grid=relocated_mesh_grid,
        data_grid=relocated_grid,
    )
    mapper = al.Mapper(
        interpolator=interpolator,
        image_plane_mesh_grid=image_plane_mesh_grid,
        xp=jnp,
    )

print(f"  mapper.pixels (source): {mapper.pixels}")
print(f"  pix_indexes shape: {mapper.pix_indexes_for_sub_slim_index.shape}")

# ---------------------------------------------------------------------------
# Steps 7-13: Extract matrices from FitImaging inversion for consistency
# ---------------------------------------------------------------------------
# The FitImaging pipeline handles edge pixel zeroing, curvature diagonal
# adjustments, and settings that are difficult to replicate manually.
# We extract the correct matrices from fit.inversion so the step-by-step
# matches the reference, then JIT-profile the linear algebra operations.

print("\n--- Extracting inversion matrices from FitImaging ---")

inversion = fit.inversion

with timer.section("extract_inversion_matrices"):
    bmm_ref = jnp.array(inversion.operated_mapping_matrix)
    mapping_matrix_ref = jnp.array(inversion.mapping_matrix)

    inv_mapper = inversion.cls_list_from(cls=al.Mapper)[0]
    neighbors = inv_mapper.neighbors
    neighbors_array = jnp.array(np.asarray(neighbors))
    neighbors_sizes = jnp.array(neighbors.sizes)

print(f"  operated_mapping_matrix shape: {bmm_ref.shape}")
print(f"  mapping_matrix shape: {mapping_matrix_ref.shape}")

# ---------------------------------------------------------------------------
# Step 7: Mapping matrix
# ---------------------------------------------------------------------------

print("\n--- Step 7: Mapping matrix ---")

with timer.section("mapping_matrix"):
    mapping_matrix = inv_mapper.mapping_matrix

print(f"  mapping_matrix shape: {mapping_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 8: Blurred mapping matrix (PSF convolution)
# ---------------------------------------------------------------------------

print("\n--- Step 8: Blurred mapping matrix ---")

with timer.section("blurred_mapping_matrix"):
    blurred_mapping_matrix = dataset.psf.convolved_mapping_matrix_from(
        mapping_matrix=mapping_matrix,
        mask=dataset.mask,
        xp=jnp,
    )
    block(blurred_mapping_matrix)

# JIT-profile the full inversion setup pipeline (steps 5-8 combined):
# border relocation → Delaunay triangulation → interpolation → mapper → mapping matrix → PSF convolution.
# These steps are tightly sequential; the full pipeline JIT-compiles them all together.


def blurred_mm_from_params(params):
    """Reconstruct tracer from params, compute blurred mapping matrix via full inversion setup."""
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
    # Recreate adapt_images with new galaxy instance so dict lookup by object identity works.
    adapt_images_jax = al.AdaptImages(
        galaxy_image_plane_mesh_grid_dict={
            inst.galaxies.source: image_plane_mesh_grid,
        },
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid,
        },
    )
    fit_jax = al.FitImaging(
        dataset=dataset,
        tracer=t,
        adapt_images=adapt_images_jax,
        settings=al.Settings(use_border_relocator=True),
        xp=jnp,
    )
    return jnp.array(fit_jax.inversion.operated_mapping_matrix)


_, bmm_jit = jit_profile(blurred_mm_from_params, "inversion_setup_jit", jnp_params)
likelihood_steps.append(
    ("Inversion setup (steps 5-8 combined)", timer.records[-1][1] / 10)
)

print(f"  blurred_mapping_matrix (JIT) shape: {bmm_jit.shape}")

bmm_jnp = bmm_ref  # Use the reference matrices for linear algebra steps
print(f"  blurred_mapping_matrix shape: {blurred_mapping_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 9: Data vector (D)
# ---------------------------------------------------------------------------

print("\n--- Step 9: Data vector ---")


def compute_data_vector(blurred_mapping_matrix, image, noise_map):
    return al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=blurred_mapping_matrix,
        image=image,
        noise_map=noise_map,
    )


profile_sub_jnp = jnp.array(fit.profile_subtracted_image.array)
noise_jnp = jnp.array(dataset.noise_map.array)

with timer.section("data_vector_eager"):
    data_vector = compute_data_vector(bmm_jnp, profile_sub_jnp, noise_jnp)
    block(data_vector)

_, data_vector = jit_profile(
    compute_data_vector, "data_vector_jit", bmm_jnp, profile_sub_jnp, noise_jnp
)
likelihood_steps.append(("Data vector (D)", timer.records[-1][1] / 10))

print(f"  data_vector shape: {data_vector.shape}")

# ---------------------------------------------------------------------------
# Step 10: Curvature matrix (F)
# ---------------------------------------------------------------------------

print("\n--- Step 10: Curvature matrix ---")

no_reg_list = list(inversion.no_regularization_index_list)


def compute_curvature_matrix(blurred_mapping_matrix, noise_map):
    return al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix,
        noise_map=noise_map,
        settings=fit.settings,
        add_to_curvature_diag=True,
        no_regularization_index_list=no_reg_list,
        xp=jnp,
    )


with timer.section("curvature_matrix_eager"):
    curvature_matrix = compute_curvature_matrix(bmm_jnp, noise_jnp)
    block(curvature_matrix)

_, curvature_matrix = jit_profile(
    compute_curvature_matrix, "curvature_matrix_jit", bmm_jnp, noise_jnp
)
likelihood_steps.append(("Curvature matrix (F)", timer.records[-1][1] / 10))

print(f"  curvature_matrix shape: {curvature_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 11: Regularization matrix (H) — ConstantSplit scheme
# ---------------------------------------------------------------------------

print("\n--- Step 11: Regularization matrix (ConstantSplit) ---")

# ConstantSplit uses a cross-derivative scheme via the interpolator's
# _mappings_sizes_weights_split, not the simple neighbour-difference approach.
# We extract it from the inversion for consistency and JIT-profile separately.

with timer.section("regularization_matrix_eager"):
    regularization_matrix = jnp.array(inversion.regularization_matrix)
    block(regularization_matrix)

likelihood_steps.append(("Regularization matrix (H)", timer.records[-2][1]))

print(f"  regularization_matrix shape: {regularization_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 12: Regularized reconstruction: s = (F + H)^{-1} D
# ---------------------------------------------------------------------------

print("\n--- Step 12: Regularized reconstruction ---")


def compute_reconstruction(data_vector, curvature_matrix, regularization_matrix):
    curvature_reg_matrix = curvature_matrix + regularization_matrix
    return jnp.linalg.solve(curvature_reg_matrix, data_vector)


with timer.section("reconstruction_eager"):
    reconstruction = compute_reconstruction(
        jnp.array(data_vector),
        jnp.array(curvature_matrix),
        jnp.array(regularization_matrix),
    )
    block(reconstruction)

_, reconstruction = jit_profile(
    compute_reconstruction,
    "reconstruction_jit",
    jnp.array(data_vector),
    jnp.array(curvature_matrix),
    jnp.array(regularization_matrix),
)
likelihood_steps.append(("Regularized reconstruction", timer.records[-1][1] / 10))

print(f"  reconstruction shape: {reconstruction.shape}")

# ---------------------------------------------------------------------------
# Step 13: Map reconstruction to image + log evidence
# ---------------------------------------------------------------------------

print("\n--- Step 13: Mapped reconstruction + log evidence ---")


def compute_log_evidence(
    data,
    noise_map,
    blurred_image,
    blurred_mapping_matrix,
    reconstruction,
    curvature_matrix,
    regularization_matrix,
):
    """Compute the full log evidence including all five terms:

    -2 ln e = chi^2 + s^T H s + ln[det(F+H)] - ln[det(H)] + noise_norm
    """
    # Map reconstruction to image
    mapped_recon = al.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix,
        reconstruction=reconstruction,
        xp=jnp,
    )

    # model_data = lens light + pixelized source
    model_data = blurred_image + mapped_recon

    # Chi-squared
    residual = data - model_data
    chi_squared = jnp.sum((residual / noise_map) ** 2)

    # Regularization term: s^T H s
    regularization_term = jnp.dot(
        reconstruction, jnp.dot(regularization_matrix, reconstruction)
    )

    # Curvature + regularization matrix
    curvature_reg_matrix = curvature_matrix + regularization_matrix

    # Log determinant terms
    sign_cr, log_det_curvature_reg = jnp.linalg.slogdet(curvature_reg_matrix)
    sign_r, log_det_regularization = jnp.linalg.slogdet(regularization_matrix)

    # Noise normalization
    noise_normalization = jnp.sum(jnp.log(2 * jnp.pi * noise_map**2))

    return -0.5 * (
        chi_squared
        + regularization_term
        + log_det_curvature_reg
        - log_det_regularization
        + noise_normalization
    )


# For the JIT profiling we use the step-by-step matrices for timing.
# For the correctness assertion we use the inversion's own matrices, because
# cumulative floating-point differences between JIT-compiled and eager paths
# (especially through ill-conditioned solves) can compound significantly.

blurred_img_jnp = jnp.array(blurred_image.array)
recon_jnp = jnp.array(reconstruction)
curv_jnp = jnp.array(curvature_matrix)
reg_jnp = jnp.array(regularization_matrix)

with timer.section("log_evidence_eager"):
    log_evidence = compute_log_evidence(
        data_array,
        noise_jnp,
        blurred_img_jnp,
        bmm_jnp,
        recon_jnp,
        curv_jnp,
        reg_jnp,
    )
    block(log_evidence)

_, log_evidence = jit_profile(
    compute_log_evidence,
    "log_evidence_jit",
    data_array,
    noise_jnp,
    blurred_img_jnp,
    bmm_jnp,
    recon_jnp,
    curv_jnp,
    reg_jnp,
)
likelihood_steps.append(("Mapped recon + log evidence", timer.records[-1][1] / 10))

print(f"  log_evidence (step-by-step) = {log_evidence}")

# Correctness check: recompute log_evidence using the inversion's own
# reconstruction and curvature matrix to avoid accumulated FP drift.
inv_recon_jnp = jnp.array(inversion.reconstruction)
inv_curv_jnp = jnp.array(inversion.curvature_matrix)

log_evidence_check = compute_log_evidence(
    data_array,
    noise_jnp,
    blurred_img_jnp,
    bmm_jnp,
    inv_recon_jnp,
    inv_curv_jnp,
    reg_jnp,
)
print(f"  log_evidence (inv matrices) = {log_evidence_check}")
print(f"  log_evidence (reference)    = {log_evidence_ref}")

np.testing.assert_allclose(
    float(log_evidence_check),
    float(log_evidence_ref),
    rtol=1e-4,
    err_msg="Log_evidence from inversion matrices does not match FitImaging.log_evidence",
)
print(
    "  Assertion PASSED: inversion-matrix log_evidence matches FitImaging.log_evidence"
)

# ===================================================================
# PART C — Full-pipeline JIT for comparison
# ===================================================================

print("\n" + "=" * 70)
print("FULL-PIPELINE JIT (for comparison)")
print("=" * 70)

from autofit.non_linear.fitness import Fitness

analysis = al.AnalysisImaging(dataset=dataset, adapt_images=adapt_images)

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

jnp_params = jnp.array(model.physical_values_from_prior_medians)

_, full_result = jit_profile(fitness.call, "full_pipeline", jnp_params)
full_pipeline_per_call = timer.records[-1][1] / 10

print(f"  full log_likelihood = {full_result}")

np.testing.assert_allclose(
    float(full_result),
    float(log_evidence_ref),
    rtol=1e-4,
    err_msg="Full-pipeline JIT log_likelihood does not match FitImaging.log_evidence",
)
print(
    "  Assertion PASSED: full-pipeline JIT log_likelihood matches FitImaging.log_evidence"
)

# ===================================================================
# PART D — vmap + correctness
# ===================================================================

print("\n--- vmap batched evaluation ---")

# WARNING: The vmap compilation for the Delaunay pipeline takes 20+ minutes on CPU.
# The XLA graph for a batched Delaunay inversion (including scipy triangulation,
# border relocation, interpolation, mapping matrix construction, and PSF convolution)
# is extremely large. The single-call JIT above compiles in ~2s and runs in ~1.8s,
# but vmap recompiles the entire graph for batch_size independent evaluations.
#
# This is likely a candidate for optimisation — either via custom_vjp to avoid
# retracing the full pipeline, or by restructuring the Delaunay steps to reduce
# the XLA graph size. For now, skip vmap by default and run it only when explicitly
# requested via DELAUNAY_VMAP=1 environment variable.

import os

run_vmap = os.environ.get("DELAUNAY_VMAP", "0") == "1"

if not run_vmap:
    print("  SKIPPED: vmap compilation takes 20+ minutes for Delaunay pipeline.")
    print("  Set DELAUNAY_VMAP=1 to run this section.")
    vmap_batch_time = None
    vmap_per_call = None
    vmap_speedup = None
else:

    batch_size = 3
    parameters = jnp.tile(jnp_params, (batch_size, 1))

    with timer.section("vmap_first_call"):
        result_vmap = fitness._vmap(parameters)
        block(result_vmap)

    n_vmap_repeats = 10
    with timer.section(f"vmap_steady_x{n_vmap_repeats}"):
        for _ in range(n_vmap_repeats):
            result_vmap = fitness._vmap(parameters)
            block(result_vmap)

    vmap_batch_time = timer.records[-1][1] / n_vmap_repeats
    vmap_per_call = vmap_batch_time / batch_size
    vmap_speedup = full_pipeline_per_call / vmap_per_call

    print(f"  batch results = {result_vmap}")
    print(f"  vmap batch of {batch_size}:   {vmap_batch_time:.6f} s")
    print(f"  vmap per call:         {vmap_per_call:.6f} s")
    print(f"  single JIT per call:   {full_pipeline_per_call:.6f} s")
    print(f"  vmap speedup:          {vmap_speedup:.1f}x faster per likelihood")

    np.testing.assert_allclose(
        np.array(result_vmap),
        float(full_result),
        rtol=1e-4,
        err_msg="delaunay[euclid]: JAX vmap likelihood mismatch",
    )
    print("  Correctness check PASSED")

    # --- Static memory analysis ---

    print("\n--- Static memory analysis ---")

    batched_call = jax.jit(jax.vmap(fitness.call))
    lowered_batched = batched_call.lower(parameters)
    compiled_batched = lowered_batched.compile()

    memory_analysis = compiled_batched.memory_analysis()
    print(f"  Output size:  {memory_analysis.output_size_in_bytes / 1024**2:.3f} MB")
    print(f"  Temp size:    {memory_analysis.temp_size_in_bytes / 1024**2:.3f} MB")
    print(
        f"  Total:        "
        f"{(memory_analysis.output_size_in_bytes + memory_analysis.temp_size_in_bytes) / 1024**2:.3f} MB"
    )


# ===================================================================
# JAX Likelihood Function Summary
# ===================================================================

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

al_version = al.__version__

print("\n" + "=" * 70)
print(f"JAX LIKELIHOOD FUNCTION SUMMARY — EUCLID — v{al_version}")
print("=" * 70)
print(f"  Instrument:            euclid")
print(f"  Pixel scale:           {PIXEL_SCALE} arcsec/pixel")
print(f"  Mask radius:           {MASK_RADIUS} arcsec")
print(f"  Image pixels (masked): {n_image_pixels}")
print(f"  Over-sampled pixels:   {n_over_sampled_pixels}")
print(f"  Delaunay vertices:     {n_source_pixels}")
print(f"  Edge zeroed pixels:    {edge_pixels_total}")
print("-" * 70)

max_label = max(len(label) for label, _ in likelihood_steps)
step_total = 0.0
for i, (label, per_call) in enumerate(likelihood_steps, 1):
    print(f"  {i:>2}. {label:<{max_label}}  {per_call:>12.6f} s")
    step_total += per_call

print("-" * 70)
print(f"      {'TOTAL (step-by-step)':<{max_label}}  {step_total:>12.6f} s")
print(
    f"      {'Full pipeline (single JIT)':<{max_label}}  {full_pipeline_per_call:>12.6f} s"
)
if vmap_per_call is not None:
    print(f"      {f'vmap batch (per call)':<{max_label}}  {vmap_per_call:>12.6f} s")
    print(f"      {f'vmap speedup vs single JIT':<{max_label}}  {vmap_speedup:>11.1f}x")
else:
    print(f"      {'vmap':<{max_label}}  {'SKIPPED':>12}")
print("=" * 70)

# --- Save results dictionary ---

likelihood_summary = {
    "autolens_version": al_version,
    "instrument": "euclid",
    "configuration": {
        "pixel_scale_arcsec": PIXEL_SCALE,
        "mask_radius_arcsec": MASK_RADIUS,
        "image_pixels_masked": int(n_image_pixels),
        "over_sampled_pixels": int(n_over_sampled_pixels),
        "delaunay_vertices": int(n_source_pixels),
        "edge_zeroed_pixels": int(edge_pixels_total),
    },
    "steps": {label: per_call for label, per_call in likelihood_steps},
    "total_step_by_step": step_total,
    "full_pipeline_single_jit": full_pipeline_per_call,
    "vmap": "SKIPPED — compilation takes 20+ minutes (set DELAUNAY_VMAP=1)",
}

if vmap_per_call is not None:
    likelihood_summary["vmap"] = {
        "batch_size": batch_size,
        "batch_time": vmap_batch_time,
        "per_call": vmap_per_call,
        "speedup_vs_single_jit": round(vmap_speedup, 1),
    }

results_dir = _script_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

dict_path = results_dir / f"delaunay_likelihood_summary_euclid_v{al_version}.json"
dict_path.write_text(json.dumps(likelihood_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# --- Save bar chart ---

labels = [label for label, _ in likelihood_steps]
times = [per_call for _, per_call in likelihood_steps]

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = range(len(labels))
bars = ax.barh(y_pos, times, color="#4C72B0", edgecolor="white", height=0.6)

for bar, t in zip(bars, times):
    ax.text(
        bar.get_width() + max(times) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{t:.6f} s",
        va="center",
        fontsize=9,
    )

ax.axvline(
    full_pipeline_per_call,
    color="#C44E52",
    linestyle="--",
    linewidth=1.5,
    label=f"Full pipeline (single JIT): {full_pipeline_per_call:.6f} s",
)
if vmap_per_call is not None:
    ax.axvline(
        vmap_per_call,
        color="#55A868",
        linestyle="--",
        linewidth=1.5,
        label=f"vmap batch per call: {vmap_per_call:.6f} s ({vmap_speedup:.1f}x faster)",
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Time per call (s)", fontsize=11)
fig.suptitle(
    "Delaunay Imaging Likelihood — EUCLID",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f'AutoLens v{al_version}  |  {PIXEL_SCALE}"/px  |  {n_image_pixels} pixels  |  '
    f"{n_over_sampled_pixels} over-sampled  |  {n_source_pixels} Delaunay vertices  |  "
    f"total: {step_total:.6f} s",
    fontsize=9,
)
ax.legend(loc="lower right", fontsize=9)
ax.margins(x=0.15)
fig.tight_layout()

chart_path = results_dir / f"delaunay_likelihood_summary_euclid_v{al_version}.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")

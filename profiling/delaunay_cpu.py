"""
CPU Profiling: Delaunay Imaging Likelihood with Sparse Operator (Euclid)
========================================================================

Profiles the Delaunay imaging likelihood on CPU using the **sparse
numba operator** (``dataset.apply_sparse_operator_cpu()``) instead of
JAX. This is the direct CPU counterpart to ``profiling/delaunay.py``:
same Euclid dataset, same mesh, same model — swapped backend.

The sparse operator precomputes a PSF precision operator that exploits
matrix sparsity during pixelized source reconstruction. On many-core
CPUs it can outperform JAX-on-CPU for pixelized sources; see
``autolens_workspace/scripts/imaging/features/pixelization/cpu_fast_modeling.py``
for the reference usage pattern.

Pipeline steps (shared with ``delaunay.py`` for direct comparison):

1. Ray-trace data grid to source plane
2. Ray-trace mesh grid to source plane
3. Lens light images (pre-PSF) + PSF convolution
4. Profile-subtracted image
5. Border relocation (data grid + mesh grid)
6. Delaunay triangulation + interpolation + mapper
7. Mapping matrix
8. Blurred mapping matrix (PSF convolution)
9. Data vector (D)

The full inversion math (curvature matrix, reconstruction, log
evidence) is timed only via ``FitImaging`` in Part C, because that is
the path the sparse operator actually accelerates — calling
``al.util.inversion.*`` directly would bypass it.

Run from the repository root::

    python profiling/delaunay_cpu.py

The simulator (``profiling/simulator.py``) is invoked automatically the
first time if the dataset is not on disk.
"""

import numpy as np
import time
import subprocess
import sys
from pathlib import Path
from contextlib import contextmanager

import autofit as af
import autolens as al

# ---------------------------------------------------------------------------
# Configuration (Euclid VIS)
# ---------------------------------------------------------------------------

PIXEL_SCALE = 0.1
MASK_RADIUS = 3.5
N_REPEATS = 10


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


def eager_profile(func, label, *args, n_repeats=N_REPEATS):
    """Run *func* once as warmup, then *n_repeats* times for steady state.

    Returns the function's last result. Numba-compiled paths may pay
    compilation cost on the first call; we time warmup and steady state
    separately and report the steady-state per-call average.
    """
    with timer.section(f"{label}_warmup"):
        result = func(*args)

    with timer.section(f"{label}_steady_x{n_repeats}"):
        for _ in range(n_repeats):
            result = func(*args)

    per_call = timer.records[-1][1] / n_repeats
    print(f"    -> per-call avg: {per_call:.6f} s")
    return result


timer = Timer()
likelihood_steps = []  # (label, per_call_seconds) for the final summary

# ===================================================================
# PART A — Setup
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
# 2. Sparse operator precompute — the one-off setup cost for CPU modelling
# ---------------------------------------------------------------------------

print("\n--- Sparse operator precompute (numba CPU) ---")

with timer.section("apply_sparse_operator_cpu"):
    dataset = dataset.apply_sparse_operator_cpu()

sparse_operator_setup_time = timer.records[-1][1]
print(f"  Sparse operator ready: {sparse_operator_setup_time:.4f} s")

# ---------------------------------------------------------------------------
# 3. Image mesh + edge points (Delaunay-specific)
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
# 4. Model construction
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

with timer.section("model_build"):
    lens_bulge = af.Model(al.lp.Sersic)
    lens_bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    lens_bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    _lens_bulge_ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
    lens_bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=_lens_bulge_ell[0], sigma=0.01)
    lens_bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=_lens_bulge_ell[1], sigma=0.01)
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

    lens = af.Model(
        al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear
    )

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
# 5. Instantiate concrete objects from prior medians
# ---------------------------------------------------------------------------

print("\n--- Instantiate concrete model ---")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

tracer = al.Tracer(galaxies=list(instance.galaxies))

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
# Configuration summary
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
print(f"  Sparse operator setup:   {sparse_operator_setup_time:.4f} s (one-off)")

# ---------------------------------------------------------------------------
# 6. Full-pipeline reference (FitImaging) — used as correctness oracle
# ---------------------------------------------------------------------------

print("\n--- Full FitImaging (reference) ---")

with timer.section("fit_imaging_reference"):
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
# PART B — Per-step eager profiling (numpy, no sparse operator path)
# ===================================================================
#
# Steps 1-9 are building-block operations that do NOT touch the sparse
# operator — they are the same calls the JAX profiler makes, just with
# xp=np. Timing them here gives a direct CPU-vs-JAX comparison for the
# non-inversion side of the likelihood.
#
# Steps 10-13 (curvature matrix, regularization, reconstruction, log
# evidence) are deliberately skipped at the step level because calling
# al.util.inversion.* with xp=np bypasses the sparse operator — the
# numbers would not reflect real run-time on the CPU pixelized path.
# They are captured instead by the full-pipeline FitImaging timing in
# Part C, which is where the sparse operator actually kicks in.

print("\n" + "=" * 70)
print("PER-STEP PROFILING (eager, numpy)")
print("=" * 70)

grid_lp = dataset.grids.lp
grid_pix = dataset.grids.pixelization
grid_blurring = dataset.grids.blurring

# ---------------------------------------------------------------------------
# Step 1: Ray-trace data grid
# ---------------------------------------------------------------------------

print("\n--- Step 1: Ray-trace data grid ---")

def ray_trace_data():
    return tracer.traced_grid_2d_list_from(grid=grid_pix, xp=np)

traced_grids = eager_profile(ray_trace_data, "ray_trace_data")
likelihood_steps.append(("Ray-trace data grid", timer.records[-1][1] / N_REPEATS))
print(f"  Number of planes traced: {len(traced_grids)}")

# ---------------------------------------------------------------------------
# Step 2: Ray-trace mesh grid
# ---------------------------------------------------------------------------

print("\n--- Step 2: Ray-trace mesh grid ---")

_mesh_irregular = al.Grid2DIrregular(image_plane_mesh_grid)

def ray_trace_mesh():
    return tracer.traced_grid_2d_list_from(grid=_mesh_irregular, xp=np)

traced_mesh = eager_profile(ray_trace_mesh, "ray_trace_mesh")
likelihood_steps.append(("Ray-trace mesh grid", timer.records[-1][1] / N_REPEATS))

# ---------------------------------------------------------------------------
# Step 3: Blurred lens light image (PSF convolution)
# ---------------------------------------------------------------------------

print("\n--- Step 3: Blurred image (lens light + PSF) ---")

def blurred_image_compute():
    return tracer.blurred_image_2d_from(
        grid=grid_lp,
        psf=dataset.psf,
        blurring_grid=grid_blurring,
        xp=np,
    )

blurred_image = eager_profile(blurred_image_compute, "blurred_image")
likelihood_steps.append(("Blurred image (lens light + PSF)", timer.records[-1][1] / N_REPEATS))

# ---------------------------------------------------------------------------
# Step 4: Profile-subtracted image
# ---------------------------------------------------------------------------

print("\n--- Step 4: Profile-subtracted image ---")

data_array = np.asarray(dataset.data.array)
blurred_img_np = np.asarray(blurred_image.array)

def profile_subtract():
    return data_array - blurred_img_np

profile_subtracted = eager_profile(profile_subtract, "profile_subtract")
likelihood_steps.append(("Profile-subtracted image", timer.records[-1][1] / N_REPEATS))

# ---------------------------------------------------------------------------
# Step 5: Border relocation
# ---------------------------------------------------------------------------

print("\n--- Step 5: Border relocation ---")

from autoarray.inversion.mesh.border_relocator import BorderRelocator

border_relocator = BorderRelocator(mask=dataset.mask, sub_size=1)

traced_source_grid = tracer.traced_grid_2d_list_from(grid=grid_pix, xp=np)[-1]
traced_mesh_source = tracer.traced_grid_2d_list_from(grid=_mesh_irregular, xp=np)[-1]

def border_relocation():
    relocated_grid = border_relocator.relocated_grid_from(grid=traced_source_grid)
    relocated_mesh_grid = border_relocator.relocated_mesh_grid_from(
        grid=traced_source_grid, mesh_grid=traced_mesh_source,
    )
    return relocated_grid, relocated_mesh_grid

relocated_grid, relocated_mesh_grid = eager_profile(border_relocation, "border_relocation")
likelihood_steps.append(("Border relocation", timer.records[-1][1] / N_REPEATS))

# ---------------------------------------------------------------------------
# Step 6: Delaunay triangulation + mapper construction
# ---------------------------------------------------------------------------

print("\n--- Step 6: Delaunay triangulation + mapper ---")

pixelization_obj = instance.galaxies.source.pixelization

def delaunay_and_mapper():
    interpolator = al.InterpolatorDelaunay(
        mesh=pixelization_obj.mesh,
        mesh_grid=relocated_mesh_grid,
        data_grid=relocated_grid,
    )
    mapper = al.Mapper(
        interpolator=interpolator,
        image_plane_mesh_grid=image_plane_mesh_grid,
        xp=np,
    )
    return mapper

mapper = eager_profile(delaunay_and_mapper, "delaunay_and_mapper")
likelihood_steps.append(("Delaunay + interpolation + mapper", timer.records[-1][1] / N_REPEATS))
print(f"  mapper.pixels (source): {mapper.pixels}")

# ---------------------------------------------------------------------------
# Step 7: Mapping matrix
# ---------------------------------------------------------------------------

print("\n--- Step 7: Mapping matrix ---")

def mapping_matrix_compute():
    return mapper.mapping_matrix

mapping_matrix = eager_profile(mapping_matrix_compute, "mapping_matrix")
likelihood_steps.append(("Mapping matrix", timer.records[-1][1] / N_REPEATS))
print(f"  mapping_matrix shape: {mapping_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 8: Blurred mapping matrix (PSF convolution)
# ---------------------------------------------------------------------------

print("\n--- Step 8: Blurred mapping matrix ---")

def blurred_mapping_matrix_compute():
    return dataset.psf.convolved_mapping_matrix_from(
        mapping_matrix=mapping_matrix,
        mask=dataset.mask,
        xp=np,
    )

blurred_mapping_matrix = eager_profile(
    blurred_mapping_matrix_compute, "blurred_mapping_matrix"
)
likelihood_steps.append(("Blurred mapping matrix (PSF)", timer.records[-1][1] / N_REPEATS))
print(f"  blurred_mapping_matrix shape: {blurred_mapping_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 9: Data vector (D)
# ---------------------------------------------------------------------------

print("\n--- Step 9: Data vector ---")

profile_sub_np = np.asarray(fit.profile_subtracted_image.array)
noise_np = np.asarray(dataset.noise_map.array)

def data_vector_compute():
    return al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=np.asarray(blurred_mapping_matrix),
        image=profile_sub_np,
        noise_map=noise_np,
    )

data_vector = eager_profile(data_vector_compute, "data_vector")
likelihood_steps.append(("Data vector (D)", timer.records[-1][1] / N_REPEATS))
print(f"  data_vector shape: {data_vector.shape}")


# ===================================================================
# PART C — Full FitImaging profiling (this is where the sparse operator kicks in)
# ===================================================================

print("\n" + "=" * 70)
print("FULL FITIMAGING PROFILING (sparse CPU)")
print("=" * 70)


def full_fit_imaging():
    f = al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=al.Settings(use_border_relocator=True),
        xp=np,
    )
    # Touch figure_of_merit to force the inversion through
    _ = f.figure_of_merit
    return f


with timer.section("full_fit_warmup"):
    fit_warm = full_fit_imaging()

log_evidence_warm = fit_warm.figure_of_merit

with timer.section(f"full_fit_steady_x{N_REPEATS}"):
    for _ in range(N_REPEATS):
        fit_steady = full_fit_imaging()
        _ = fit_steady.figure_of_merit

full_fit_per_call = timer.records[-1][1] / N_REPEATS
print(f"    -> per-call avg: {full_fit_per_call:.6f} s")

# Correctness: warmup + steady full fits match reference
np.testing.assert_allclose(
    float(log_evidence_warm),
    float(log_evidence_ref),
    rtol=1e-4,
    err_msg="delaunay_cpu[euclid]: warmup FitImaging log_evidence drifted",
)
np.testing.assert_allclose(
    float(fit_steady.figure_of_merit),
    float(log_evidence_ref),
    rtol=1e-4,
    err_msg="delaunay_cpu[euclid]: steady FitImaging log_evidence drifted",
)
print(f"  Assertion PASSED: FitImaging log_evidence is reproducible across warmup + steady calls")


# ===================================================================
# Summary
# ===================================================================

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

al_version = al.__version__

print("\n" + "=" * 70)
print(f"CPU LIKELIHOOD FUNCTION SUMMARY — EUCLID — v{al_version}")
print("=" * 70)
print(f"  Instrument:            euclid")
print(f"  Pixel scale:           {PIXEL_SCALE} arcsec/pixel")
print(f"  Mask radius:           {MASK_RADIUS} arcsec")
print(f"  Image pixels (masked): {n_image_pixels}")
print(f"  Over-sampled pixels:   {n_over_sampled_pixels}")
print(f"  Delaunay vertices:     {n_source_pixels}")
print(f"  Edge zeroed pixels:    {edge_pixels_total}")
print(f"  Sparse operator setup: {sparse_operator_setup_time:.4f} s (one-off)")
print("-" * 70)

max_label = max(len(label) for label, _ in likelihood_steps)
step_total = 0.0
for i, (label, per_call) in enumerate(likelihood_steps, 1):
    print(f"  {i:>2}. {label:<{max_label}}  {per_call:>12.6f} s")
    step_total += per_call

print("-" * 70)
print(f"      {'TOTAL (steps 1-9 sum)':<{max_label}}  {step_total:>12.6f} s")
print(f"      {'Full FitImaging (per call)':<{max_label}}  {full_fit_per_call:>12.6f} s")
print("=" * 70)

# Cross-reference against the JAX JSON if it exists — surfaces the
# CPU-vs-JAX comparison right in the log without needing a separate tool.
jax_summary_path = (
    _script_dir
    / "results"
    / f"delaunay_likelihood_summary_euclid_v{al_version}.json"
)
jax_full_per_call = None
if jax_summary_path.exists():
    jax_summary = json.loads(jax_summary_path.read_text())
    jax_full_per_call = jax_summary.get("full_pipeline_single_jit")
    if jax_full_per_call is not None:
        ratio = full_fit_per_call / jax_full_per_call
        print(
            f"\n  vs JAX (same config): CPU {full_fit_per_call:.4f}s  /  "
            f"JAX {jax_full_per_call:.4f}s  =  {ratio:.2f}x"
        )
    else:
        print("\n  JAX summary present but no 'full_pipeline_single_jit' field.")
else:
    print(
        f"\n  JAX summary not found at {jax_summary_path.name}; "
        f"run profiling/delaunay.py to generate it."
    )

# --- Save results dictionary ---

likelihood_summary = {
    "autolens_version": al_version,
    "instrument": "euclid",
    "backend": "cpu_sparse_numba",
    "configuration": {
        "pixel_scale_arcsec": PIXEL_SCALE,
        "mask_radius_arcsec": MASK_RADIUS,
        "image_pixels_masked": int(n_image_pixels),
        "over_sampled_pixels": int(n_over_sampled_pixels),
        "delaunay_vertices": int(n_source_pixels),
        "edge_zeroed_pixels": int(edge_pixels_total),
    },
    "sparse_operator_setup_s": sparse_operator_setup_time,
    "steps": {label: per_call for label, per_call in likelihood_steps},
    "total_steps_1_to_9": step_total,
    "full_fit_imaging_per_call": full_fit_per_call,
}

if jax_full_per_call is not None:
    likelihood_summary["jax_full_pipeline_single_jit"] = jax_full_per_call
    likelihood_summary["cpu_over_jax_ratio"] = full_fit_per_call / jax_full_per_call

results_dir = _script_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

dict_path = results_dir / f"delaunay_cpu_likelihood_summary_euclid_v{al_version}.json"
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
    full_fit_per_call,
    color="#C44E52",
    linestyle="--",
    linewidth=1.5,
    label=f"Full FitImaging (per call): {full_fit_per_call:.6f} s",
)
if jax_full_per_call is not None:
    ax.axvline(
        jax_full_per_call,
        color="#55A868",
        linestyle="--",
        linewidth=1.5,
        label=f"JAX full pipeline (per call): {jax_full_per_call:.6f} s",
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Time per call (s)", fontsize=11)
fig.suptitle(
    "Delaunay Imaging Likelihood — CPU (Sparse Operator) — EUCLID",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f"AutoLens v{al_version}  |  {PIXEL_SCALE}\"/px  |  {n_image_pixels} pixels  |  "
    f"{n_over_sampled_pixels} over-sampled  |  {n_source_pixels} Delaunay vertices  |  "
    f"sparse setup: {sparse_operator_setup_time:.2f} s",
    fontsize=9,
)
ax.legend(loc="lower right", fontsize=9)
ax.margins(x=0.15)
fig.tight_layout()

chart_path = results_dir / f"delaunay_cpu_likelihood_summary_euclid_v{al_version}.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")

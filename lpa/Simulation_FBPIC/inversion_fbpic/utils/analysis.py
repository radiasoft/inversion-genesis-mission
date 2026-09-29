"""
analysis.py

Electron beam analysis functions for FBPIC/WarpX output.  Contains functions for loading, cropping, analyzing, and
  visualizing standard ebeam statistics.  The statistics calculated include
- Current profiles
- Emittance (normalized, geometric)
- Twiss parameters (alpha, beta, gamma)
- Energy statistics (mean, RMS, FWHM)
- RMS beam sizes

Typical usage involves loading particle data from an openPMD folder using `load_beam_data()`,  applying
  selection cuts with `apply_cuts()`, and then calling the analysis and plotting functions:  `analyze_beam()`,
  `print_beam_summary()`, and `plot_beam_analysis()`.  See `scripts/ebeam/plot_ebeam_analysis.py` for an example.

All physical units are SI unless otherwise noted.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from openpmd_viewer.addons import LpaDiagnostics


# Methods for loading data and selectively cropping electrons


def load_beam_data(
    folder: Union[str, Path],
    species_name: str = "electrons",
    iteration: int = -1,
    select: Optional[dict[str, tuple[Optional[float], Optional[float]]]] = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    LpaDiagnostics,
]:
    """
    Loads beam data from a diags folder using LpaDiagnostics.

    Args:
        folder (Union[str, Path]): Path to ebeam folder.
        species_name (str, optional): Name of the species to load. Defaults to 'electrons'.
        iteration (int, optional): Which dump to load within "folder".  Defaults to -1 for last dump
        select (dict): Selection criteria for "get_particle" function. Defaults to None

    Returns:
        Tuple: (x, y, z, ux, uy, uz, w, q, ts)
            x, y, z, ux, uy, uz, w, q (np.ndarray): arrays for specified iteration.  These represent the following:
              x - horizontal
              y - vertical
              z - longitudinal
              ux - horiz. momentum
              uy - vert. momentum
              uz - long. momentum
              w - weights
              q - charge of species
            ts (LpaDiagnostics): LpaDiagnostics object.
    """
    ts = LpaDiagnostics(folder, check_all_files=True)
    x, y, z, ux, uy, uz, w, q = ts.get_particle(
        ["x", "y", "z", "ux", "uy", "uz", "w", "charge"],
        iteration=ts.iterations[iteration],
        species=species_name,
        plot=False,
        select=select,
    )
    return x, y, z, ux, uy, uz, w, q, ts


def apply_cut(
    arrays: dict[str, np.ndarray], variable: str, threshold: Optional[float], op: str
) -> dict[str, np.ndarray]:
    """
    Applies a cut to all arrays in the dict based on a condition on one variable.

    Args:
        arrays (dict): Dictionary of arrays (e.g., {'x': x, 'y': y, ...}).
        variable (str): Which variable to cut on (e.g., 'z').
        threshold (float): Value to cut at.  Set to None for no cropping
        op (str): Operator to use:
            'gt' (>)
            'lt' (<)
            'ge' (>=)
            'le' (<=)
            'eq' (==)
            'ne' (!=)

    Returns:
        dict: All arrays in input "arrays" cut according to the selection.

    Raises:
        ValueError: If an unknown operator is provided.
    """
    ops = {
        "gt": operator.gt,
        "lt": operator.lt,
        "ge": operator.ge,
        "le": operator.le,
        "eq": operator.eq,
        "ne": operator.ne,
    }
    if op not in ops:
        raise ValueError(f"Unknown op: {op}")
    if threshold is None:
        return arrays
    arr = np.asarray(arrays[variable])
    selection = ops[op](arr, threshold)
    return {k: np.asarray(v)[selection] for k, v in arrays.items()}


# Main analysis function that calls all sub-analysis routines


def analyze_beam(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    w: np.ndarray,
    q: np.ndarray,
    bins: int = 100,
) -> dict[str, Union[float, dict[str, Union[float, dict[str, float]]]]]:
    """
    Performs comprehensive beam analysis and packages results into a dictionary

    Args:
        x (np.ndarray): Particle x positions in meters.
        y (np.ndarray): Particle y positions in meters.
        z (np.ndarray): Particle z positions in meters.
        ux (np.ndarray): Normalized x-momenta (dimensionless).
        uy (np.ndarray): Normalized y-momenta (dimensionless).
        uz (np.ndarray): Normalized z-momenta (dimensionless).
        w (np.ndarray): Particle weights.
        q (np.ndarray): Particle charges in Coulombs.
        bins (int, optional): Number of bins for current calculation. Defaults to 100.

    Returns:
        dict: All beam parameters (current, Twiss, normalized emittance, energy, sizes, total charge).

    Raises:
        ValueError: If input arrays have different lengths or are empty.
    """
    lens = [len(v) for v in [x, y, z, ux, uy, uz, w]]
    if len(set(lens)) != 1:
        raise ValueError(
            f"Input arrays must have the same length, but got lengths: {lens}"
        )
    if set(lens) == {0}:
        raise ValueError("Input arrays are empty; cannot analyze beam.")

    current, z_axis = calculate_current(z, ux, uy, uz, w, q, bins)
    twiss = calculate_twiss_parameters(x, y, ux, uy, uz, w)
    energy_params = calculate_energy_parameters(ux, uy, uz, w)
    emittance = calculate_geometric_emittance(x, y, ux, uy, uz, w)
    normalized_emittance = {
        "x": emittance["x"] * energy_params["gamma_beam"],
        "y": emittance["y"] * energy_params["gamma_beam"],
    }
    beam_sizes = calculate_beam_size(x, y, z, w)
    total_charge = np.sum(w * q)
    return {
        "current": current,
        "z_axis": z_axis,
        "twiss_parameters": twiss,
        "emittance": normalized_emittance,
        "energy_parameters": energy_params,
        "beam_sizes": beam_sizes,
        "total_charge_c": total_charge,
    }


# Sub-routines for analyzing specific aspects of an electron beam


def calculate_current(
    z: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    w: np.ndarray,
    q: np.ndarray,
    bins: int = 100,
    selection: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the current profile along the z-axis and returns it as a numpy array.

    Args:
        z (np.ndarray): Particle z positions in meters.
        ux (np.ndarray): Normalized x-momenta (dimensionless).
        uy (np.ndarray): Normalized y-momenta (dimensionless).
        uz (np.ndarray): Normalized z-momenta (dimensionless).
        w (np.ndarray): Particle weights.
        q (np.ndarray): Particle charges in Coulombs.
        bins (int, optional): Number of bins for histogram. Defaults to 100.
        selection (np.ndarray, optional): Boolean array to select a subset. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - np.ndarray: Current profile in Amperes.
            - np.ndarray: z-axis bin centers in meters.
    """
    if selection is not None:
        z = z[selection]
        ux = ux[selection]
        uy = uy[selection]
        uz = uz[selection]
        w = w[selection]

    # Calculate Lorentz factor for all particles
    gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)

    # Calculate particle velocities
    vz = uz / gamma * const.c

    # Length to be separated in bins
    len_z = np.max(z) - np.min(z)
    vzq_sum, _ = np.histogram(z, bins=bins, weights=(vz * w * q))

    # Calculate the current in each bin
    current = np.abs(vzq_sum * bins / len_z)
    axis = np.linspace(np.min(z), np.max(z), bins)
    return current, axis


def calculate_slice_energy_spread(
    diag_folder: str,
    species_name: str,
    iteration: int,
    bins: int = 200,
    selection: Optional[dict[str, tuple[Optional[float], Optional[float]]]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate energy spread for each slice in z and normalized weights.

    Calculates the energy spread for each longitudinal slice and the normalized
    weight of each slice defined by its slice charge divided by its slice energy
    spread. This can be used to weight slices with low energy spread more than
    other slices (e.g., for a chirped electron beam used in metrology).

    Args:
        diag_folder: Path to diagnostics folder containing beam data.
        species_name: Name of species to load from beam data.
        iteration: Iteration number to load. This corresponds to the iteration
            value in `ts.iterations[iteration_index]`.
        bins: Number of z slices to create. Defaults to 200.
        selection: Dictionary containing any cropping criteria upon loading
            particles. Keys are variable names, values are tuples of (min, max).
            Defaults to None.

    Returns:
        A tuple containing:
            - energy_spread_per_slice: Array of energy spread in MeV for each
              slice (np.ndarray)
            - z_position_per_slice: Array of z positions (bin centers) for each
              slice in meters (np.ndarray)
            - normalized_per_slice: Array of normalized values calculated as
              charge * mean_energy / energy_spread (np.ndarray)
    """
    ts = LpaDiagnostics(diag_folder, check_all_files=True)
    x, y, z, ux, uy, uz, w, q = ts.get_particle(
        ["x", "y", "z", "ux", "uy", "uz", "w", "charge"],
        iteration=iteration,
        species=species_name,
        plot=False,
        select=selection,
    )

    gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    energy_mev = (gamma - 1) * const.m_e * const.c**2 / const.e / 1e6

    # Create bin edges
    z_bin_edges = np.linspace(np.min(z), np.max(z), bins + 1)

    # Calculate bin centers for z positions
    z_bin_centers = (z_bin_edges[:-1] + z_bin_edges[1:]) / 2

    # Initialize arrays to store results
    energy_spread_per_slice = np.zeros(bins)
    normalized_per_slice = np.zeros(bins)

    charge_beam = np.sum(w * q) * -1e12

    # Loop through each slice
    for i in range(bins):
        # Create mask for particles in this z slice
        # For the last bin, include the upper edge
        if i == bins - 1:
            slice_mask = (z >= z_bin_edges[i]) & (z <= z_bin_edges[i + 1])
        else:
            slice_mask = (z >= z_bin_edges[i]) & (z < z_bin_edges[i + 1])

        # Skip empty slices
        if not np.any(slice_mask):
            continue

        # Get energy values and weights for particles in this slice
        energy_slice = energy_mev[slice_mask]
        w_slice = w[slice_mask]

        # Calculate weighted standard deviation of energy for this slice
        weighted_mean = np.average(energy_slice, weights=w_slice)
        weighted_variance = np.average(
            (energy_slice - weighted_mean) ** 2, weights=w_slice
        )
        energy_spread = np.sqrt(weighted_variance)

        # Calculate charge within slice (convert to pC)
        charge_slice_pc = np.sum(w_slice * q) * -1e12  # Negative because electrons

        # Calculate normalized value: charge / (energy_spread / mean_energy)
        # This is equivalent to: charge * mean_energy / energy_spread
        #  Add 1 to energy spread to avoid division by zero in denominator
        if energy_spread > 0 and np.abs(charge_slice_pc) > 0.1 / bins * np.abs(
            charge_beam
        ):
            normalized = charge_slice_pc * weighted_mean / (1 + energy_spread)
        else:
            normalized = 0.0

        # Store results
        energy_spread_per_slice[i] = energy_spread
        normalized_per_slice[i] = normalized

    # Convert to numpy arrays
    return energy_spread_per_slice, z_bin_centers, normalized_per_slice


def plot_slice_statistics(
    current_axis: np.ndarray,
    current_values: np.ndarray,
    region_axis: np.ndarray,
    region_slice_energy_spread: np.ndarray,
    region_normalized_weights: np.ndarray,
) -> None:
    """Plot slice statistics calculated during chirped beam analysis.

    Creates a multi-axis plot showing the current profile, slice energy spread,
    and normalized weights for slices within the high current region.

    Args:
        current_axis: z-axis for the full data loaded in meters.
        current_values: Current across z slices for the full data loaded in
            Amperes.
        region_axis: z-axis for the slices within the high current region in
            meters.
        region_slice_energy_spread: Energy spreads for slices within the high
            current region in MeV.
        region_normalized_weights: Calculated normalized weights for slices
            within the high current region.
    """
    # Create figure with primary axis for current profile
    fig, ax1 = plt.subplots()

    # Plot current profile on primary axis
    ax1.plot(current_axis, current_values, "b-", label="Current")
    ax1.set_xlabel("z (m)")
    ax1.set_ylabel("Current (A)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    dz = region_axis[1] - region_axis[0]
    ax1.set_xlim(region_axis.min() - 50 * dz, region_axis.max() + 50 * dz)

    # Create secondary axis for energy spread
    ax2 = ax1.twinx()
    ax2.plot(region_axis, region_slice_energy_spread, "r-", label="Energy Spread")
    ax2.set_ylabel("Energy Spread (MeV)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    # Create third axis for normalized values (offset to the right)
    ax3 = ax1.twinx()
    # Offset the third axis spine to the right
    ax3.spines["right"].set_position(("outward", 60))
    ax3.plot(region_axis, region_normalized_weights, "g-", label="Normalized")
    ax3.set_ylabel("Normalized [pC·MeV/MeV]", color="g")
    ax3.tick_params(axis="y", labelcolor="g")

    # Add vertical lines for z_region boundaries
    ax1.vlines(
        [region_axis[0], region_axis[-1]],
        ymin=0,
        ymax=np.max(current_values),
        colors="k",
        ls="--",
        alpha=0.5,
    )

    # Add title and adjust layout
    plt.title("Current Profile, Slice Energy Spread, and Normalized")
    fig.tight_layout()
    plt.show()


def _calculate_moments(
    x: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    w: np.ndarray,
) -> dict[str, float]:
    """
    Calculates the statistical weighted 2nd order moments from the electron beam's 6D phase space

    Args:
        x (np.ndarray): Transverse x positions in meters.
        y (np.ndarray): Transverse y positions in meters.
        ux (np.ndarray): Normalized x-momenta (dimensionless).
        uy (np.ndarray): Normalized y-momenta (dimensionless).
        uz (np.ndarray): Normalized z-momenta (dimensionless).
        w (np.ndarray): Particle weights.

    Returns:
        dict: Dictionary containing weighted centroids and second-order moments
            of the electron beam.
    """
    gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    px_over_p = ux / gamma
    py_over_p = uy / gamma
    x_mean = np.average(x, weights=w)
    y_mean = np.average(y, weights=w)
    px_mean = np.average(px_over_p, weights=w)
    py_mean = np.average(py_over_p, weights=w)
    x2 = np.average((x - x_mean) ** 2, weights=w)
    y2 = np.average((y - y_mean) ** 2, weights=w)
    px2 = np.average((px_over_p - px_mean) ** 2, weights=w)
    py2 = np.average((py_over_p - py_mean) ** 2, weights=w)
    xpx = np.average((x - x_mean) * (px_over_p - px_mean), weights=w)
    ypy = np.average((y - y_mean) * (py_over_p - py_mean), weights=w)
    return {
        "x_mean": float(x_mean),
        "y_mean": float(y_mean),
        "px_mean": float(px_mean),
        "py_mean": float(py_mean),
        "x2": float(x2),
        "y2": float(y2),
        "px2": float(px2),
        "py2": float(py2),
        "xpx": float(xpx),
        "ypy": float(ypy),
    }


def calculate_geometric_emittance(
    x: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    w: np.ndarray,
) -> dict[str, float]:
    """
    Calculates geometric transverse emittance for x and y planes.

    Args:
        x (np.ndarray): Transverse x positions in meters.
        y (np.ndarray): Transverse y positions in meters.
        ux (np.ndarray): Normalized x-momenta (dimensionless).
        uy (np.ndarray): Normalized y-momenta (dimensionless).
        uz (np.ndarray): Normalized z-momenta (dimensionless).
        w (np.ndarray): Particle weights.

    Returns:
        dict: Geometric emittance values for x and y planes in meter-radian.
    """
    moments = _calculate_moments(x, y, ux, uy, uz, w)
    emittance_x = np.sqrt(moments["x2"] * moments["px2"] - moments["xpx"] ** 2)
    emittance_y = np.sqrt(moments["y2"] * moments["py2"] - moments["ypy"] ** 2)
    return {"x": emittance_x, "y": emittance_y}


def calculate_twiss_parameters(
    x: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    w: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    Calculates Twiss parameters (alpha, beta, gamma) for x and y planes.

    Args:
        x (np.ndarray): Transverse x positions in meters.
        y (np.ndarray): Transverse y positions in meters.
        ux (np.ndarray): Normalized x-momenta (dimensionless).
        uy (np.ndarray): Normalized y-momenta (dimensionless).
        uz (np.ndarray): Normalized z-momenta (dimensionless).
        w (np.ndarray): Particle weights.

    Returns:
        dict: Twiss parameters for x and y planes.
    """
    moments = _calculate_moments(x, y, ux, uy, uz, w)
    emittance = calculate_geometric_emittance(x, y, ux, uy, uz, w)
    emittance_x = emittance["x"]
    emittance_y = emittance["y"]
    beta_x = moments["x2"] / emittance_x
    alpha_x = -moments["xpx"] / emittance_x
    gamma_x = moments["px2"] / emittance_x
    beta_y = moments["y2"] / emittance_y
    alpha_y = -moments["ypy"] / emittance_y
    gamma_y = moments["py2"] / emittance_y
    return {
        "x": {"alpha": alpha_x, "beta": beta_x, "gamma": gamma_x},
        "y": {"alpha": alpha_y, "beta": beta_y, "gamma": gamma_y},
    }


def convert_to_mev(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray) -> np.ndarray:
    """
    For a distribution of particles, convert its momentum in x,y,z to energy in units of MeV

    Args:
        ux (np.ndarray): Normalized x-momenta (dimensionless).
        uy (np.ndarray): Normalized y-momenta (dimensionless).
        uz (np.ndarray): Normalized z-momenta (dimensionless).

    Returns:
        np.ndarray: Particle energy in units of MeV
    """
    gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    energy_mev = (gamma - 1) * const.m_e * const.c**2 / const.e / 1e6
    return energy_mev


def calculate_energy_parameters(
    ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, w: np.ndarray
) -> dict[str, float]:
    """
    Calculates Lorentzian gamma, central energy, and energy spread in both RMS and FWHM

    Args:
        ux (np.ndarray): Normalized x-momenta (dimensionless).
        uy (np.ndarray): Normalized y-momenta (dimensionless).
        uz (np.ndarray): Normalized z-momenta (dimensionless).
        w (np.ndarray): Particle weights.

    Returns:
        dict: average Lorentzian gamma, Central energy [MeV], energy RMS [MeV], FWHM [MeV]
    """
    gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    energy_mev = (gamma - 1) * const.m_e * const.c**2 / const.e / 1e6
    central_energy = np.average(energy_mev, weights=w)
    energy_std = np.sqrt(np.average((energy_mev - central_energy) ** 2, weights=w))
    hist, bin_edges = np.histogram(energy_mev, bins=100, weights=w)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peak_idx = np.argmax(hist)
    peak_value = hist[peak_idx]
    half_max = peak_value / 2
    left_idx = peak_idx
    right_idx = peak_idx
    while left_idx > 0 and hist[left_idx] > half_max:
        left_idx -= 1
    while right_idx < len(hist) - 1 and hist[right_idx] > half_max:
        right_idx += 1
    fwhm = bin_centers[right_idx] - bin_centers[left_idx]
    return {
        "gamma_beam": np.average(gamma, weights=w),
        "central_energy_mev": central_energy,
        "energy_std_mev": energy_std,
        "energy_fwhm_mev": fwhm,
    }


def calculate_beam_size(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray
) -> dict[str, float]:
    """
    Calculates RMS beam sizes in x, y, and z directions, and the mean position in x, y, and z

    Args:
        x (np.ndarray): Particle x positions in meters.
        y (np.ndarray): Particle y positions in meters.
        z (np.ndarray): Particle z positions in meters.
        w (np.ndarray): Particle weights.

    Returns:
        dict: RMS beam sizes [m] and means [m].
    """
    x_mean = np.average(x, weights=w)
    y_mean = np.average(y, weights=w)
    z_mean = np.average(z, weights=w)
    sigma_x = np.sqrt(np.average((x - x_mean) ** 2, weights=w))
    sigma_y = np.sqrt(np.average((y - y_mean) ** 2, weights=w))
    sigma_z = np.sqrt(np.average((z - z_mean) ** 2, weights=w))
    return {
        "sigma_x_m": sigma_x,
        "sigma_y_m": sigma_y,
        "sigma_z_m": sigma_z,
        "x_mean_m": x_mean,
        "y_mean_m": y_mean,
        "z_mean_m": z_mean,
    }


# Visualization tools for printing out beam statistics and plotting different phase spaces.


def print_beam_summary(
    analysis_results: dict[
        str, Union[float, dict[str, Union[float, dict[str, float]]]]
    ],
    output_to_file: str | Path | None = None,
) -> None:
    """
    Prints a formatted summary of beam analysis results obtained from the `analyze_beam()` method

    Args:
        analysis_results (dict): Output dictionary from analyze_beam().
        output_to_file (str|Path|None): If provided, save the summary to a file.
    Returns:
        None
    """

    f = None
    if output_to_file is not None:
        output_to_file = Path(output_to_file)
        output_to_file.parent.mkdir(parents=True, exist_ok=True)
        f = open(output_to_file, "w")

    def my_print(x: str = ""):
        if output_to_file is not None:
            f.write(x + "\n")
        else:
            print(x)

    # put everything in try block to ensure file gets closed
    try:
        my_print("=== BEAM ANALYSIS SUMMARY ===")
        my_print(f"Total Charge: {analysis_results['total_charge_c']*1e12:.2f} pC")
        my_print(f"Peak Current: {np.max(analysis_results['current'])*1e-3:.2f} kA")
        my_print()
        my_print("=== ENERGY PARAMETERS ===")
        my_print(
            f"Central Energy: {analysis_results['energy_parameters']['central_energy_mev']:.1f} MeV"
        )
        my_print(
            f"Energy RMS: {analysis_results['energy_parameters']['energy_std_mev']:.1f} MeV"
        )
        my_print(
            f"Energy FWHM: {analysis_results['energy_parameters']['energy_fwhm_mev']:.1f} MeV"
        )
        my_print()
        my_print("=== BEAM SIZES (RMS) ===")
        my_print(f"σx: {analysis_results['beam_sizes']['sigma_x_m']*1e6:.2f} μm")
        my_print(f"σy: {analysis_results['beam_sizes']['sigma_y_m']*1e6:.2f} μm")
        my_print(f"σz: {analysis_results['beam_sizes']['sigma_z_m']*1e6:.2f} μm")
        my_print()
        my_print("=== EMITTANCE ===")
        my_print(f"εnx: {analysis_results['emittance']['x']*1e6:.2f} um⋅rad")
        my_print(f"εny: {analysis_results['emittance']['y']*1e6:.2f} um⋅rad")
        my_print()
        my_print("=== TWISS PARAMETERS ===")
        my_print("X plane:")
        my_print(f"  αx: {analysis_results['twiss_parameters']['x']['alpha']:.3f}")
        my_print(
            f"  βx: {analysis_results['twiss_parameters']['x']['beta']*1e3:.3f} mm"
        )
        my_print(
            f"  γx: {analysis_results['twiss_parameters']['x']['gamma']/1e3:.3f} mm^-1"
        )
        my_print("Y plane:")
        my_print(f"  αy: {analysis_results['twiss_parameters']['y']['alpha']:.3f}")
        my_print(
            f"  βy: {analysis_results['twiss_parameters']['y']['beta']*1e3:.3f} mm"
        )
        my_print(
            f"  γy: {analysis_results['twiss_parameters']['y']['gamma']/1e3:.3f} mm^-1"
        )
        my_print()
        my_print("=== TWISS PARAMETER VALIDATION ===")
        x_twiss = analysis_results["twiss_parameters"]["x"]
        y_twiss = analysis_results["twiss_parameters"]["y"]
        x_invariant = x_twiss["alpha"] ** 2 + x_twiss["beta"] * x_twiss["gamma"]
        y_invariant = y_twiss["alpha"] ** 2 + y_twiss["beta"] * y_twiss["gamma"]
        my_print(f"X plane invariant (α² + βγ): {x_invariant:.6f} (should be 1.0)")
        my_print(f"Y plane invariant (α² + βγ): {y_invariant:.6f} (should be 1.0)")
        if abs(x_invariant - 1.0) > 0.01 or abs(y_invariant - 1.0) > 0.01:
            my_print("Warning: Twiss parameters may be unreliable!")
    except Exception as e:
        print(f"Error in print_beam_summary: {e}")
    finally:
        if output_to_file is not None and f is not None:
            f.close()


def plot_beam_analysis(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    w: np.ndarray,
    analysis_results: dict[
        str, Union[float, dict[str, Union[float, dict[str, float]]]]
    ],
    do_hist: bool = True,
    supertitle: str = "",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    dpi: int = 200,
) -> None:
    """
    Plots standard beam analysis statistics in a single figure.  For use with the output dict of "analyze_beam()"

    Args:
        x (np.ndarray): Particle x positions in meters.
        y (np.ndarray): Particle y positions in meters.
        z (np.ndarray): Particle z positions in meters.
        ux (np.ndarray): Normalized x-momenta (dimensionless).
        uy (np.ndarray): Normalized y-momenta (dimensionless).
        uz (np.ndarray): Normalized z-momenta (dimensionless).
        w (np.ndarray): Particle weights.
        analysis_results (dict): Output dictionary from "analyze_beam()".
        do_hist (bool): If true, 2D plot is histogram. False for scatter plot.
        supertitle (str): Text to be displayed on top of plot array.
        save_path (Optional[Union[str, Path]]): If provided, save the plot to this path.
            The directory will be created if it doesn't exist. Defaults to None (no saving).
        show (bool): Whether to display the plot. Defaults to True.
        dpi (int): Resolution in dots per inch for saved plots. Defaults to 200.

    Returns:
        None
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(
        analysis_results["z_axis"] * 1e6, analysis_results["current"] * 1e-3
    )
    axes[0, 0].set_xlabel("z (μm)")
    axes[0, 0].set_ylabel("Current (kA)")
    axes[0, 0].set_title("Beam Current Profile")
    gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    energy_mev = (gamma - 1) * const.m_e * const.c**2 / const.e / 1e6
    axes[0, 1].hist2d(z * 1e6, energy_mev, bins=300, weights=w, cmap="viridis")
    axes[0, 1].set_xlabel("z (μm)")
    axes[0, 1].set_ylabel("Energy (MeV)")
    axes[0, 1].set_title("Energy vs Z Distribution")

    # Use _calculate_moments for means and variances
    moments = _calculate_moments(x, y, ux, uy, uz, w)
    x_std = np.sqrt(moments["x2"]) * 1e6
    y_std = np.sqrt(moments["y2"]) * 1e6
    gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    px_over_p = ux / gamma
    py_over_p = uy / gamma
    px_std = np.sqrt(moments["px2"])
    py_std = np.sqrt(moments["py2"])

    x_ave = moments["x_mean"] * 1e6
    y_ave = moments["y_mean"] * 1e6
    ux_ave = moments["px_mean"]
    uy_ave = moments["py_mean"]

    x_range = [-2 * x_std + x_ave, 2 * x_std + x_ave]
    px_range = [-2 * px_std + ux_ave, 2 * px_std + ux_ave]
    y_range = [-2 * y_std + y_ave, 2 * y_std + y_ave]
    py_range = [-2 * py_std + uy_ave, 2 * py_std + uy_ave]

    if do_hist:
        axes[1, 0].hist2d(
            x * 1e6,
            px_over_p,
            bins=200,
            weights=w,
            cmap="viridis",
            range=[x_range, px_range],
            # vmax=5e5,
        )
    else:
        axes[1, 0].scatter(x * 1e6, px_over_p, c=w, alpha=0.5, s=1)
    axes[1, 0].set_xlabel("x (μm)")
    axes[1, 0].set_ylabel("ux/γ")
    axes[1, 0].set_title("X Phase Space")
    if do_hist:
        axes[1, 1].hist2d(
            y * 1e6,
            py_over_p,
            bins=200,
            weights=w,
            cmap="viridis",
            range=[y_range, py_range],
            # vmax=5e5,
        )
    else:
        axes[1, 1].scatter(y * 1e6, py_over_p, c=w, alpha=0.5, s=1)
    axes[1, 1].set_xlabel("y (μm)")
    axes[1, 1].set_ylabel("uy/γ")
    axes[1, 1].set_title("Y Phase Space")

    if supertitle:
        plt.suptitle(supertitle)
    plt.tight_layout()

    # Save plot if path is provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    # Display plot if requested
    if show:
        plt.show()
    else:
        plt.close(fig)

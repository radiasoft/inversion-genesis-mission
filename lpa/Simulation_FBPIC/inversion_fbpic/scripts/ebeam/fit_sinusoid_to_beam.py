"""
Script to fit the observed sinusoidal pattern in the ebeam's x-z phase space to calculate the wavelength.
Alternatively, skip the fitting and just plot a given sinusoidal overtop.

Usage:
    Provide the diagnostics folder with ``--diag-folder``.
    To run the fitting algorithm, make sure DO_FITS = True.  Otherwise the script will just plot the sinusoid given by
      the parameters of SINUSOIDAL_PARAMS (the latter is useful if data is messy and you want to compare wavelengths)
    If DO_FITS = True, then it will try to perform fits on all *other* diagnostics folders adjacent to the supplied directory and
      plot the variation.  (If, for instance, you want to plot the wavelength vs number of cells in the simulation)
    Change MIN_Z as needed to crop out unwanted electrons upstream of the main electron bunch of interest
    Run with any python interpreter.
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import pi
from inversion_fbpic.utils.analysis import load_beam_data, apply_cut
import re

MIN_Z: float = 8863e-6  # Minimum z when selecting elections, set to None for no selection
DO_FITS: bool = False  # Set to True to attempt a fit across adjacent diagnostics folders
SINUSOIDAL_PARAMS: tuple[float, float, float, float] = (
    2.5e-6,  # Amplitude
    2 * pi / 4e-6,  # Wavenumber (to change wavelength, remember that k = 2*pi/lambda)
    1.3,  # Phase offset
    0,  # Vertical offset of sinusoid
)  # If not doing a fit, this is the function plotted


def sinusoid(
    z: np.ndarray, A: float, k: float, phi: float, offset: float
) -> np.ndarray:
    """
    Sinusoidal fit function (constant amplitude).

    Args:
        z (np.ndarray): z positions.
        A (float): Amplitude.
        k (float): Wavenumber (1/m).
        phi (float): Phase offset.
        offset (float): Vertical offset.

    Returns:
        np.ndarray: Sinusoidal values at z.
    """
    return A * np.sin(k * z + phi) + offset


def get_wavelength(k: float) -> float:
    """
    Calculate the wavelength from the wavenumber k.

    Args:
        k (float): The wavenumber (in 1/m).

    Returns:
        float: The wavelength (in meters), or np.nan if k is zero.
    """
    if k != 0:
        return 2 * np.pi / np.abs(k)
    else:
        return np.nan


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fit or overlay a sinusoid on an electron beam x-z phase space."
    )
    parser.add_argument(
        "-d",
        "--diag-folder",
        type=Path,
        required=True,
        help="Simulation directory containing the ebeam diagnostics folder.",
    )
    return parser.parse_args()


def main(diag_folder: Path) -> None:
    """
    Script entry point for fitting or plotting a sinusoidal pattern in the ebeam's x-z phase space.

    Returns:
        None
    """
    base_dir = diag_folder

    # Prepare for side-by-side plots
    if DO_FITS:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, axis = plt.subplots(1, 1, figsize=(7, 6))
        axes = [axis]

    # --- PART 1: 2D histogram and fit for data within the specified diag folder ---
    ebeam_dir = base_dir / "ebeam"
    h5_files = sorted(ebeam_dir.glob("*.h5"))
    if h5_files:
        # Use the latest file
        x, y, z, ux, uy, uz, w, q, ts = load_beam_data(ebeam_dir, "electrons")
        arrs = {"x": x, "y": y, "z": z, "ux": ux, "uy": uy, "uz": uz, "w": w}
        arrs = apply_cut(arrs, "z", MIN_Z, op="gt")
        x, z = arrs["x"], arrs["z"]
        # 2D histogram
        hist = axes[0].hist2d(z * 1e6, x * 1e6, bins=200, cmap="viridis")
        axes[0].set_xlabel("z (μm)")
        axes[0].set_ylabel("x (μm)")
        axes[0].set_title("2D Histogram of x vs z")
        plt.colorbar(hist[3], ax=axes[0], label="Counts")

        # Bin z and get mean x for each z bin for fitting
        bins = np.linspace(z.min(), z.max(), 100)
        digitized = np.digitize(z, bins)
        z_bin_centers = 0.5 * (bins[:-1] + bins[1:])

        if DO_FITS:
            x_means = np.array(
                [
                    x[digitized == i].mean() if np.any(digitized == i) else np.nan
                    for i in range(1, len(bins))
                ]
            )
            # Remove NaNs for fitting
            valid = ~np.isnan(x_means)
            z_fit = z_bin_centers[valid]
            x_fit = x_means[valid]
            # Initial guess: amplitude, k, phase, offset
            A0 = (np.nanmax(x_fit) - np.nanmin(x_fit)) / 2
            k0 = 2 * np.pi / 0.8e-6  # 800 nm wavelength
            phi0 = 0
            offset0 = np.nanmean(x_fit)
            try:
                popt, _ = curve_fit(sinusoid, z_fit, x_fit, p0=[A0, k0, phi0, offset0])
                z_fit_dense = np.linspace(z_fit.min(), z_fit.max(), len(z_fit) * 10)
                x_fit_curve = sinusoid(z_fit_dense, *popt)
                axes[0].plot(
                    z_fit_dense * 1e6,
                    x_fit_curve * 1e6,
                    color="r",
                    lw=2,
                    label=f"Sinusoidal fit ({2 * pi / popt[1] * 1e6:.2f} um)",
                )
                axes[0].legend()
            except Exception as e:
                print(f"Fit failed: {e}")
        else:
            z_fit_dense = np.linspace(
                z_bin_centers.min(), z_bin_centers.max(), len(z_bin_centers) * 10
            )
            x_fit_curve = sinusoid(z_fit_dense, *SINUSOIDAL_PARAMS)
            axes[0].plot(
                z_fit_dense * 1e6,
                x_fit_curve * 1e6,
                color="r",
                lw=2,
                label=f"Sinusoidal Func ({2 * pi / SINUSOIDAL_PARAMS[1] * 1e6:.2f} um)",
            )
            axes[0].legend()

    else:
        print(f"No ebeam h5 files found in {base_dir}")

    # --- PART 2: Wavelength vs N for all *_diags folders ---
    if DO_FITS:
        N_folders = sorted(
            [
                d
                for d in base_dir.parent.iterdir()
                if d.is_dir() and re.match(r"^\d+_diags$", d.name)
            ]
        )
        wavelengths: list[float] = []
        N_values: list[float] = []

        for N_folder in N_folders:
            ebeam_dir = N_folder / "ebeam"
            h5_files = sorted(ebeam_dir.glob("*.h5"))
            if not h5_files:
                continue
            x, y, z, ux, uy, uz, w, q, ts = load_beam_data(ebeam_dir, "electrons")
            arrs = {"x": x, "y": y, "z": z, "ux": ux, "uy": uy, "uz": uz, "w": w}
            arrs = apply_cut(arrs, "z", 220e-6, op="gt")
            x, z = arrs["x"], arrs["z"]
            bins = np.linspace(z.min(), z.max(), 100)
            digitized = np.digitize(z, bins)
            z_bin_centers = 0.5 * (bins[:-1] + bins[1:])
            x_means = np.array(
                [
                    x[digitized == i].mean() if np.any(digitized == i) else np.nan
                    for i in range(1, len(bins))
                ]
            )
            valid = ~np.isnan(x_means)
            z_fit = z_bin_centers[valid]
            x_fit = x_means[valid]
            if len(z_fit) > 5:
                try:
                    A0 = (np.nanmax(x_fit) - np.nanmin(x_fit)) / 2
                    k0 = 2 * np.pi / 0.8e-6  # 800 nm wavelength
                    phi0 = 0
                    offset0 = np.nanmean(x_fit)
                    popt, _ = curve_fit(
                        sinusoid, z_fit, x_fit, p0=[A0, k0, phi0, offset0]
                    )
                    wavelength = get_wavelength(popt[1])
                except Exception as e:
                    print(f"Fit failed for {N_folder.name}: {e}")
                    wavelength = np.nan
            else:
                wavelength = np.nan
            try:
                N = int(N_folder.name.split("_")[0])
            except Exception:
                N = np.nan
            N_values.append(N)
            wavelengths.append(
                wavelength * 1e6 if wavelength is not np.nan else np.nan
            )  # Convert to microns
        N_values = np.array(N_values)
        wavelengths = np.array(wavelengths)
        sort_idx = np.argsort(N_values)
        N_values = N_values[sort_idx]
        wavelengths = wavelengths[sort_idx]
        axes[1].plot(N_values, wavelengths, marker="o")
        axes[1].set_xlabel("N")
        axes[1].set_ylabel("Fitted Wavelength (μm)")
        axes[1].set_title("Fitted Sinusoidal Wavelength vs N")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(parse_args().diag_folder)

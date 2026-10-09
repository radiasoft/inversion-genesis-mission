#!/usr/bin/env python3
"""
Generate PNG images from .npy files for creating rho_electron movies.


"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm, LogNorm
import re
from typing import Literal
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import e, epsilon_0, mu_0

try:
    from inversion_fbpic.utils.field_analysis import cosine_squared_band
except ImportError:
    from field_analysis import cosine_squared_band  # type: ignore


def _get_label_and_scale_from_field_name(
    field_name: str | None,
    component: Literal["x", "y", "z", "abs", "sqr", "eme"] | None = None,
) -> tuple[str, float]:
    """
    Get the label and scale from the field name.
    """
    if component == "eme":
        label = r"$\mathcal{E}$ (J/$\mu$m$^3$)"
        field_scale = 1e-18
        return label, field_scale

    if "rho" in field_name:
        label = r"$\rho$ (cm$^{-3}$)"
        field_scale = -1e-6 / e
    else:
        component_sub = f"_{{{component}}}" if component is not None else ""
        label = f"${field_name}{component_sub}$"
        field_scale = 1.0

    return label, field_scale


def _extract_int(f: Path, extension: str = "npy", prefix: str = r".*_") -> int:
    """
    Extract integer from filename using regex. Returns -1 if no integer is found.
    """
    match = re.search(rf"{prefix}(\d+)\.{extension}$", f.name)
    return int(match.group(1)) if match else -1


def em_bandpass_filter(
    exyz_bxyz: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
    bbox: tuple[float, float, float, float],
    num_bands: int = 6,
) -> list[np.ndarray]:
    """
    Apply a bandpass filter to the EM field and get the EM energy density.
    The bands are overlapping cosine-squared bands such that their sum results in the original field (ignoring cross-band terms).
    They are distributed in wavenumber space from 0 to the maximum wavenumber in the field.

    Args:
        exyz_bxyz: Tuple of 6 numpy arrays containing the x, y, z components of the E and B fields.
        bbox: Tuple containing the minimum and maximum values of the z, r coordinates.
        num_bands: Number of bands to use for the bandpass filter.
            If 1, no bandpass filter is applied.
            If > 1, the bands are overlapping cosine-squared bands such that their sum results in the original field.
            They are distributed in wavenumber space from 0 to the maximum wavenumber in the field.

    Returns:
        List of numpy arrays containing the filtered EM energy densities.
    """
    filtered_energies = []
    if num_bands > 1:
        kx, ky = np.meshgrid(
            np.fft.fftfreq(exyz_bxyz[0].shape[1]), np.fft.fftfreq(exyz_bxyz[0].shape[0])
        )
        kx = 2 * np.pi * exyz_bxyz[0].shape[1] / (bbox[1] - bbox[0]) * kx
        ky = 2 * np.pi * exyz_bxyz[0].shape[0] / (bbox[3] - bbox[2]) * ky
        k2 = np.sqrt(kx**2 + ky**2)

        eb_f = [np.fft.fft2(field) for field in exyz_bxyz]

        band_k0s = np.linspace(0, np.max(k2), num_bands)
        band_kw = band_k0s[1] - band_k0s[0]
        for i in range(num_bands):
            k0 = band_k0s[i]
            win = cosine_squared_band(k2, k0, band_kw)

            eb_b = [np.fft.ifft2(field * win).real ** 2 for field in eb_f]

            energy_density = 0.5 * epsilon_0 * sum(eb_b[0:3]) + 0.5 / mu_0 * sum(
                eb_b[3:6]
            )
            filtered_energies.append(energy_density)
    elif num_bands == 1:
        eb2 = [field**2 for field in exyz_bxyz]
        filtered_energies.append(
            0.5 * epsilon_0 * sum(eb2[0:3]) + 0.5 / mu_0 * sum(eb2[3:6])
        )
    else:
        raise ValueError(f"Invalid number of bands: {num_bands}")

    return filtered_energies


def plot_from_npy(
    data_path: Path | str,
    save_path: Path | str,
    field_name: str,
    component: Literal["x", "y", "z", "abs", "sqr"] | None = None,
    vminmax: tuple[float, float] | Literal["even"] | None = None,
    scale: Literal["linear", "log"] = "linear",
    cmap: str = "bwr",
    idx: int | None = None,
    rmax: float | None = None,
    font_size: int = 16,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """
    Plot a PNG from a .npy file containing a 2D array with axis labels. This is intended to be used after calling `extract_hdf5_field.process_single_simulation`.

    Args:
        data_path (Path | str): Path to the .npy file or directory containing .npy files. This directory is returned by `extract_hdf5_field.process_single_simulation`. This directory (or the parent directory of a single .npy file) will contain the resulting images.
        save_path (Path | str): Path to save the plot to. This directory will contain the resulting images.
        field_name (str): Name of the field to plot (e.g., "rho", "E", "B")
        component (Literal["x", "y", "z", "abs", "sqr"] | None): Component of the field to plot. Defaults to None (scalar field).
        vminmax (tuple[float, float] | Literal["even"] | None): The minimum and maximum value of the color scale. If "even", the color scale is symmetric around 0. If either entry is None, the minimum and maximum value of the field are used for the None entry. Defaults to None.
        scale (Literal["linear", "log"]): Scale of the color scale. Defaults to "linear".
        cmap (str): Colormap to use. Defaults to "bwr".
        idx (int|None): Index for output filename. For a single .npy file, if None,
            an index is inferred from the filename suffix `_<int>.npy` and falls back
            to -1 when no suffix is present.
        rmax (float|None): Maximum radial extent to display (in meters). If None, no cropping is performed. Defaults to None.
        font_size (int): Font size for the plot. Defaults to 16.
        vmin (float): Lower limit on color scale (must be < 0 for SymLogNorm). Defaults to None. Note: This is only for backwards compatibility. Ignored if vminmax is not None.
        vmax (float): Upper limit on color scale (must be > 0 for SymLogNorm). Defaults to None. Note: This is only for backwards compatibility. Ignored if vminmax is not None.

    Returns:
        Path | None: Path to the directory containing the PNG images. If no .npy files are found, returns None.
    """
    data_path = Path(data_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if vminmax is None and (vmin is not None or vmax is not None):
        vminmax = (vmin, vmax)

    component_suffix = "_" + component if component is not None else ""
    full_field_name = f"{field_name}{component_suffix}"

    # if a directory is provided, process all .npy files in the directory
    if data_path.is_dir():
        npy_files = list(data_path.glob(f"{full_field_name}*.npy"))
        npy_files = [
            f for f in npy_files if re.match(rf"^{full_field_name}_\d+\.npy", f.name)
        ]
        if not npy_files:
            print(f"No .npy files found in {data_path} for field {full_field_name}")
            return None

        # sort by index if all indices are unique; if not, indexing will be out of order (acceptable)
        indices = [_extract_int(f) for f in npy_files]
        if len(indices) == len(set(indices)):
            npy_files.sort(key=_extract_int)
            indices.sort()

        for i, npy_file in zip(indices, npy_files):
            plot_from_npy(
                data_path=npy_file,
                save_path=save_path,
                field_name=field_name,
                component=component,
                vminmax=vminmax,
                scale=scale,
                cmap=cmap,
                idx=i,
                rmax=rmax,
                font_size=font_size,
            )
        return save_path

    if idx is None:
        idx = _extract_int(data_path)

    # if a single .npy file is provided, process it
    arr = np.load(data_path)
    # Extract axes
    x = arr[-1, 1:]
    y = arr[:-1, 0]
    field = arr[:-1, 1:]

    label, field_scale = _get_label_and_scale_from_field_name(field_name, component)

    ax = plot_field(
        field,
        (x[0], x[-1], y[0], y[-1]),
        (None, None, rmax),
        vminmax=vminmax,
        scale=scale,
        quantity_label=label,
        field_scale=field_scale,
        cmap=cmap,
        font_size=font_size,
        save_path=save_path / f"{full_field_name}_{idx:06d}.png",
    )
    plt.close(ax.figure)

    return save_path


def plot_from_hdf5_series(
    series_path: Path | str,
    save_path: Path | str,
    field_name: str | None = None,
    component: Literal["x", "y", "z", "abs", "sqr", "eme"] | None = None,
    vminmax: tuple[float, float] | Literal["even"] | None = None,
    scale: Literal["linear", "log"] = "linear",
    field_offset: float = 0.0,
    cmap: str = "bwr",
    rmax: float | None = None,
    font_size: int = 16,
) -> tuple[Path, str]:
    """
    Plot a series of PNG images from an OpenPMD time series.
    Args:
        series_path (Path | str): Path to the OpenPMD time series.
        save_path (Path | str): Path to save the plot to. This directory will contain the resulting images.
        field_name (str | None): Name of the field to plot (e.g., "rho", "E", "B"). If component is "eme", this is ignored and "E" and "B" are utilized.
        component (Literal["x", "y", "z", "abs", "sqr", "eme"] | None): Component of the field to plot. Defaults to None (scalar field).
        vminmax (tuple[float, float] | Literal["even"] | None): The minimum and maximum value of the color scale. If "even", the color scale is symmetric around 0. If None, the minimum and maximum value of the field are used. Defaults to None.
        scale (Literal["linear", "log"]): Scale of the color scale. Defaults to "linear".
        field_offset (float): Offset to add to the field. Defaults to 0.0.
        cmap (str): Colormap to use. Defaults to "bwr".
        rmax (float | None): Maximum radial extent to display (in meters). If None, no cropping is performed. Defaults to None.
        font_size (int): Font size for the plot. Defaults to 16.
    Returns:
        Path: Path to the directory containing the PNG images.
        str: Prefix for the file names.
    """
    if field_name is None and component != "eme":
        raise ValueError("field_name must be provided if component is not 'eme'")

    series_path = Path(series_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    file_prefix = ""
    if field_name is not None and component not in ["eme", "abs", "sqr"]:
        component_suffix = "_" + component if component is not None else ""
        file_prefix = f"{field_name}{component_suffix}"
    elif component == "eme":
        file_prefix = "eme"
    else:
        raise ValueError(f"Invalid field, component pair: {field_name}, {component}")

    # OpenPMD: try the given path first, then a common `hdf5/` layout next to diagnostics.
    try:
        series = OpenPMDTimeSeries(series_path)
    except FileNotFoundError:
        try:
            series_path = series_path / "hdf5"
            series = OpenPMDTimeSeries(series_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No HDF5 files found in `{series_path.parent}` or `{series_path}`"
            )

    # Charge density: show n_e in cm^-3 (sign flip + scale from C/m^3).
    # Otherwise, keep as-is
    label, field_scale = _get_label_and_scale_from_field_name(field_name, component)

    its = series.iterations
    if component in ["x", "y", "z", None]:
        for i in its:
            field, info = series.get_field(field_name, component, iteration=i)

            try:
                time = info.time
            except AttributeError:
                time = None
            ax = plot_field(
                field,
                (info.zmin, info.zmax, info.rmin, info.rmax),
                (None, None, rmax),
                vminmax=vminmax,
                scale=scale,
                time=time,
                quantity_label=label,
                field_scale=field_scale,
                field_offset=field_offset,
                cmap=cmap,
                font_size=font_size,
                save_path=save_path / f"{file_prefix}_{i:06d}.png",
            )
            plt.close(ax.figure)
    elif component == "eme":
        for i in its:
            ex, info = series.get_field("E", "x", iteration=i)
            ey, info = series.get_field("E", "y", iteration=i)
            ez, info = series.get_field("E", "z", iteration=i)
            bx, info = series.get_field("B", "x", iteration=i)
            by, info = series.get_field("B", "y", iteration=i)
            bz, info = series.get_field("B", "z", iteration=i)
            emes = em_bandpass_filter(
                (ex, ey, ez, bx, by, bz), (info.zmin, info.zmax, info.rmin, info.rmax)
            )

            try:
                time = info.time
            except AttributeError:
                time = None

            fig, axs = plt.subplots(2, 3, figsize=(16, 8))
            for j, (ax, eme) in enumerate(zip(axs.flat, emes)):
                plot_field(
                    eme,
                    (info.zmin, info.zmax, -info.rmax, info.rmax),
                    (None, None, rmax),
                    vminmax=vminmax,
                    scale=scale,
                    time=time,
                    quantity_label=label,
                    field_scale=field_scale,
                    field_offset=field_offset,
                    cmap=cmap,
                    font_size=font_size,
                    ax=ax,
                    title_override=f"Band {j}",
                )
            fig.suptitle(
                _construct_title(None, label, time, info.zmin), fontsize=font_size
            )
            plt.tight_layout()
            fig.savefig(save_path / f"{file_prefix}_{i:06d}.png")
            plt.close(fig)

    else:
        # TODO: implement absolute or squared magnitude of a vector field
        raise NotImplementedError(
            "Absolute or squared magnitude of a field is not yet implemented"
        )

    return save_path, file_prefix


def _construct_title(
    title_prefix: str | None,
    quantity_label: str | None,
    time: float | None,
    d_zmin: float,
) -> str:
    title_str = ""

    def sep() -> str:
        return " | " if title_str else ""

    if title_prefix is not None:
        title_str += f"{title_prefix}"
    if quantity_label is not None:
        title_str += f"{sep()}{quantity_label}"
    if time is not None:
        title_str += f"{sep()}$t$ = {time*1e12:.2f} ps"
    title_str += f"{sep()}$z_0$ = {d_zmin*1e3:.2f} mm"
    return title_str


def plot_field(
    field: np.ndarray,
    data_grid_limits: tuple[float, float, float, float],
    display_grid_limits: tuple[float, float, float] | None = None,
    vminmax: tuple[float, float] | Literal["even"] | None = None,
    scale: Literal["linear", "log"] = "linear",
    time: float | None = None,
    title_prefix: str | None = None,
    title_override: str | None = None,
    quantity_label: str | None = None,
    ax: plt.Axes | None = None,
    field_scale: float = 1.0,
    field_offset: float = 0.0,
    cmap: str = "bwr",
    font_size: int = 16,
    save_path: Path | None = None,
) -> plt.Axes:
    """
    Plot a scalar field (or an element of a vector field).
    Supports linear and logarithmic color scales.

    Args:
        field (np.ndarray): The field to plot.
        data_grid_limits (tuple[float, float, float, float]): The limits of the data grid (zmin, zmax, rmin, rmax) in meters. From an OpenPMDTimeSeries, these are contained in the info object returned by `get_field`.
        display_grid_limits (tuple[float, float, float] | None): The limits of the display grid (zmin, zmax, rmax), in the moving frame, in micrometers. If an entry is None, the corresponding axis is not cropped. Defaults to None.
        vminmax (tuple[float, float] | Literal["even"] | None): The minimum and maximum value of the color scale. If "even", the color scale is symmetric around 0. If either entry is None, the minimum and maximum value of the field are used for the None entry. Defaults to None.
        scale (Literal["linear", "log"]): The scale of the color scale. Defaults to "linear".
            If "linear", the color scale is linear.
            If "log", the color scale is:
            - If vmax > vmin > 0 or vmin < vmax < 0, the color scale is logarithmic in the absolute value of the field.
            - If vmax > 0 > vmin, the color scale is logarithmic in the absolute value of the field, except for the region within [+/- (abs(vmax) + abs(vmin)) / 1e5], where it is linear.
        time (float | None): The time of the frame to plot in seconds. This is converted to picoseconds for the title. If None, no time is displayed. Defaults to None.
        title_prefix (str | None): A prefix to add to the title. Defaults to None.
        title_override (str | None): An override for the title. If None, the title is constructed from the title_prefix, quantity_label, time, and data_grid_limits. Defaults to None.
        quantity_label (str | None): The label of the quantity to plot (for the colorbar and the title). Defaults to None.
        ax (plt.Axes | None): The axes to plot on. If None, a new figure is created. Defaults to None.
        field_scale (float): The scalar factor to apply to the field. Defaults to 1.0.
        field_offset (float): The offset to add to the field. Defaults to 0.0.
        cmap (str): The colormap to use. Defaults to "bwr".
        font_size (int): The font size for the plot. Defaults to 16.
        save_path (Path | None): The path to save the plot to. If None, the plot is not saved. Defaults to None.

    Returns:
        plt.Axes: The axes object. If `ax` is provided, the same axes object is returned.
    """
    (d_zmin, d_zmax, d_rmin, d_rmax) = data_grid_limits
    if display_grid_limits is not None:
        (g_zmin, g_zmax, g_rmax) = display_grid_limits
    else:
        g_zmin = None
        g_zmax = None
        g_rmax = None
    field = field * field_scale + field_offset

    # Per-frame color limits unless the caller pins vminmax.
    if not isinstance(vminmax, tuple):
        _vmin = np.min(field)
        _vmax = np.max(field)
        if vminmax == "even":
            _vmax = np.max([abs(_vmin), abs(_vmax)])
            _vmin = -_vmax
    else:
        _vmin = vminmax[0] if vminmax[0] is not None else np.min(field)
        _vmax = vminmax[1] if vminmax[1] is not None else np.max(field)

    # ensure order
    if _vmin > _vmax:
        _vmin, _vmax = _vmax, _vmin

    if scale == "linear":
        norm = Normalize(vmin=_vmin, vmax=_vmax, clip=True)
    elif scale == "log" and _vmin > 0:
        norm = LogNorm(vmin=_vmin, vmax=_vmax, clip=True)
    elif scale == "log" and _vmax < 0:
        norm = LogNorm(vmin=_vmax, vmax=_vmin, clip=True)
    elif scale == "log" and _vmin < 0 and _vmax > 0:
        norm = SymLogNorm(
            vmin=_vmin,
            vmax=_vmax,
            base=10,
            linthresh=(abs(_vmin) + abs(_vmax)) / 1e5,
            clip=True,
        )
    else:
        raise ValueError(f"Invalid scale: {scale}")

    with plt.rc_context({"font.size": font_size}):
        if ax is None:
            _, ax = plt.subplots(figsize=(16, 8))
        im = ax.imshow(
            field,
            aspect="auto",
            origin="lower",
            norm=norm,
            cmap=cmap,
            extent=[
                0.0,
                (d_zmax - d_zmin) * 1e6,
                d_rmin * 1e6,
                d_rmax * 1e6,
            ],
        )
        ax.set_ylim(
            bottom=-g_rmax * 1e6 if g_rmax is not None else None,
            top=g_rmax * 1e6 if g_rmax is not None else None,
        )
        ax.set_xlim(
            left=g_zmin * 1e6 if g_zmin is not None else None,
            right=g_zmax * 1e6 if g_zmax is not None else None,
        )
        ax.figure.colorbar(im, ax=ax, label=quantity_label)
        ax.set_xlabel("$\\zeta$ ($\\mu$m)")
        ax.set_ylabel("$r$ ($\\mu$m)")

        if title_override is not None:
            title_str = title_override
        else:
            title_str = _construct_title(title_prefix, quantity_label, time, d_zmin)
        ax.set_title(title_str)
        ax.figure.tight_layout()
        if save_path is not None:
            ax.figure.savefig(save_path)

    return ax

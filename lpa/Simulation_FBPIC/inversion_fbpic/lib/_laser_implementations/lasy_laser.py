"""LASY laser pulse: a thin ``SerializableConfig`` wrapper around ``HighOrderLasyLaser``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal, TYPE_CHECKING

import attrs

from inversion_fbpic.lib.laser import _gaussian_r_extent, _LaserPulse

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from fbpic.lpa_utils.laser.laser_profiles import LaserProfile


def _zernike_names() -> tuple[str, ...]:
    """Zernike names accepted by ``HighOrderLasyLaser`` (lazy: utils.laser imports lasy)."""
    from inversion_fbpic.utils.laser import ZERNIKE_OSA_INDICES

    return tuple(ZERNIKE_OSA_INDICES)


def _normalize_zernike_coefficients(value: Any) -> dict[str, float]:
    """Return all Zernike names in canonical order, missing ones as 0.0."""
    names = _zernike_names()
    value = {} if value is None else value
    if not isinstance(value, Mapping):
        raise ValueError("zernike_coefficients must be a mapping of name -> amplitude.")
    if unknown := set(value) - set(names):
        raise ValueError(
            f"Unknown zernike_coefficients keys: {sorted(unknown)}. Allowed: {list(names)}"
        )
    return {name: float(value.get(name, 0.0)) for name in names}


def _pair_of(cast: Any) -> Any:
    """Converter turning any two-element sequence into a tuple of ``cast`` values."""
    return lambda value: tuple(cast(item) for item in value)


def _has_two_entries(instance: Any, attribute: Any, value: tuple) -> None:
    if len(value) != 2:
        raise ValueError(
            f"{attribute.name} must have exactly two entries, got {len(value)}."
        )


def _optional_bandwidth(value: Any) -> float | str | None:
    """``None`` (use the utils default), ``"auto"`` (transform-limited), or a float."""
    if value is None or value == "auto":
        return value
    return float(value)


def _resolve_comm(comm: Any | None) -> tuple[int, Any | None]:
    """Return ``(rank, mpi_comm)`` for the communicator forms ``prepare`` accepts.

    ``comm`` may be ``None`` (serial, rank 0), FBPIC's ``BoundaryCommunicator``
    (exposes ``rank`` and ``mpi_comm``, the latter ``None`` without MPI), or an
    mpi4py communicator such as ``MPI.COMM_WORLD`` (exposes ``Get_rank``).
    """
    if comm is None:
        return 0, None
    if hasattr(comm, "mpi_comm"):
        return int(comm.rank), comm.mpi_comm
    if hasattr(comm, "Get_rank"):
        return int(comm.Get_rank()), comm
    raise TypeError(
        "comm must be None, FBPIC's sim.comm, or an mpi4py communicator; "
        f"got {type(comm).__name__}."
    )


@attrs.define(kw_only=True, slots=False, frozen=True)
class LasyLaserPulse(_LaserPulse):
    """
    Super-Gaussian laser pulse with Zernike and spectral-phase aberrations, built with LASY.

    Thin wrapper around ``inversion_fbpic.utils.laser.HighOrderLasyLaser``: every
    field below maps onto one of its physical parameters or hyperparameters and
    carries the same default. The pulse is constructed at focus, back-propagated
    to the simulation start plane, optionally re-centred, normalized to
    ``energy``, and written to a LASY HDF5 file that FBPIC reads through
    ``FromLasyFileLaser``. The build is expensive and happens once, in
    ``prepare()``, on MPI rank 0 only; the other ranks block in a broadcast and
    receive either the file path or rank 0's error, so a bad configuration
    fails on every rank instead of hanging. Physical validation (grid/mode consistency, spectral-phase
    requirements, pulse-duration limits) is performed by ``HighOrderLasyLaser``
    at build time.

    Only ``energy`` may be provided. ``a0`` is measured numerically from the
    field at focus during ``prepare()`` and is reported as ``out_a0``
    (``null`` in YAML written before the build).

    LASY pulses can only be emitted through an antenna, so ``method`` is fixed
    to ``"antenna"``, ``v_antenna`` to ``0.0``, and ``z0_antenna`` is required.
    FBPIC resets the LASY time axis to zero, so the peak intensity leaves the
    antenna at ``t_start`` plus the peak's delay from the start of the LASY time
    window: ``3 * tau_fwhm`` for the default transform-limited pulse, or exactly
    ``peak_delay_from_file_start`` when that is set. ``z0`` is informational
    only (nominal centroid at ``t = 0``, used for plotting extents); for a
    consistent picture set ``z0 = z0_antenna - c * (t_start + peak_delay)``.

    Args:
        energy: (float) [J] Energy of the laser pulse in Joules. Required: the LASY field is normalized to this energy and `a0` is measured from it.
        a0: (None) Not accepted. The normalized amplitude is measured numerically at focus during `prepare()` and reported as `out_a0` (`null` before the build).
        z0: (float) [m] Nominal centroid position at t = 0 in meters, used only for plotting extents. Emission timing is set by `z0_antenna`, `t_start`, and `peak_delay_from_file_start`; for a consistent picture use `z0 = z0_antenna - c * (t_start + peak_delay)`.
        method: (Literal["antenna"]|None) |OPTIONAL| Fixed to "antenna": LASY files can only be emitted by FBPIC's laser antenna. None means "antenna"; "direct" is rejected.
        z0_antenna: (float) [m] Position of the stationary antenna that emits the pulse, in meters. Required.
        v_antenna: (float|None) [m/s] |OPTIONAL| Must be 0 or None (stored as 0.0): a LASY pulse cannot be emitted from a moving antenna.
        wavelength: (float) [m] Central wavelength of the laser pulse in meters.
        tau_fwhm: (float) [s] Full-width at half-maximum intensity duration of the transform-limited pulse in seconds.
        waist: (float) [m] Super-Gaussian spot size (1/e^2 radius for order 2) at focus in meters.
        focal_position: (float) [m] Focal position of the laser pulse in meters, relative to the simulation start plane.
        super_gaussian_order: (float) Super-Gaussian order of the transverse profile. 2.0 is Gaussian.
        zernike_coefficients: (dict[str, float]) [rad] |OPTIONAL| Zernike phase amplitudes at focus keyed by name (astigmatism_2, astigmatism_4, coma_y, coma_x, trefoil_y, trefoil_x, spherical_3, astigmatism_6, coma_5_y, coma_5_x, secondary_trefoil_y, secondary_trefoil_x). Missing names default to 0.0; unknown names are rejected.
        spectral_bandwidth: (float|Literal["auto"]|None) [rad/s] |OPTIONAL| Gaussian spectral-intensity FWHM in angular frequency. `"auto"` derives the transform-limited value from `tau_fwhm`; None (default) keeps `HighOrderLasyLaser`'s default of 0, i.e. an analytic Gaussian envelope without spectral phase.
        cep: (float|None) [rad] |OPTIONAL| Carrier-envelope phase. None (default) keeps `HighOrderLasyLaser`'s default of 0.
        gdd: (float|None) [s^2] |OPTIONAL| Group-delay dispersion about the central frequency. Requires a nonzero `spectral_bandwidth`. None (default) applies none; mutually exclusive with `gdd_relative`.
        tod: (float|None) [s^3] |OPTIONAL| Third-order spectral phase. Requires a nonzero `spectral_bandwidth`. None (default) applies none; mutually exclusive with `tod_relative`.
        fod: (float|None) [s^4] |OPTIONAL| Fourth-order spectral phase. Requires a nonzero `spectral_bandwidth`. None (default) applies none.
        gdd_relative: (float|None) |OPTIONAL| Duration-normalized GDD in [-1, 1]; +1 corresponds to 5e-28 s^2 at 30 fs FWHM and scales with duration squared. None (default) applies none.
        tod_relative: (float|None) |OPTIONAL| Duration-normalized TOD in [-1, 1]; +1 corresponds to 1e-41 s^3 at 30 fs FWHM and scales with duration cubed. None (default) applies none.
        polarization: (tuple[float, float]) |OPTIONAL| Real Jones vector (Ex, Ey) passed to LASY. Defaults to (1, 0), linear along x.
        n_azimuthal_modes: (int) |OPTIONAL| Number of azimuthal modes in the LASY r-t grid. Defaults to 5.
        num_points: (tuple[int, int]) |OPTIONAL| LASY grid points (radial, temporal). Defaults to (600, 900).
        hi_range: (float) [waists] |OPTIONAL| Radial extent of the LASY grid in units of `waist`. Defaults to 8.0.
        center_and_remove_tilt: (bool) |OPTIONAL| Re-centre the fluence and remove the mean transverse phase gradient at the start plane. Defaults to True.
        centering_angles: (int) |OPTIONAL| Number of polar angles used for centering; must be at least 2 * n_azimuthal_modes - 1. Defaults to 72.
        spectral_time_window_factor: (float) |OPTIONAL| Temporal half-width of the LASY grid in units of the transform-limited half-duration when `spectral_bandwidth` is nonzero. Defaults to 6.0.
        peak_delay_from_file_start: (float|None) [s] |OPTIONAL| Place the on-axis intensity peak at this delay after the start of the LASY time window (the delay FBPIC uses for emission). None (default) leaves the peak at the window centre.
        maximum_pulse_duration_fwhm: (float|None) [s] |OPTIONAL| Reject the build if the start-plane on-axis FWHM exceeds this duration. None (default) disables the check.
        lasy_file: (Path|str) |OPTIONAL| Output prefix for the LASY HDF5 file; the written file is `<parent>/<stem>_00000.h5`. If not absolute, this is relative to the `working_directory` passed to `Simulation.setup_simulation()` (or to `prepare(relative_to=...)`), falling back to the current working directory. Defaults to `diags/lasy_laser`.
        t_start: (float) [s] |OPTIONAL| Delay before the antenna starts emitting the LASY file, as in FBPIC's `FromLasyFileLaser`. Defaults to 0.0.
    """

    SUBCLASS: ClassVar[str] = "lasy"

    # Physical parameters (required ones first, then optional pass-throughs that
    # keep HighOrderLasyLaser's own default while None).
    wavelength: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    tau_fwhm: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    waist: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    focal_position: float = attrs.field(converter=float)
    super_gaussian_order: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    zernike_coefficients: dict[str, float] = attrs.field(
        default=None, converter=_normalize_zernike_coefficients, hash=False
    )
    spectral_bandwidth: float | Literal["auto"] | None = attrs.field(
        default=None, converter=_optional_bandwidth
    )
    cep: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )
    gdd: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )
    tod: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )
    fod: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )
    gdd_relative: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )
    tod_relative: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )

    # Hyperparameters, with HighOrderLasyLaser's defaults.
    polarization: tuple[float, float] = attrs.field(
        default=(1.0, 0.0), converter=_pair_of(float), validator=_has_two_entries
    )
    n_azimuthal_modes: int = attrs.field(default=5, converter=int)
    num_points: tuple[int, int] = attrs.field(
        default=(600, 900), converter=_pair_of(int), validator=_has_two_entries
    )
    hi_range: float = attrs.field(default=8.0, converter=float)
    center_and_remove_tilt: bool = attrs.field(default=True, converter=bool)
    centering_angles: int = attrs.field(default=72, converter=int)
    spectral_time_window_factor: float = attrs.field(default=6.0, converter=float)
    peak_delay_from_file_start: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )
    maximum_pulse_duration_fwhm: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )

    # FBPIC-side settings.
    lasy_file: Path = attrs.field(default=Path("diags/lasy_laser"), converter=Path)
    t_start: float = attrs.field(default=0.0, converter=float)

    # Filled by prepare(). The HighOrderLasyLaser exists only on the rank that
    # built it; the written file path is known on every rank.
    _high_order_laser: Any = attrs.field(init=False, default=None, repr=False, eq=False)
    lasy_file_path: Path | None = attrs.field(
        init=False, default=None, repr=False, eq=False
    )

    # Field name -> HighOrderLasyLaser key. None-valued fields are not forwarded.
    _PHYSICAL_KEYS: ClassVar[dict[str, str]] = {
        "wavelength": "laser_wavelength_m",
        "energy": "laser_energy_J",
        "tau_fwhm": "laser_pulse_duration_fwhm_s",
        "waist": "laser_spot_size_m",
        "super_gaussian_order": "laser_super_gaussian_order",
        "focal_position": "laser_focal_position_m",
        "spectral_bandwidth": "laser_spectral_bandwidth_rad_s",
        "cep": "laser_cep_phase_rad",
        "gdd": "laser_gdd_s2",
        "tod": "laser_tod_s3",
        "fod": "laser_fod_s4",
        "gdd_relative": "laser_gdd_relative",
        "tod_relative": "laser_tod_relative",
    }
    _HYPERPARAMETER_KEYS: ClassVar[dict[str, str]] = {
        "polarization": "polarization",
        "n_azimuthal_modes": "n_azimuthal_modes",
        "num_points": "num_points",
        "hi_range": "hi_range",
        "center_and_remove_tilt": "center_and_remove_tilt",
        "centering_angles": "centering_angles",
        "spectral_time_window_factor": "spectral_time_window_factor",
        "peak_delay_from_file_start": "peak_delay_from_file_start_s",
        "maximum_pulse_duration_fwhm": "maximum_pulse_duration_fwhm_s",
    }

    def __attrs_post_init__(self) -> None:
        # Deliberately does not call the base implementation: a0 is derived
        # numerically at build time, not analytically at construction.
        if self.a0 is not None:
            raise ValueError(
                "LasyLaserPulse derives a0 numerically from energy during prepare(); "
                "do not pass a0."
            )
        if self.energy is None or self.energy <= 0:
            raise ValueError("energy must be provided and > 0.")
        object.__setattr__(self, "_amplitude_source", "energy")

        if self.method not in (None, "antenna"):
            raise ValueError(
                "LasyLaserPulse can only be emitted with method='antenna'."
            )
        object.__setattr__(self, "method", "antenna")
        if self.v_antenna is not None and self.v_antenna != 0.0:
            raise ValueError(
                "LasyLaserPulse requires a stationary antenna (v_antenna=0)."
            )
        object.__setattr__(self, "v_antenna", 0.0)
        if self.z0_antenna is None:
            raise ValueError(
                "z0_antenna is required: LASY pulses are emitted by an antenna."
            )

    # ------------------------------------------------------------------
    # Mapping onto HighOrderLasyLaser
    # ------------------------------------------------------------------

    @property
    def physical_parameters(self) -> dict[str, Any]:
        """``HighOrderLasyLaser`` physical parameters; unset optional fields are omitted."""
        parameters = {
            key: getattr(self, name)
            for name, key in self._PHYSICAL_KEYS.items()
            if getattr(self, name) is not None
        }
        parameters.update(
            {
                f"zernike_{name}": value
                for name, value in self.zernike_coefficients.items()
            }
        )
        return parameters

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """``HighOrderLasyLaser`` hyperparameters built from this config."""
        return {
            key: getattr(self, name) for name, key in self._HYPERPARAMETER_KEYS.items()
        }

    @property
    def is_prepared(self) -> bool:
        """Whether ``prepare()`` has produced the LASY file."""
        return self.lasy_file_path is not None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def resolve_lasy_file(self, relative_to: Path | str | None = None) -> Path:
        """Absolute output prefix for the LASY file.

        A relative ``lasy_file`` is anchored at *relative_to* when given, otherwise
        at the current working directory.
        """
        if self.lasy_file.is_absolute():
            return self.lasy_file
        base = Path(relative_to) if relative_to is not None else Path.cwd()
        return (base / self.lasy_file).resolve()

    def prepare(
        self, comm: Any | None = None, *, relative_to: Path | str | None = None
    ) -> None:
        """
        Build the LASY pulse, write its HDF5 file, and measure a0 at focus.

        Runs the expensive build on rank 0 only. With MPI, the other ranks block
        in a broadcast until rank 0 has either written the file (they receive
        its path and a0) or failed (they receive the error and raise a
        ``RuntimeError`` naming it, while rank 0 re-raises the original
        exception). Calling this again after a successful build is a no-op.

        Args:
            comm: (BoundaryCommunicator|mpi4py.MPI.Comm|None) Communicator for the
                rank-0 build and result broadcast. Accepts FBPIC's ``sim.comm``, an mpi4py
                communicator such as ``MPI.COMM_WORLD``, or ``None`` when running
                without MPI (the calling process builds the file itself).
            relative_to: (Path|str|None) Directory a relative ``lasy_file`` is written
                under. ``Simulation`` passes its ``working_directory``; ``None`` means
                the current working directory.
        """
        if self.is_prepared:
            return

        rank, mpi_comm = _resolve_comm(comm)

        payload: tuple[str, float] | None = None
        error: Exception | None = None
        if rank == 0:
            try:
                from inversion_fbpic.utils.laser import HighOrderLasyLaser

                high_order_laser = HighOrderLasyLaser(
                    self.physical_parameters, self.hyperparameters
                )
                written_path = high_order_laser.save(
                    self.resolve_lasy_file(relative_to)
                )
                focus_a0 = high_order_laser.compute_focus_a0()
            except Exception as exc:  # forwarded to the other ranks below
                error = exc
            else:
                object.__setattr__(self, "_high_order_laser", high_order_laser)
                payload = (str(written_path.resolve()), float(focus_a0))

        # The broadcast synchronizes the ranks, so no separate barrier is needed.
        # Sending the failure too keeps a bad config from hanging ranks 1..N.
        message = None if error is None else f"{type(error).__name__}: {error}"
        if mpi_comm is not None:
            payload, message = mpi_comm.bcast((payload, message), root=0)
        if error is not None:
            raise error  # rank 0 keeps the original exception and traceback
        if message is not None:
            raise RuntimeError(f"LASY build failed on rank 0: {message}")
        if payload is None:
            raise RuntimeError("Rank 0 did not produce the LASY laser file.")

        path_str, focus_a0 = payload
        lasy_file_path = Path(path_str)
        if not lasy_file_path.is_file():
            raise FileNotFoundError(
                f"LASY laser file was not created: {lasy_file_path}"
            )
        object.__setattr__(self, "lasy_file_path", lasy_file_path)
        object.__setattr__(self, "a0", focus_a0)
        object.__setattr__(self, "out_a0", focus_a0)

    def to_dict(self, *, include_nones: bool = True) -> dict[str, Any]:
        payload = super().to_dict(include_nones=include_nones)
        # ``lasy_file`` is an output prefix anchored at the simulation working
        # directory, not an input file, so it is written verbatim rather than
        # relative to the config file being saved.
        payload["parameters"]["lasy_file"] = self.lasy_file.as_posix()
        return payload

    def resolve_laser_energy(self) -> float:
        return float(self.energy)

    def resolve_laser_a0(self) -> float:
        if self.out_a0 is None:
            raise RuntimeError(
                "a0 is measured from the LASY field during prepare(); call prepare() first."
            )
        return float(self.out_a0)

    def build_laser_profile(self) -> LaserProfile | list[LaserProfile]:
        from fbpic.lpa_utils.laser.laser_profiles import FromLasyFileLaser

        if not self.is_prepared:
            self.prepare(None)
        return FromLasyFileLaser(str(self.lasy_file_path), t_start=self.t_start)

    # ------------------------------------------------------------------
    # Extents and plotting
    # ------------------------------------------------------------------

    def get_r_extent(
        self, simulation_extent: tuple[float, float], num_sigma: float = 3.0
    ) -> float:
        """
        Get the radial extent of the laser pulse in meters. Assumes vacuum propagation.
        This uses a Gaussian approximation, and higher order modes can
        extend the radial extent of the pulse beyond this value. Use with caution.

        Args:
            simulation_extent: (tuple[float, float]) The extent of the full simulation in meters.
            num_sigma: (float) Number of sigmas to account for in the radial extent.

        Returns:
            float: The radial extent of the laser pulse in meters.
        """
        print(
            "WARNING: LasyLaserPulse.get_r_extent() uses a Gaussian approximation, and higher order modes can extend the radial extent of the pulse beyond this value. Use with caution."
        )
        return _gaussian_r_extent(
            self.waist,
            self.wavelength,
            self.focal_position,
            simulation_extent,
            num_sigma,
        )

    def plot(
        self,
        *,
        mode: Literal["lineout", "lineout_and_2d"] = "lineout_and_2d",
        ax: "plt.Axes | None" = None,
        num: int = 600,
        output_path: Path | str | None = None,
        show: bool = False,
        label: str | None = None,
    ) -> "plt.Figure":
        """Plot the start-plane LASY envelope and (optionally) a face-on |E| map.

        Delegates to ``utils.laser.plot_start_plane``. Needs the LASY ``Laser``
        object, so this only works on the rank that ran ``prepare()`` (it runs
        ``prepare()`` itself if needed).

        Args:
            mode: (Literal["lineout", "lineout_and_2d"]) Panel layout.
                ``"lineout"`` shows only the on-axis longitudinal envelope.
                ``"lineout_and_2d"`` (default) adds a face-on field-amplitude
                map at the start plane with a quiver overlay of the
                polarization field.
            ax: (matplotlib.axes.Axes|None) If provided, the longitudinal envelope
                is also drawn on this external axes (for combined overlay figures).
            num: (int) Number of points for the resampled longitudinal lineout.
            output_path: (Path|str|None) If given, the figure is saved here.
            show: (bool) Whether to call ``plt.show()``.
            label: (str|None) Label for the external *ax* lineout. Defaults to
                ``SUBCLASS (polarization)``.

        Returns:
            The created matplotlib Figure.
        """
        from inversion_fbpic.utils.laser import plot_start_plane

        if not self.is_prepared:
            self.prepare(None)
        if self._high_order_laser is None:
            raise RuntimeError(
                "plot() needs the LASY Laser object, which only exists on the rank "
                "that ran prepare()."
            )
        return plot_start_plane(
            self._high_order_laser.laser,
            z0=self.z0,
            polarization=self.polarization,
            mode=mode,
            ax=ax,
            num=num,
            output_path=output_path,
            show=show,
            label=label,
            title=self.SUBCLASS,
            a0_annotation=self.out_a0,
        )

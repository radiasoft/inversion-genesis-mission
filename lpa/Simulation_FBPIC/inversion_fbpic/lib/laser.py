"""
Attrs-based laser profile configuration.
"""

from __future__ import annotations

import copy
from abc import abstractmethod
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable, ClassVar, TYPE_CHECKING, Literal, Iterable

import attrs

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from fbpic.lpa_utils.laser.laser_profiles import LaserProfile

import numpy as np
import numpy.typing as npt

from scipy.constants import c, pi

from inversion_fbpic.lib.serializable_config import SerializableConfig
from inversion_fbpic.utils.simulation_setup_tools import (
    calculate_laser_a0_from_energy,
    calculate_laser_energy_from_a0,
    calculate_laser_tau_from_fwhm_intensity,
)

DensityCallable = Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]


def _ensure_profile_list(
    profiles: "LaserProfile | list[LaserProfile]",
) -> "list[LaserProfile]":
    if isinstance(profiles, list):
        return profiles
    return [profiles]


def _format_polarization(pol: "float | list | str") -> str:
    """Return a human-readable polarization label."""
    if isinstance(pol, str):
        return f"{pol} circular"
    if isinstance(pol, Iterable) and not isinstance(pol, str):
        seq = list(pol)
        if len(seq) == 2:
            return (
                f"elliptical (\u03b8={np.degrees(seq[0]):.1f}\u00b0, "
                f"\u03b2={np.degrees(seq[1]):.1f}\u00b0)"
            )
    return f"linear (\u03b8={np.degrees(pol):.1f}\u00b0)"


def _format_jones(pol: tuple[float, float]) -> str:
    """Return a human-readable label for a real Jones vector."""
    return f"Jones ({pol[0]:g}, {pol[1]:g})"


def _zernike_names() -> tuple[str, ...]:
    """Canonical Zernike coefficient names accepted by ``HighOrderLasyLaser``."""
    # Lazy import: ``inversion_fbpic.utils.laser`` pulls in lasy.
    from inversion_fbpic.utils.laser import ZERNIKE_OSA_INDICES

    return tuple(ZERNIKE_OSA_INDICES)


def _default_zernike_coefficients() -> dict[str, float]:
    return {name: 0.0 for name in _zernike_names()}


def _normalize_zernike_coefficients(value: Any) -> dict[str, float]:
    """Validate Zernike names and return a complete, canonically ordered dict."""
    if value is None:
        return _default_zernike_coefficients()
    if not isinstance(value, Mapping):
        raise ValueError("zernike_coefficients must be a mapping of name -> amplitude.")
    names = _zernike_names()
    unknown = set(value) - set(names)
    if unknown:
        raise ValueError(
            f"Unknown zernike_coefficients keys: {sorted(unknown)}. "
            f"Allowed: {list(names)}"
        )
    return {name: float(value.get(name, 0.0)) for name in names}


def _as_pair(value: Any, name: str) -> tuple[Any, Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a sequence of two numbers.")
    seq = list(value)
    if len(seq) != 2:
        raise ValueError(f"{name} must have exactly two entries, got {len(seq)}.")
    return seq[0], seq[1]


def _to_float_pair(value: Any) -> tuple[float, float]:
    """Convert a two-element sequence to a tuple of floats (real Jones vector)."""
    first, second = _as_pair(value, "polarization")
    return (float(first), float(second))


def _to_int_pair(value: Any) -> tuple[int, int]:
    """Convert a two-element sequence to a tuple of ints, each at least 2."""
    first, second = _as_pair(value, "num_points")
    pair = (int(first), int(second))
    if min(pair) < 2:
        raise ValueError("num_points entries must be at least 2.")
    return pair


_DERIVED_YAML_KEYS = ("out_a0", "out_energy")


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


def _gaussian_r_extent(
    waist: float,
    wavelength: float,
    focal_position: float,
    simulation_extent: tuple[float, float],
    num_sigma: float,
) -> float:
    """Radial extent of a Gaussian beam over *simulation_extent*, from its Rayleigh length."""
    rayleigh_length = pi * waist**2 / wavelength
    waist_max = waist * np.sqrt(
        1
        + max(
            abs(simulation_extent[1] - focal_position),
            abs(focal_position - simulation_extent[0]),
        )
        ** 2
        / rayleigh_length**2
    )
    return waist_max * num_sigma / 2.0


@attrs.define(kw_only=True, slots=False, frozen=True)
class _LaserPulse(SerializableConfig):
    """
    Base class for serializable laser profile configurations.

    Provide exactly one of ``energy`` or ``a0``; the other is derived from beam
    parameters at construction time.  When serialized (YAML / JSON) the derived
    companion is written as ``out_a0`` or ``out_energy`` for reference but is
    ignored on load.

    Args:
        energy: (float|None) [J] |OPTIONAL| Energy of the laser pulse in Joules. Provide this or a0, not both.
        a0: (float|None) |OPTIONAL| Normalized laser amplitude parameter. Provide this or energy, not both. Not accepted by ``LasyLaserPulse``, which derives it numerically.
        z0: (float) [m] Position of the laser pulse within the simulation window in meters.
        method: (Literal["direct", "antenna"]|None) |OPTIONAL| Method to use for laser pulse propagation. If None, the laser pulse will be propagated using the direct method.
        z0_antenna: (float|None) [m] |OPTIONAL| Position of the antenna within the simulation window in meters. Required if method is "antenna".
        v_antenna: (float|None) [m/s] |OPTIONAL| Velocity of the antenna in meters per second. Required if method is "antenna".
    """

    energy: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )
    a0: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float), hash=False
    )
    z0: float = attrs.field(converter=float)
    method: Literal["direct", "antenna"] | None = attrs.field(default=None)
    z0_antenna: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )
    v_antenna: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )

    # Derived amplitude fields are excluded from the hash: ``LasyLaserPulse``
    # fills ``a0``/``out_a0`` during ``prepare()``, after construction.
    _amplitude_source: Literal["energy", "a0"] = attrs.field(
        init=False, repr=False, hash=False
    )
    out_a0: float | None = attrs.field(init=False, default=None, repr=False, hash=False)
    out_energy: float | None = attrs.field(
        init=False, default=None, repr=False, hash=False
    )

    # CONFIG_TYPE identifies this domain in the top-level registry held on
    # SerializableConfig. It is set on the domain base class (here) only;
    # concrete subclasses inherit it and must NOT override it.
    CONFIG_TYPE: ClassVar[str] = "laser_pulse"

    # Per-domain registry: SUBCLASS -> concrete LaserProfile subclass.
    # Concrete subclasses register into this dict automatically by declaring
    # SUBCLASS in their own class body.
    _CONCRETE_REGISTRY: ClassVar[dict[str, type["_LaserPulse"]]] = {}

    def __attrs_post_init__(self) -> None:
        has_energy = self.energy is not None
        has_a0 = self.a0 is not None
        if has_energy == has_a0:
            raise ValueError("Provide exactly one of energy or a0.")
        if has_energy:
            if self.energy <= 0:
                raise ValueError("energy must be > 0.")
            object.__setattr__(self, "_amplitude_source", "energy")
            object.__setattr__(self, "a0", float(self.resolve_laser_a0()))
            object.__setattr__(self, "out_a0", self.a0)
        else:
            if self.a0 <= 0:
                raise ValueError("a0 must be > 0.")
            object.__setattr__(self, "_amplitude_source", "a0")
            object.__setattr__(self, "energy", float(self.resolve_laser_energy()))
            object.__setattr__(self, "out_energy", self.energy)

    def to_dict(self, *, include_nones: bool = True) -> dict[str, Any]:
        payload = super().to_dict(include_nones=include_nones)
        params = payload["parameters"]
        # Remove the undefined input and add in the derived output.
        if self._amplitude_source == "energy":
            params.pop("a0", None)
            params.pop("out_energy", None)
            if self.out_a0 is not None or include_nones:
                params["out_a0"] = self.out_a0
        else:
            params.pop("energy", None)
            params.pop("out_a0", None)
            if self.out_energy is not None or include_nones:
                params["out_energy"] = self.out_energy
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        overrides: dict[str, Any] | None = None,
    ) -> "_LaserPulse":
        # Remove the derived output keys from the input payload.
        payload = copy.deepcopy(payload)
        params = payload.get("parameters", {})
        if isinstance(params, dict):
            for key in _DERIVED_YAML_KEYS:
                params.pop(key, None)
        return super().from_dict(payload, overrides=overrides)

    def prepare(
        self, comm: Any | None = None, *, relative_to: Path | str | None = None
    ) -> None:
        """
        Perform one-time, possibly collective, setup before ``build_laser_profile``.

        ``Simulation.setup_simulation`` calls this on every MPI rank before adding
        the laser. The base implementation is a no-op; subclasses that need to
        write files or run expensive precomputation (e.g. ``LasyLaserPulse``)
        override it.

        Args:
            comm: (BoundaryCommunicator|mpi4py.MPI.Comm|None) Communicator for
                collective setup. Accepts FBPIC's ``sim.comm``, an mpi4py
                communicator such as ``MPI.COMM_WORLD``, or ``None`` when running
                without MPI (everything happens on the calling process).
            relative_to: (Path|str|None) Directory against which relative output paths
                are resolved. ``Simulation`` passes its ``working_directory``; ``None``
                means the current working directory.
        """
        return None

    def get_z_extent(self, num_sigma: float = 3.0) -> tuple[float, float]:
        """
        Get the longitudinal extent of the laser pulse in meters.

        Args:
            num_sigma: (float) Number of sigmas to account for in the longitudinal extent.

        Returns:
            tuple[float, float]: The longitudinal extent of the laser pulse in meters.
        """
        tau = calculate_laser_tau_from_fwhm_intensity(self.tau_fwhm)
        return (self.z0 - num_sigma * tau * c, self.z0 + num_sigma * tau * c)

    @abstractmethod
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
        """Plot the laser pulse envelope and (optionally) a face-on x-y cross-section.

        Args:
            mode: (Literal["lineout", "lineout_and_2d"]) Panel layout.
                ``"lineout"`` shows only the longitudinal envelope.
                ``"lineout_and_2d"`` (default) adds a face-on x-y field-amplitude
                map at the focal plane with a quiver overlay showing polarization.
            ax: (matplotlib.axes.Axes|None) If provided, the longitudinal envelope
                is also drawn on this external axes (for combined overlay figures).
            num: (int) Grid resolution along each axis.
            output_path: (Path|str|None) If given, the figure is saved here.
            show: (bool) Whether to call ``plt.show()``.
            label: (str|None) Label for the external *ax* lineout. Defaults to
                ``SUBCLASS (polarization)``.

        Returns:
            The created matplotlib Figure.
        """

    @abstractmethod
    def get_r_extent(
        self, simulation_extent: tuple[float, float], num_sigma: float = 3.0
    ) -> float:
        """
        Get the radial extent of the laser pulse in meters.

        Args:
            simulation_extent: (tuple[float, float]) The extent of the full simulation in meters.
            num_sigma: (float) Number of sigmas to account for in the radial extent.

        Returns:
            float: The radial extent of the laser pulse in meters.
        """

    @abstractmethod
    def resolve_laser_energy(self) -> float:
        """
        Resolve the laser energy from the laser pulse parameters.
        """

    @abstractmethod
    def resolve_laser_a0(self) -> float:
        """
        Resolve the laser a0 from the laser pulse parameters.
        """

    @abstractmethod
    def build_laser_profile(self) -> LaserProfile | list[LaserProfile]:
        """
        Return a LaserProfile object or list of LaserProfile objects that represents the laser pulse.
        """


@attrs.define(kw_only=True, slots=False, frozen=True)
class _GaussianTemporalLaserPulse(_LaserPulse):
    """
    Laser pulse with an explicit Gaussian temporal (longitudinal) profile.
    No transverse profile is defined; do not build this abstract class.

    Args:
        wavelength: (float) [m] Wavelength of the laser pulse in meters.
        tau_fwhm: (float) [s] Full-width at half-maximum temporal duration of the laser pulse in seconds.
        cep: (float) [rad] Carrier envelope phase of the laser pulse in radians.
        waist: (float) [m] Waist radius of the laser pulse in meters.
        focal_position: (float) [m] Focal position of the laser pulse in meters.
        polarization: (float|tuple[float, float]|Literal["left", "right"]) [rad] Polarization angle of the laser pulse in radians.
            If a float is provided, it is interpreted as the polarization angle theta in the x-y plane.
            If a tuple is provided, it is interpreted as the Jones vector angles (theta, beta) with:
                a_x = a_0 * cos(theta) * phi(z,t)
                a_y = a_0 * sin(theta) * exp(i beta) * phi(z,t)
                phi(z,t) = exp(i (2pi/lambda) (z - z0) + i (omega t - cep))
                e.g., (pi/4,0) is linear polarization at 45 degrees to x, (pi/4, pi/2) is right circular polarization
            If "left" is provided, it is interpreted as a left-hand circular polarization with maximum laser parameter a0/sqrt(2).
            If "right" is provided, it is interpreted as a right-hand circular polarization with maximum laser parameter a0/sqrt(2).
    """

    wavelength: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    tau_fwhm: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    cep: float = attrs.field(converter=float)
    waist: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    focal_position: float = attrs.field(converter=float)
    polarization: float | tuple[float, float] | Literal["left", "right"] = attrs.field()

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if isinstance(self.polarization, (tuple, list)):
            object.__setattr__(
                self, "polarization", tuple(float(p) for p in self.polarization)
            )
        elif isinstance(self.polarization, str):
            object.__setattr__(self, "polarization", self.polarization.lower())
            if self.polarization not in ["left", "right"]:
                raise ValueError("Invalid polarization type.")
        elif isinstance(self.polarization, (float, int)):
            object.__setattr__(self, "polarization", float(self.polarization))

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
        """Plot the laser pulse envelope and (optionally) a face-on x-y cross-section.

        Args:
            mode: (Literal["lineout", "lineout_and_2d"]) Panel layout.
                ``"lineout"`` shows only the longitudinal envelope.
                ``"lineout_and_2d"`` (default) adds a face-on x-y field-amplitude
                map at the focal plane with a quiver overlay showing polarization.
            ax: (matplotlib.axes.Axes|None) If provided, the longitudinal envelope
                is also drawn on this external axes (for combined overlay figures).
            num: (int) Grid resolution along each axis.
            output_path: (Path|str|None) If given, the figure is saved here.
            show: (bool) Whether to call ``plt.show()``.
            label: (str|None) Label for the external *ax* lineout. Defaults to
                ``SUBCLASS (polarization)``.

        Returns:
            The created matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        profiles = _ensure_profile_list(self.build_laser_profile())
        pol_label = _format_polarization(self.polarization)

        z_min, z_max = self.get_z_extent()
        z_arr = np.linspace(z_min, z_max, num)

        envelope_sq = np.zeros(num)
        for p in profiles:
            e0 = np.hypot(p.E0x, p.E0y)
            for i, zi in enumerate(z_arr):
                comp = e0 * np.abs(p.longitudinal_profile.evaluate(z=zi, t=0.0))
                envelope_sq[i] += comp**2
        omega = 2 * pi * c / self.wavelength
        from scipy.constants import e as q_e, m_e as m_electron

        e_to_a0 = q_e / (m_electron * c * omega)
        envelope = np.sqrt(envelope_sq) * e_to_a0

        if ax is not None:
            default_label = f"{self.SUBCLASS} ({pol_label})"
            ax.plot(
                z_arr * 1e3,
                envelope,
                lw=1.5,
                label=label if label is not None else default_label,
            )

        if mode == "lineout":
            fig, ax_z = plt.subplots(1, 1, figsize=(8, 4.5))
        else:
            fig, (ax_z, ax_xy) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax_z.plot(z_arr * 1e6, envelope, color="C0", lw=1.5)
        ax_z.set_xlabel("z (um)")
        ax_z.set_ylabel("Envelope amplitude (a₀)")
        ax_z.set_title(f"Longitudinal envelope\n{pol_label}")
        ax_z.grid(True, alpha=0.3)

        if mode == "lineout_and_2d":
            r_ext = self.get_r_extent((z_min, z_max))
            half = r_ext * 1.2
            n_xy = min(num, 200)
            xy = np.linspace(-half, half, n_xy)
            xm, ym = np.meshgrid(xy, xy, indexing="xy")
            ex_grid = np.zeros_like(xm)
            ey_grid = np.zeros_like(xm)
            z_slice = self.z0

            for p in profiles:
                for i in range(n_xy):
                    for j in range(n_xy):
                        ex_val, ey_val = p.E_field(
                            x=xm[i, j], y=ym[i, j], z=z_slice, t=0.0
                        )
                        ex_grid[i, j] += ex_val
                        ey_grid[i, j] += ey_val
            amp = np.sqrt(ex_grid**2 + ey_grid**2)

            extent_um = (-half * 1e6, half * 1e6, -half * 1e6, half * 1e6)
            im = ax_xy.imshow(
                amp,
                origin="lower",
                aspect="equal",
                extent=extent_um,
                cmap="inferno",
            )
            ax_xy.set_xlabel(r"x ($\mu$m)")
            ax_xy.set_ylabel(r"y ($\mu$m)")
            ax_xy.set_title(f"Face-on |E| at z\u2080\n{pol_label}")
            cbar = fig.colorbar(im, ax=ax_xy, fraction=0.046, pad=0.04)
            cbar.set_label("|E| (V/m)")

            stride = max(1, n_xy // 15)
            xs = xm[::stride, ::stride]
            ys = ym[::stride, ::stride]
            us = ex_grid[::stride, ::stride]
            vs = ey_grid[::stride, ::stride]
            mag = np.sqrt(us**2 + vs**2)
            mask = mag > 0.05 * np.max(mag)
            if mask.any():
                ax_xy.quiver(
                    xs[mask] * 1e6,
                    ys[mask] * 1e6,
                    us[mask] / mag[mask],
                    vs[mask] / mag[mask],
                    color="white",
                    alpha=0.6,
                    scale=25,
                    width=0.004,
                    headwidth=3,
                )

        fig.suptitle(f"{self.SUBCLASS}  —  {pol_label}", fontsize=11)
        fig.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


@attrs.define(kw_only=True, slots=False, frozen=True)
class LasyLaserPulse(_LaserPulse):
    """
    Super-Gaussian laser pulse with Zernike aberrations, built with LASY.

    Wraps ``inversion_fbpic.utils.laser.HighOrderLasyLaser``: the pulse is
    constructed at focus, back-propagated to the simulation start plane,
    optionally re-centred, normalized to ``energy``, and written to a LASY HDF5
    file that FBPIC reads through ``FromLasyFileLaser``. The build is expensive
    and happens once, in ``prepare()``, on MPI rank 0 only; other ranks wait at
    a barrier and receive the file path.

    Only ``energy`` may be provided. ``a0`` is measured numerically from the
    field at focus during ``prepare()`` and is reported as ``out_a0``
    (``null`` in YAML written before the build).

    LASY pulses can only be emitted through an antenna, so ``method`` is fixed
    to ``"antenna"``, ``v_antenna`` to ``0.0``, and ``z0_antenna`` is required.
    FBPIC resets the LASY time axis to zero, so the peak intensity leaves the
    antenna at ``t = t_start + 3 * tau_fwhm``. ``z0`` is informational only
    (nominal centroid at ``t = 0``, used for plotting extents); for a
    consistent picture set ``z0 = z0_antenna - c * (t_start + 3 * tau_fwhm)``.

    Args:
        wavelength: (float) [m] Central wavelength of the laser pulse in meters.
        tau_fwhm: (float) [s] Full-width at half-maximum intensity duration of the laser pulse in seconds.
        waist: (float) [m] Super-Gaussian spot size (1/e^2 radius for order 2) at focus in meters.
        focal_position: (float) [m] Focal position of the laser pulse in meters, relative to the simulation start plane.
        super_gaussian_order: (float) Super-Gaussian order of the transverse profile. 2.0 is Gaussian.
        zernike_coefficients: (dict[str, float]) [wavelengths] |OPTIONAL| Zernike phase amplitudes at focus keyed by name (astigmatism_2, astigmatism_4, coma_y, coma_x, trefoil_y, trefoil_x, spherical_3, astigmatism_6, coma_5_y, coma_5_x, secondary_trefoil_y, secondary_trefoil_x). Missing names default to 0.0; unknown names are rejected.
        polarization: (tuple[float, float]) |OPTIONAL| Real Jones vector (Ex, Ey) passed to LASY. Defaults to (1, 0), linear along x.
        n_azimuthal_modes: (int) |OPTIONAL| Number of azimuthal modes in the LASY r-t grid. Defaults to 5.
        num_points: (tuple[int, int]) |OPTIONAL| LASY grid points (radial, temporal). Defaults to (600, 900).
        hi_range: (float) [waists] |OPTIONAL| Radial extent of the LASY grid in units of `waist`. Defaults to 8.0.
        center_and_remove_tilt: (bool) |OPTIONAL| Re-centre the fluence and remove the mean transverse phase gradient at the start plane. Defaults to True.
        centering_angles: (int) |OPTIONAL| Number of polar angles used for centering; must be at least 2 * n_azimuthal_modes - 1. Defaults to 72.
        lasy_file: (Path|str) |OPTIONAL| Output prefix for the LASY HDF5 file; the written file is `<parent>/<stem>_00000.h5`. If not absolute, this is relative to the `working_directory` passed to `Simulation.setup_simulation()` (or to `prepare(relative_to=...)`), falling back to the current working directory. Defaults to `diags/lasy_laser`.
        t_start: (float) [s] |OPTIONAL| Delay before the antenna starts emitting the LASY file, as in FBPIC's `FromLasyFileLaser`. Defaults to 0.0.
    """

    SUBCLASS: ClassVar[str] = "lasy"

    wavelength: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    tau_fwhm: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    waist: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    focal_position: float = attrs.field(converter=float)
    super_gaussian_order: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    zernike_coefficients: dict[str, float] = attrs.field(
        factory=_default_zernike_coefficients,
        converter=_normalize_zernike_coefficients,
        hash=False,
    )
    polarization: tuple[float, float] = attrs.field(
        default=(1.0, 0.0), converter=_to_float_pair
    )
    n_azimuthal_modes: int = attrs.field(
        default=5, converter=int, validator=attrs.validators.ge(1)
    )
    num_points: tuple[int, int] = attrs.field(
        default=(600, 900), converter=_to_int_pair
    )
    hi_range: float = attrs.field(
        default=8.0, converter=float, validator=attrs.validators.gt(0.0)
    )
    center_and_remove_tilt: bool = attrs.field(default=True, converter=bool)
    centering_angles: int = attrs.field(
        default=72, converter=int, validator=attrs.validators.ge(1)
    )
    lasy_file: Path = attrs.field(default=Path("diags/lasy_laser"), converter=Path)
    t_start: float = attrs.field(default=0.0, converter=float)

    # Filled by prepare(). The HighOrderLasyLaser exists only on the rank that
    # built it; the written file path is known on every rank.
    _high_order_laser: Any = attrs.field(init=False, default=None, repr=False, eq=False)
    lasy_file_path: Path | None = attrs.field(
        init=False, default=None, repr=False, eq=False
    )

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

        minimum_angles = 2 * self.n_azimuthal_modes - 1
        if self.centering_angles < minimum_angles:
            raise ValueError(
                "centering_angles must be at least "
                f"{minimum_angles} for the configured azimuthal modes."
            )
        if np.hypot(*self.polarization) == 0.0:
            raise ValueError("polarization must be a non-zero Jones vector.")

    # ------------------------------------------------------------------
    # Mapping onto HighOrderLasyLaser
    # ------------------------------------------------------------------

    @property
    def physical_parameters(self) -> dict[str, Any]:
        """``HighOrderLasyLaser`` physical parameters built from this config."""
        parameters: dict[str, Any] = {
            "laser_wavelength_m": self.wavelength,
            "laser_energy_J": self.energy,
            "laser_pulse_duration_fwhm_s": self.tau_fwhm,
            "laser_spot_size_m": self.waist,
            "laser_super_gaussian_order": self.super_gaussian_order,
            "laser_focal_position_m": self.focal_position,
        }
        parameters.update(
            {
                f"zernike_{name}": amplitude
                for name, amplitude in self.zernike_coefficients.items()
            }
        )
        return parameters

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """``HighOrderLasyLaser`` hyperparameters built from this config."""
        return {
            "polarization": self.polarization,
            "n_azimuthal_modes": self.n_azimuthal_modes,
            "num_points": self.num_points,
            "hi_range": self.hi_range,
            "center_and_remove_tilt": self.center_and_remove_tilt,
            "centering_angles": self.centering_angles,
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

        Runs the expensive build on rank 0 only. With MPI, the other ranks wait
        at a barrier and then receive the written path and a0 by broadcast.
        Calling this again after a successful build is a no-op.

        Args:
            comm: (BoundaryCommunicator|mpi4py.MPI.Comm|None) Communicator for the
                rank-0 build and barrier. Accepts FBPIC's ``sim.comm``, an mpi4py
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
        if rank == 0:
            from inversion_fbpic.utils.laser import HighOrderLasyLaser

            high_order_laser = HighOrderLasyLaser(
                self.physical_parameters, self.hyperparameters
            )
            written_path = high_order_laser.save(self.resolve_lasy_file(relative_to))
            focus_a0 = high_order_laser.compute_focus_a0()
            object.__setattr__(self, "_high_order_laser", high_order_laser)
            payload = (str(written_path.resolve()), float(focus_a0))

        if mpi_comm is not None:
            mpi_comm.barrier()
            payload = mpi_comm.bcast(payload, root=0)
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

        Requires the LASY ``Laser`` object, so this only works on the rank that
        ran ``prepare()`` (it runs ``prepare()`` itself if needed).

        Args:
            mode: (Literal["lineout", "lineout_and_2d"]) Panel layout.
                ``"lineout"`` shows only the on-axis longitudinal envelope.
                ``"lineout_and_2d"`` (default) adds a face-on field-amplitude
                map at the start plane with the polarization direction marked.
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
        import matplotlib.pyplot as plt
        from scipy.constants import e as q_e, m_e as m_electron

        from inversion_fbpic.utils.laser import polar_fields, transverse_fluence

        if not self.is_prepared:
            self.prepare(None)
        if self._high_order_laser is None:
            raise RuntimeError(
                "plot() needs the LASY Laser object, which only exists on the rank "
                "that ran prepare()."
            )
        laser = self._high_order_laser.laser
        _, time = laser.grid.axes
        e_to_a0 = q_e / (m_electron * c * laser.profile.omega0)
        pol_label = _format_jones(self.polarization)

        # On-axis envelope vs. time, mapped to z about the nominal centroid z0.
        on_axis = np.abs(polar_fields(laser, np.array([0.0]))[0, 0, :]) * e_to_a0
        t_peak = time[int(np.argmax(on_axis))]
        z_of_t = self.z0 - c * (time - t_peak)
        order = np.argsort(z_of_t)
        z_arr = np.linspace(z_of_t.min(), z_of_t.max(), num)
        envelope = np.interp(z_arr, z_of_t[order], on_axis[order])

        if ax is not None:
            default_label = f"{self.SUBCLASS} ({pol_label})"
            ax.plot(
                z_arr * 1e3,
                envelope,
                lw=1.5,
                label=label if label is not None else default_label,
            )

        if mode == "lineout":
            fig, ax_z = plt.subplots(1, 1, figsize=(8, 4.5))
        else:
            fig, (ax_z, ax_xy) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax_z.plot(z_arr * 1e6, envelope, color="C0", lw=1.5)
        ax_z.set_xlabel("z (um)")
        ax_z.set_ylabel("On-axis envelope amplitude (a\u2080)")
        ax_z.set_title(f"Longitudinal envelope at start plane\n{pol_label}")
        ax_z.grid(True, alpha=0.3)

        if mode == "lineout_and_2d":
            radius, angles, fluence, peak_field = transverse_fluence(laser, 361)
            amplitude = np.abs(peak_field)
            # Explicit polar cell edges: the Cartesian mesh is not monotonic, so
            # pcolormesh cannot infer them from cell centres.
            d_theta = angles[1] - angles[0]
            theta_edges = np.append(angles - d_theta / 2.0, angles[-1] + d_theta / 2.0)
            r_edges = np.concatenate(
                ([0.0], 0.5 * (radius[1:] + radius[:-1]), [radius[-1]])
            )
            theta_grid, r_grid = np.meshgrid(theta_edges, r_edges, indexing="ij")
            x_um = r_grid * np.cos(theta_grid) * 1e6
            y_um = r_grid * np.sin(theta_grid) * 1e6
            im = ax_xy.pcolormesh(x_um, y_um, amplitude, cmap="inferno", shading="flat")
            ax_xy.set_aspect("equal")

            # Zoom to where the azimuthally averaged fluence is above 1e-3 of peak.
            radial_fluence = fluence.mean(axis=0)
            above = np.flatnonzero(radial_fluence > 1e-3 * radial_fluence.max())
            r_view = radius[int(above[-1])] * 1e6 if above.size else radius[-1] * 1e6
            ax_xy.set_xlim(-r_view, r_view)
            ax_xy.set_ylim(-r_view, r_view)

            ax_xy.set_xlabel(r"x ($\mu$m)")
            ax_xy.set_ylabel(r"y ($\mu$m)")
            ax_xy.set_title(f"Face-on |E| at start plane\n{pol_label}")
            cbar = fig.colorbar(im, ax=ax_xy, fraction=0.046, pad=0.04)
            cbar.set_label("|E| (V/m)")

            px, py = self.polarization
            norm = float(np.hypot(px, py))
            arrow = 0.35 * r_view
            ax_xy.annotate(
                "",
                xy=(px / norm * arrow, py / norm * arrow),
                xytext=(-px / norm * arrow, -py / norm * arrow),
                arrowprops=dict(arrowstyle="<->", color="white", lw=1.5),
            )
            if self.out_a0 is not None:
                ax_xy.text(
                    0.02,
                    0.98,
                    f"a\u2080 at focus = {self.out_a0:.3g}",
                    transform=ax_xy.transAxes,
                    color="white",
                    va="top",
                    fontsize=9,
                )

        fig.suptitle(f"{self.SUBCLASS}  \u2014  {pol_label}", fontsize=11)
        fig.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


@attrs.define(kw_only=True, slots=False, frozen=True)
class GaussianLaserPulse(_GaussianTemporalLaserPulse):
    """
    Temporal-gaussian laser pulse with a Gaussian transverse profile.
    """

    SUBCLASS: ClassVar[str] = "gaussian"

    def get_r_extent(
        self, simulation_extent: tuple[float, float], num_sigma: float = 3.0
    ) -> float:
        return _gaussian_r_extent(
            self.waist,
            self.wavelength,
            self.focal_position,
            simulation_extent,
            num_sigma,
        )

    def resolve_laser_energy(self) -> float:
        return calculate_laser_energy_from_a0(
            self.a0, self.wavelength * 1e6, self.waist, self.tau_fwhm
        )

    def resolve_laser_a0(self) -> float:
        return calculate_laser_a0_from_energy(
            self.energy, self.wavelength * 1e6, self.waist, self.tau_fwhm
        )

    def build_laser_profile(self) -> LaserProfile | list[LaserProfile]:
        from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser

        base_params = {
            "waist": self.waist,
            "tau": calculate_laser_tau_from_fwhm_intensity(self.tau_fwhm),
            "z0": self.z0,
            "zf": self.focal_position,
            "lambda0": self.wavelength,
        }
        if self.polarization in ["left", "right"]:
            if self.polarization == "left":
                return [
                    GaussianLaser(
                        **base_params,
                        cep_phase=self.cep,
                        a0=self.a0 / np.sqrt(2.0),
                        theta_pol=0.0,
                    ),
                    GaussianLaser(
                        **base_params,
                        cep_phase=self.cep - pi / 2.0,
                        a0=self.a0 / np.sqrt(2.0),
                        theta_pol=pi / 2.0,
                    ),
                ]
            elif self.polarization == "right":
                return [
                    GaussianLaser(
                        **base_params,
                        cep_phase=self.cep,
                        a0=self.a0 / np.sqrt(2.0),
                        theta_pol=0.0,
                    ),
                    GaussianLaser(
                        **base_params,
                        cep_phase=self.cep + pi / 2.0,
                        a0=self.a0 / np.sqrt(2.0),
                        theta_pol=pi / 2.0,
                    ),
                ]
            else:
                raise ValueError("Invalid polarization type.")
        elif isinstance(self.polarization, Iterable):
            return [
                GaussianLaser(
                    **base_params,
                    cep_phase=self.cep,
                    a0=self.a0 * np.cos(self.polarization[0]),
                    theta_pol=0.0,
                ),
                GaussianLaser(
                    **base_params,
                    cep_phase=self.cep + self.polarization[1],
                    a0=self.a0 * np.sin(self.polarization[0]),
                    theta_pol=pi / 2.0,
                ),
            ]
        elif isinstance(self.polarization, (float, int)):
            return GaussianLaser(**base_params, a0=self.a0, theta_pol=self.polarization)
        else:
            raise ValueError("Invalid polarization type.")

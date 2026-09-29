"""
Attrs-based laser profile configuration.
"""

from __future__ import annotations

import copy
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar, TYPE_CHECKING, Literal, Iterable

import attrs

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from fbpic.lpa_utils.laser.laser_profiles import LaserProfile
    from lasy.laser import Laser as LasyLaser

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


_DERIVED_YAML_KEYS = ("out_a0", "out_energy")


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
        a0: (float|None) |OPTIONAL| Normalized laser amplitude parameter. Provide this or energy, not both.
        z0: (float) [m] Position of the laser pulse within the simulation window in meters.
        method: (Literal["direct", "antenna"]|None) |OPTIONAL| Method to use for laser pulse propagation. If None, the laser pulse will be propagated using the direct method.
        z0_antenna: (float|None) [m] |OPTIONAL| Position of the antenna within the simulation window in meters. Required if method is "antenna".
        v_antenna: (float|None) [m/s] |OPTIONAL| Velocity of the antenna in meters per second. Required if method is "antenna".
    """

    energy: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )
    a0: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )
    z0: float = attrs.field(converter=float)
    method: Literal["direct", "antenna"] | None = attrs.field(default=None)
    z0_antenna: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )
    v_antenna: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )

    _amplitude_source: Literal["energy", "a0"] = attrs.field(init=False, repr=False)
    out_a0: float | None = attrs.field(init=False, default=None, repr=False)
    out_energy: float | None = attrs.field(init=False, default=None, repr=False)

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
            params["out_a0"] = self.out_a0
            params.pop("out_energy", None)
        else:
            params.pop("energy", None)
            params["out_energy"] = self.out_energy
            params.pop("out_a0", None)
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


@attrs.define(kw_only=True, slots=False, init=False, frozen=True)
class LasyLaserPulse(_LaserPulse):
    """
    Laser pulse from a `Lasy` profile.
    This is not yet implemented!
    """

    SUBCLASS: ClassVar[str] = "lasy"

    # In order for serialization to work we need to be able to serialize the lasy Laser object.
    # Might need to be from a file and we store the path or something like that.
    def __init__(self, lasy_profile: LasyLaser) -> None:
        # use lazy import to avoid expensive imports (type checking import already done)
        # from lasy.laser import Laser as LasyLaser
        raise NotImplementedError("Not implemented yet.")

    def get_r_extent(
        self, simulation_extent: tuple[float, float], num_sigma: float = 3.0
    ) -> float:
        raise NotImplementedError("Not implemented yet.")

    def get_z_extent(self, num_sigma: float = 3.0) -> tuple[float, float]:
        raise NotImplementedError("Not implemented yet.")

    def resolve_laser_energy(self) -> float:
        raise NotImplementedError("Not implemented yet.")

    def resolve_laser_a0(self) -> float:
        raise NotImplementedError("Not implemented yet.")

    def build_laser_profile(self) -> LaserProfile | list[LaserProfile]:
        raise NotImplementedError("Not implemented yet.")


@attrs.define(kw_only=True, slots=False, frozen=True)
class GaussianLaserPulse(_GaussianTemporalLaserPulse):
    """
    Temporal-gaussian laser pulse with a Gaussian transverse profile.
    """

    SUBCLASS: ClassVar[str] = "gaussian"

    def get_r_extent(
        self, simulation_extent: tuple[float, float], num_sigma: float = 3.0
    ) -> float:
        rayleigh_length = pi * self.waist**2 / self.wavelength
        waist_max = self.waist * np.sqrt(
            1
            + max(
                abs(simulation_extent[1] - self.focal_position),
                abs(self.focal_position - simulation_extent[0]),
            )
            ** 2
            / rayleigh_length**2
        )
        return waist_max * num_sigma / 2.0

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

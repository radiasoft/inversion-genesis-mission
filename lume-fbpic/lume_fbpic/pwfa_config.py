"""Config classes for beam-driven (PWFA) simulations: the grid and the particle bunches.

They are frozen attrs classes on `inversion_fbpic`'s `SerializableConfig`, which only supplies
the serialization (YAML, JSON and HDF5) and the type registry; nothing here uses the LWFA
`Simulation` or its laser classes. The plasma is an ordinary `_DensityProfile` (for example `LinearRampFlattop` with
`species=None`, bare electrons).

Two bunch shapes: `FlatTopBunch` (uniform density, `add_particle_bunch`) and `GaussianBunch`
(`add_particle_bunch_gaussian`). A Gaussian bunch is placed by its focus: it is generated at
`zf` and then moved back to where it was `tf` seconds earlier, ignoring space charge.
"""

from __future__ import annotations

from abc import abstractmethod
import typing

import attrs
from scipy.constants import c, e, m_e, pi

from inversion_fbpic.lib.serializable_config import SerializableConfig

_BOUNDARIES = {"z": ("open", "periodic"), "r": ("open", "reflective")}


@attrs.define(kw_only=True, slots=False, frozen=True)
class PWFAGrid(SerializableConfig):
    """The simulation box, time step and diagnostics of a lab-frame PWFA run.

    Args:
        zmin: (float) [m] Left end of the box at the start.
        zmax: (float) [m] Right end of the box at the start.
        nz: (int) Number of grid points along z.
        rmax: (float) [m] Radial size of the box.
        nr: (int) Number of grid points along r.
        nm: (int) |OPTIONAL| Number of azimuthal modes. Defaults to 1 (axisymmetric).
        n_steps: (int) Number of time steps to run.
        dt: (float|None) [s] |OPTIONAL| Time step. None means `(zmax - zmin) / nz / c`, one cell per step.
        write_period: (int) |OPTIONAL| Steps between diagnostic dumps. Defaults to 20.
        save_directory: (str) |OPTIONAL| Diagnostics directory under the working directory. Defaults to "diags".
        z_boundary: (str) |OPTIONAL| "open" or "periodic". Defaults to "open".
        r_boundary: (str) |OPTIONAL| "open" or "reflective". Defaults to "open".
        use_cuda: (bool) |OPTIONAL| Run on a GPU. Defaults to False.
        smoother_passes: (int) |OPTIONAL| Binomial smoothing passes of the charge and currents, along z and r. Defaults to 1 (fbpic's default).
        smoother_compensator: (bool) |OPTIONAL| Whether the smoother is compensated. Defaults to True (fbpic's default).
        random_seed: (int|None) |OPTIONAL| Seed for numpy's random generator before the bunches are drawn. Defaults to None (unset).
        write_plasma: (bool) |OPTIONAL| Whether the diagnostics include the plasma electrons (millions of macroparticles per dump on a large grid). Defaults to True.
    """

    CONFIG_TYPE: typing.ClassVar[str] = "pwfa_grid"
    _CONCRETE_REGISTRY: typing.ClassVar[dict[str, type["PWFAGrid"]]] = {}
    SUBCLASS: typing.ClassVar[str] = "pwfa_grid"

    zmin: float = attrs.field(converter=float)
    zmax: float = attrs.field(converter=float)
    nz: int = attrs.field(converter=int, validator=attrs.validators.gt(0))
    rmax: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    nr: int = attrs.field(converter=int, validator=attrs.validators.gt(0))
    nm: int = attrs.field(default=1, converter=int, validator=attrs.validators.gt(0))
    n_steps: int = attrs.field(converter=int, validator=attrs.validators.gt(0))
    dt: float | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=attrs.validators.optional(attrs.validators.gt(0.0)),
    )
    write_period: int = attrs.field(
        default=20, converter=int, validator=attrs.validators.gt(0)
    )
    save_directory: str = attrs.field(default="diags")
    z_boundary: str = attrs.field(default="open")
    r_boundary: str = attrs.field(default="open")
    use_cuda: bool = attrs.field(default=False)
    smoother_passes: int = attrs.field(
        default=1, converter=int, validator=attrs.validators.ge(0)
    )
    smoother_compensator: bool = attrs.field(default=True)
    random_seed: int | None = attrs.field(
        default=None, converter=attrs.converters.optional(int)
    )
    write_plasma: bool = attrs.field(default=True)

    def __attrs_post_init__(self) -> None:
        if self.zmax <= self.zmin:
            raise ValueError(
                f"zmax ({self.zmax}) must be greater than zmin ({self.zmin})"
            )
        for axis, value in (("z", self.z_boundary), ("r", self.r_boundary)):
            if value not in _BOUNDARIES[axis]:
                raise ValueError(
                    f"{axis}_boundary must be one of {_BOUNDARIES[axis]}, got {value!r}"
                )

    @property
    def boundaries(self) -> dict[str, str]:
        """The `boundaries` argument of fbpic's `Simulation`."""
        return {"z": self.z_boundary, "r": self.r_boundary}

    @property
    def time_step(self) -> float:
        """The time step [s]: `dt`, or one cell of light travel by default."""
        return self.dt if self.dt is not None else (self.zmax - self.zmin) / self.nz / c


@attrs.define(kw_only=True, slots=False, frozen=True)
class _ParticleBunch(SerializableConfig):
    """Base class for the electron bunches of a PWFA run (the driver and any witness).

    Every bunch also has a `charge`, its positive magnitude in coulombs: a field of
    `GaussianBunch`, a property of `FlatTopBunch`.

    Args:
        gamma: (float) Lorentz factor of the bunch.
    """

    CONFIG_TYPE: typing.ClassVar[str] = "particle_bunch"
    _CONCRETE_REGISTRY: typing.ClassVar[dict[str, type["_ParticleBunch"]]] = {}

    gamma: float = attrs.field(converter=float, validator=attrs.validators.gt(1.0))

    @abstractmethod
    def add_to_simulation(self, simulation: typing.Any) -> typing.Any:
        """Add the bunch to a fbpic `Simulation` and return its `Particles` species."""


@attrs.define(kw_only=True, slots=False, frozen=True)
class FlatTopBunch(_ParticleBunch):
    """A cylindrical electron bunch of uniform density, added with fbpic's `add_particle_bunch`.

    Args:
        density: (float) [m^-3] Electron density of the bunch.
        radius: (float) [m] Radius of the bunch.
        zmax: (float) [m] Position of the head (right end).
        zmin: (float) [m] Position of the tail (left end).
        p_nr: (int) |OPTIONAL| Macroparticles per cell along r. Defaults to 2.
        p_nt: (int) |OPTIONAL| Macroparticles per cell along theta. Defaults to 4.
        p_nz: (int) |OPTIONAL| Macroparticles per cell along z. Defaults to 2.
    """

    SUBCLASS: typing.ClassVar[str] = "flat_top_bunch"

    density: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    radius: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    zmax: float = attrs.field(converter=float)
    zmin: float = attrs.field(converter=float)
    p_nr: int = attrs.field(default=2, converter=int, validator=attrs.validators.gt(0))
    p_nt: int = attrs.field(default=4, converter=int, validator=attrs.validators.gt(0))
    p_nz: int = attrs.field(default=2, converter=int, validator=attrs.validators.gt(0))

    def __attrs_post_init__(self) -> None:
        if self.zmax <= self.zmin:
            raise ValueError(
                f"zmax ({self.zmax}) must be greater than zmin ({self.zmin})"
            )

    @property
    def charge(self) -> float:
        """The bunch charge [C], a positive magnitude: density times the cylinder volume."""
        return e * self.density * pi * self.radius**2 * (self.zmax - self.zmin)

    def add_to_simulation(self, simulation: typing.Any) -> typing.Any:
        from fbpic.lpa_utils.bunch import add_particle_bunch

        return add_particle_bunch(
            simulation,
            -e,
            m_e,
            self.gamma,
            self.density,
            self.zmin,
            self.zmax,
            0.0,
            self.radius,
            p_nr=self.p_nr,
            p_nz=self.p_nz,
            p_nt=self.p_nt,
        )


@attrs.define(kw_only=True, slots=False, frozen=True)
class GaussianBunch(_ParticleBunch):
    """A Gaussian electron bunch, added with fbpic's `add_particle_bunch_gaussian`.

    The bunch is drawn with its centre at `zf`, with transverse momenta from `n_emit`, and then
    moved back along its straight-line trajectory to where it was `tf` seconds earlier (no
    space charge). With `tf = 0` it starts at `zf`.

    Args:
        charge: (float) [C] Charge of the bunch, as a positive magnitude.
        sig_r: (float) [m] RMS radius (the RMS size along each of x and y).
        sig_z: (float) [m] RMS length.
        zf: (float) [m] z position of the bunch centre at the time `tf` after the start.
        n_emit: (float) |OPTIONAL| [m] Normalized transverse emittance. Defaults to 0.
        n_macroparticles: (int) |OPTIONAL| Number of macroparticles. Defaults to 1000.
        sig_gamma: (float) |OPTIONAL| Absolute RMS spread of gamma. Defaults to 0.
        symmetrize: (bool) |OPTIONAL| Four-fold symmetrize the bunch (4x the macroparticles). Defaults to False.
        tf: (float) |OPTIONAL| [s] Time after the start at which the bunch is at its focus `zf`. Defaults to 0.
    """

    SUBCLASS: typing.ClassVar[str] = "gaussian_bunch"

    charge: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    sig_r: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    sig_z: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    zf: float = attrs.field(converter=float)
    n_emit: float = attrs.field(
        default=0.0, converter=float, validator=attrs.validators.ge(0.0)
    )
    n_macroparticles: int = attrs.field(
        default=1000, converter=int, validator=attrs.validators.gt(0)
    )
    sig_gamma: float = attrs.field(
        default=0.0, converter=float, validator=attrs.validators.ge(0.0)
    )
    symmetrize: bool = attrs.field(default=False)
    tf: float = attrs.field(default=0.0, converter=float)

    def add_to_simulation(self, simulation: typing.Any) -> typing.Any:
        from fbpic.lpa_utils.bunch import add_particle_bunch_gaussian

        return add_particle_bunch_gaussian(
            simulation,
            -e,
            m_e,
            self.sig_r,
            self.sig_z,
            self.n_emit,
            self.gamma,
            self.sig_gamma,
            self.charge / e,
            self.n_macroparticles,
            tf=self.tf,
            zf=self.zf,
            symmetrize=self.symmetrize,
        )

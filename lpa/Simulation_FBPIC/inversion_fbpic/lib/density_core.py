"""
Attrs-based density profile configuration.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from pathlib import Path
import attrs

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from fbpic.main import Simulation as FBPICSimulation
    from fbpic.particles.particles import Particles

import numpy as np
import numpy.typing as npt
from scipy.constants import e, m_e, m_p
import periodictable as pt

from inversion_fbpic.lib.serializable_config import SerializableConfig

DensityCallable = Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]


def _get_atomic_mass(symbol: str) -> float:
    return pt.elements.symbol(symbol).mass


@attrs.define(kw_only=True, slots=False, frozen=True)
class _DensityProfile(SerializableConfig):
    """Base class for serializable density profile configurations.

    Args:
        nominal_density: (float) [m^-3] Baseline (nominal) plasma density in m^-3 used to scale
            the relative profile returned by `build_density_function` (which is normalized to 1).
        species: (str|None) [str] |OPTIONAL| Species name for the density profile. Should be the one- or two-character code for the species.
            If None, the profile is a bare electron species only: `nominal_density` is the electron density, there are no
            ions and no ionization. This is not supported in a boosted-frame simulation (the ion background is needed there).
            Defaults to Hydrogen.
        ionization: (int|None) [int] |OPTIONAL| Initial ionization level for this species. If None or 0, the plasma is assumed to be initially unionized. If -1, the plasma is assumed to be fully ionized.
            Defaults to unionized (ionization level 0). When `species` is None, only None, 0 (the default), -1 or 1 are accepted and all mean fully ionized (it is stored as 1, so a saved profile loads back).
        p_rmax: (float|None) [m] |OPTIONAL| Maximum radial extent of the density profile. If None, the radial extent is determined by the simulation grid.
        p_nz: (int) Number of macroparticles per gridcell along the longitudinal direction.
        p_nr: (int) Number of macroparticles per gridcell along the radial direction.
        p_nt: (int) Number of macroparticles per gridcell along the angular direction.
        elec_name: (str|None) [str] |OPTIONAL| Name of the electron species when writing diagnostics. If None, the electron particles will not be written to diagnostics.
        elec_select: (dict[str, list[float|None]]|None) [dict] |OPTIONAL| Filters for the electron species when writing diagnostics.
            The keys are the names of the particle attributes to filter on, and the values are the minimum and maximum values for the attribute. If None, no filter is applied.
            Example Python: {"uz": [10.0, None], "z": [-1e-6, 1e-6]} will filter on the uz and z attributes, keeping only particles with uz > 10.0 and z between -1e-6 and 1e-6.
            Example YAML: elec_select: {uz: [10.0, null], z: [-1.0e-6, 1.0e-6]}
            Defaults to None.
        ion_name: (str|None) [str] |OPTIONAL| Name of the ion species when writing diagnostics. If None, the ion particles will not be written to diagnostics.
        ion_select: (dict[str, list[float|None]]|None) [dict] |OPTIONAL| Filters for the ion species when writing diagnostics.
            The keys are the names of the particle attributes to filter on, and the values are the minimum and maximum values for the attribute. If None, no filter is applied.
            Example: {"uz": [10.0, None], "z": [-1e-6, 1e-6]} will filter on the uz and z attributes, keeping only particles with uz > 10.0 and z between -1e-6 and 1e-6.
            Defaults to None.
    """

    # required input by user for all density profiles
    nominal_density: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    p_nz: int = attrs.field(converter=int, validator=attrs.validators.ge(1))
    p_nr: int = attrs.field(converter=int, validator=attrs.validators.ge(1))
    p_nt: int = attrs.field(converter=int, validator=attrs.validators.ge(1))
    species: str | None = attrs.field(default="H")
    ionization: int | None = attrs.field(
        default=0, converter=attrs.converters.optional(int)
    )
    p_rmax: float | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=attrs.validators.optional(attrs.validators.gt(0.0)),
    )
    elec_name: str | None = attrs.field(default=None)
    elec_select: dict[str, list[float | None]] | None = attrs.field(default=None)
    ion_name: str | None = attrs.field(default=None)
    ion_select: dict[str, list[float | None]] | None = attrs.field(default=None)

    # CONFIG_TYPE identifies this domain in the top-level registry held on
    # SerializableConfig. It is set on the domain base class (here) only;
    # concrete subclasses inherit it and must NOT override it.
    CONFIG_TYPE: ClassVar[str] = "density_profile"

    # Per-domain registry: SUBCLASS -> concrete DensityProfile subclass.
    # Concrete subclasses register into this dict automatically by declaring
    # SUBCLASS in their own class body.
    _CONCRETE_REGISTRY: ClassVar[dict[str, type["_DensityProfile"]]] = {}

    @abstractmethod
    def get_z_extent(self) -> tuple[float, float]:
        """
        Get the longitudinal extent of the density profile in meters.

        Returns:
            tuple[float, float]: A tuple of (z_min, z_max) representing the longitudinal extent in meters.
        """

    @abstractmethod
    def get_r_extent(self) -> float | None:
        """
        Get the radial extent of the density profile in meters.

        Returns:
            float|None: The radial extent in meters, or None if not applicable.
        """

    @abstractmethod
    def build_density_function(self) -> DensityCallable:
        """
        Return a callable function that returns the relative density at position (z, r).
        The function takes two arguments:
            z: Array-like of z positions
            r: Array-like of r positions
        and returns an array-like of density values.

        Returns:
            DensityCallable: A callable function that returns the relative density at position (z, r).
        """

    def __attrs_post_init__(self) -> None:
        if self.species is None:
            # Bare electrons: one "level", already ionized.
            if self.ionization not in (None, 0, -1, 1):
                raise ValueError(
                    "species=None means bare electrons only; `ionization` must be None, 0, -1 or 1, "
                    f"got {self.ionization}."
                )
            object.__setattr__(self, "ionization", 1)
            return
        if self.ionization is not None:
            num_ionization_levels = _DensityProfile._get_num_ionization_levels(
                self.species
            )
            if self.ionization < 0:
                object.__setattr__(self, "ionization", num_ionization_levels)
            if num_ionization_levels < self.ionization:
                raise ValueError(
                    f"Species {self.species} has only {num_ionization_levels} ionization levels. Lower `ionization` or provide a larger species."
                )
        else:
            object.__setattr__(self, "ionization", 0)

    @staticmethod
    def _get_num_ionization_levels(species: str) -> int:
        """
        Get the number of ionization levels for a given species.

        Args:
            species: (str) The species name (e.g., "H", "He", "Li", etc.).

        Returns:
            int: The number of ionization levels for the given species.
        """
        from fbpic.particles.elementary_process.ionization.read_atomic_data import (
            read_ionization_energies,
        )

        try:
            ionization_energies = read_ionization_energies(species)
            return len(ionization_energies)
        except ValueError as e:
            raise ValueError(f"Species {species} was not recognized by FBPIC.") from e
        except Exception as e:
            raise ValueError(
                f"Error getting ionization levels for species {species}: {e}"
            ) from e

    def plot_z_profile(
        self,
        ax: plt.Axes,
        num: int = 100,
        r0: float = 0.0,
        z_scale: float = 1.0,
        density_scale: float = 1.0,
        z_padding_fraction: float = 0.0,
        neutral_density: bool = False,
        **kwargs: Any,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """
        Plot the on-axis (z) density profile on the given axes.

        Args:
            ax: (matplotlib.axes.Axes object) The axes to plot on.
            num: (int) The number of points to plot.
            r0: (float) The r position to plot at.
            z_scale: (float) The scale to apply to the z axis.
                Defaults to 1.0.
            density_scale: (float) The scale to apply to the density axis.
                Defaults to 1.0.
            z_padding_fraction: (float) Fractional padding applied to the z
                extent on each side.
            neutral_density: (bool) Whether to plot the neutral density or the
                fully ionized electron density.
            **kwargs: Additional arguments to pass to the plot function.

        Returns:
            tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]: Tuple of (z positions [m], density values [m^-3]).
        """
        z_min, z_max = self.get_z_extent()
        z_pad = z_padding_fraction * (z_max - z_min)
        z_min -= z_pad
        z_max += z_pad
        zs = np.linspace(z_min, z_max, num)
        dens = (
            self.build_density_function()(zs, r0 * np.ones_like(zs))
            * self.nominal_density
            / (
                1.0
                if not neutral_density
                else self._get_num_ionization_levels(self.species)
            )
        )
        ax.plot(zs * z_scale, dens * density_scale, **kwargs)
        return zs, dens

    def plot_r_profile(
        self,
        ax: plt.Axes,
        num: int = 100,
        z0: float = 0.0,
        r_scale: float = 1.0,
        density_scale: float = 1.0,
        r_max: float = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot the density profile on the given axes.

        Args:
            ax: (matplotlib.axes.Axes object) The axes to plot on.
            num: (int) The number of points to plot.
            z0: (float) The z position to plot at.
            r_scale: (float) The scale to apply to the r axis.
                Defaults to 1.0.
            density_scale: (float) The scale to apply to the density axis.
                Defaults to 1.0.
            r_max: (float) The maximum r position to plot at.
                Defaults to None.
            **kwargs: Additional arguments to pass to the plot function.

        Returns:
            None.
        """

        r_extent = self.get_r_extent()
        if r_extent is None and r_max is None:
            raise ValueError(
                "r_extent is not set for this profile and r_max is not provided."
            )
        if r_max is not None:
            rs = np.linspace(0.0, r_max, num)
        else:
            rs = np.linspace(0.0, r_extent, num)

        dens = (
            self.build_density_function()(z0 * np.ones_like(rs), rs)
            * self.nominal_density
        )
        ax.plot(rs * r_scale, dens * density_scale, **kwargs)

    def plot(
        self,
        *,
        neutral_density: bool = False,
        ax: plt.Axes | None = None,
        r_max: float | None = None,
        num: int = 600,
        z_padding_fraction: float = 0.0,
        default_r_max: float = 50e-6,
        output_path: Path | str | None = None,
        show: bool = False,
        label: str | None = None,
    ) -> plt.Figure:
        """Plot the density profile as an on-axis lineout and 2D z-r map.

        Args:
            neutral_density: (bool) Whether to plot the neutral density or the fully ionized electron density.
            ax: (matplotlib.axes.Axes|None) If provided, the on-axis lineout is
                *also* drawn on this external axes (useful for combined overlay
                figures). The method still creates its own figure.
            r_max: (float|None) [m] Maximum radial extent for the 2D map. If
                *None*, resolved from ``get_r_extent()`` / ``p_rmax`` /
                *default_r_max*.
            num: (int) Grid resolution along each axis.
            z_padding_fraction: (float) Fractional padding applied to the z
                extent on each side.
            default_r_max: (float) [m] Fallback radial extent when the profile
                does not specify one.
            output_path: (Path|str|None) If given, the figure is saved to this
                path.
            show: (bool) Whether to call ``plt.show()``.
            label: (str|None) Label used on the external *ax* lineout. Defaults
                to ``self.SUBCLASS (self.species)``.

        Returns:
            pyplot.Figure: The created matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        r_lim = r_max
        if r_lim is None:
            extent = self.get_r_extent()
            if extent is not None:
                r_lim = extent
            elif self.p_rmax is not None:
                r_lim = self.p_rmax
            else:
                r_lim = default_r_max

        dens_func = self.build_density_function()
        z_profile_kwargs = {
            "num": num,
            "z_scale": 1e3,
            "density_scale": 1e-6,
            "z_padding_fraction": z_padding_fraction,
            "neutral_density": neutral_density,
        }

        if ax is not None:
            self.plot_z_profile(
                ax,
                lw=1.5,
                label=(
                    label if label is not None else f"{self.SUBCLASS} ({self.species})"
                ),
                **z_profile_kwargs,
            )

        fig, (ax_z, ax_2d) = plt.subplots(1, 2, figsize=(12, 4.5))

        z_arr, _ = self.plot_z_profile(ax_z, color="C0", lw=1.5, **z_profile_kwargs)
        z_min, z_max = z_arr[0], z_arr[-1]
        ax_z.set_xlabel("z (mm)")
        ax_z.set_ylabel(
            f"{'Neutral' if neutral_density else 'Electron'}"
            + r" Density ($\mathrm{cm}^{-3}$)"
        )
        ax_z.set_title(f"On-axis profile (r = 0)\n{self.SUBCLASS} ({self.species})")
        ax_z.grid(True, alpha=0.3)

        r_arr = np.linspace(0.0, r_lim, num)
        z_mesh, r_mesh = np.meshgrid(z_arr, r_arr, indexing="xy")
        density_2d = dens_func(z_mesh, r_mesh) * self.nominal_density

        im = ax_2d.imshow(
            density_2d * 1e-6,
            origin="lower",
            aspect="auto",
            extent=(z_min * 1e3, z_max * 1e3, 0.0, r_lim * 1e6),
        )
        ax_2d.set_xlabel("z (mm)")
        ax_2d.set_ylabel(r"r ($\mu$m)")
        ax_2d.set_title("2D z-r density")
        cbar = fig.colorbar(im, ax=ax_2d, fraction=0.046, pad=0.04)
        cbar.set_label(
            f"{'Neutral' if neutral_density else 'Electron'}"
            + r" Density ($\mathrm{cm}^{-3}$)"
        )

        fig.suptitle(self.SUBCLASS, fontsize=11)
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

    def get_plasma_wavelength(
        self, z: float | None = None, r: float | None = None
    ) -> float:
        """
        Get the plasma wavelength at a given position. If no position is provided,
        return the plasma wavelength according to the "nominal_density" parameter.

        Args:
            z: (float|None) The z position to evaluate the plasma wavelength at.
                If None, the plasma wavelength is evaluated at z=0.0.
            r: (float|None) The r position to evaluate the plasma wavelength at.
                If None, the plasma wavelength is evaluated at r=0.0.

        Returns:
            float: The plasma wavelength at the given position. If the plasma density is zero at the given position, return infinity.
        """

        if z is None and r is None:
            if np.isclose(self.nominal_density, 0.0):
                return float("inf")
            return float(3.3e7 / np.sqrt(self.nominal_density))

        z_eval = 0.0 if z is None else z
        r_eval = 0.0 if r is None else r
        dens = self.build_density_function()
        rho_zr = self.nominal_density * dens(z_eval, r_eval)
        if np.isclose(rho_zr, 0.0):
            return float("inf")
        return float(3.3e7 / np.sqrt(rho_zr))

    def add_to_simulation(
        self, simulation: FBPICSimulation, is_boosted: bool
    ) -> tuple[Particles, Particles | None]:
        """
        Add gas to the simulation.

        Args:
            simulation: (FBPICSimulation) The simulation to add the density profile to.
            is_boosted: (bool) Whether the simulation is boosted. A bare-electron profile (species None) is rejected if True.

        Returns:
            tuple[Particles, Particles | None]: A tuple containing the added electron species and the added ion species if it exists, otherwise None (e.g., electron-only).
        """
        species = self.species
        if species is None:
            if is_boosted:
                raise ValueError(
                    "species=None (bare electrons) is not supported in a boosted-frame "
                    "simulation; provide a species so an ion background is added."
                )
            num_ionization_levels = 1
        else:
            num_ionization_levels = _DensityProfile._get_num_ionization_levels(species)

        elec = simulation.add_new_species(
            q=-e,
            m=m_e,
            n=self.nominal_density * (float(self.ionization) / num_ionization_levels),
            dens_func=self.build_density_function(),
            p_nz=self.p_nz,
            p_nr=self.p_nr,
            p_nt=self.p_nt,
            boost_positions_in_dens_func=is_boosted,
        )

        ions = None
        if species is not None:
            ions = simulation.add_new_species(
                q=e * self.ionization,
                m=m_p * _get_atomic_mass(species),
                n=self.nominal_density / num_ionization_levels,
                dens_func=self.build_density_function(),
                p_nz=self.p_nz,
                p_nr=self.p_nr,
                p_nt=self.p_nt,
                boost_positions_in_dens_func=is_boosted,
            )
            if self.ionization < num_ionization_levels:
                ions.make_ionizable(
                    species, target_species=elec, level_start=self.ionization
                )

        return elec, ions


@attrs.define(kw_only=True, slots=False, frozen=True)
class _DensityModifier(SerializableConfig):
    """
    Base class for serializable density modifier configurations.
    """

    # CONFIG_TYPE identifies this domain in the top-level registry held on
    # SerializableConfig. It is set on the domain base class (here) only;
    # concrete subclasses inherit it and must NOT override it.
    CONFIG_TYPE: ClassVar[str] = "density_modifier"

    # Per-domain registry: SUBCLASS -> concrete DensityProfile subclass.
    # Concrete subclasses register into this dict automatically by declaring
    # SUBCLASS in their own class body.
    _CONCRETE_REGISTRY: ClassVar[dict[str, type["_DensityModifier"]]] = {}

    @abstractmethod
    def modify_density_function(
        self, density_function: DensityCallable
    ) -> DensityCallable:
        """
        Modify the density function.

        Args:
            density_function: (DensityCallable) The density function to modify.

        Returns:
            DensityCallable: The modified density function.
        """

    @abstractmethod
    def get_z_extent(self) -> tuple[float, float]:
        """
        Get the longitudinal extent of the density profile in meters.

        Returns:
            tuple[float, float]: A tuple of (z_min, z_max) representing the longitudinal extent in meters.
        """

    @abstractmethod
    def get_r_extent(self) -> float | None:
        """
        Get the radial extent of the density profile in meters.

        Returns:
            float|None: The radial extent in meters, or None if not applicable.
        """

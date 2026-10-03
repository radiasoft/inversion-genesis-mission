"""FBPICSimulator: wraps `inversion_fbpic.lib.simulation.Simulation` as a `lume.base.Base`
subclass, providing `archive()`/`load_archive()` (config only, no results) and re-runnable
set/reset semantics for use by `LUMEFBPICModel`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.constants import c, e, m_e

from lume.base import Base

# Import submodules to populate the SerializableConfig subclass registries (config_type ->
# class, subclass -> class) before any from_yaml()/from_dict() dispatch is attempted.
from inversion_fbpic import lib as _inversion_fbpic_lib  # noqa: F401
# lume_fbpic's own density profiles register the same way; a fresh process (a batch worker,
# for example) must import them to load an archive that uses one.
from lume_fbpic import density_profiles as _lume_fbpic_density_profiles  # noqa: F401
from inversion_fbpic.lib.serializable_config import SerializableConfig
from inversion_fbpic.lib.simulation import (
    Simulation as FBPICSimulation,
    SimulationHyperparameters,
)
from inversion_fbpic.lib.density_core import _DensityProfile
from inversion_fbpic.lib.laser import _LaserPulse

from openpmd_viewer.addons import LpaDiagnostics

try:
    from beamphysics import ParticleGroup
    from beamphysics.writers import pmd_init
except ImportError:
    from pmd_beamphysics import ParticleGroup
    from pmd_beamphysics.writers import pmd_init


# Electron rest energy in eV. Used to convert fbpic's normalized momentum (u = gamma*beta)
# into openPMD-beamphysics' px/py/pz convention, which is in eV/c (numerically pc in eV),
# not SI kg*m/s. Verified directly against the installed pmd_beamphysics source
# (ParticleGroup.energy = sqrt(px**2+py**2+pz**2+mass**2), mass in eV) and a synthetic
# uz=100 check (expected ~50.6 MeV kinetic energy, confirmed numerically).
_MC2_EV = m_e * c**2 / e


class FBPICSimulator(Base):
    """Wraps `inversion_fbpic.lib.simulation.Simulation` for use as a LUMEModel simulator.

    Holds the current hyperparameters/laser/density-profile config objects and rebuilds a
    fresh `inversion_fbpic.lib.simulation.Simulation` on every `run()` call -- matches how
    fbpic simulations actually work: a parameter change requires a full PIC rebuild, not an
    incremental update (unlike e.g. Cheetah's cheap `track()`).

    Parameters
    ----------
    hyparams : SimulationHyperparameters
    laser : _LaserPulse
    densities : list[_DensityProfile]
    target_species : str
        Name of the electron species (matching a density profile's `elec_name`) to read back
        as `final_particles` after a run.
    working_directory : str | Path, optional
        Directory fbpic writes diagnostics under. Defaults to the current working directory.
    """

    def __init__(
        self,
        hyparams: SimulationHyperparameters,
        laser: _LaserPulse,
        densities: list[_DensityProfile],
        *,
        target_species: str,
        working_directory: str | Path | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(verbose=verbose)
        self.hyparams = hyparams
        self.laser = laser
        self.densities = list(densities)
        self.target_species = target_species
        self.working_directory = Path(working_directory) if working_directory else Path.cwd()
        self.final_particles: ParticleGroup | None = None
        self.stats: dict[str, Any] = {}
        # (final_particles, selection key, descriptor) -- see actions._descriptor
        self._descriptor_cache: tuple[Any, tuple, dict[str, float]] | None = None

    @classmethod
    def from_archive(
        cls,
        h5,
        *,
        target_species: str | None = None,
        working_directory: str | Path | None = None,
    ) -> "FBPICSimulator":
        """Build a simulator from an archive written by `archive()`.

        `target_species` defaults to the one stored in the archive; archives written before it
        was stored need it passed explicitly.
        """
        contents = _read_archive(h5)
        species = target_species or contents["target_species"]
        if species is None:
            raise ValueError("The archive has no target_species; pass target_species=...")
        simulator = cls(
            contents["hyparams"],
            contents["laser"],
            contents["densities"],
            target_species=species,
            working_directory=working_directory,
        )
        simulator.final_particles = contents["final_particles"]
        simulator.stats = contents["stats"]
        return simulator

    def configure(self) -> None:
        """Validate that the current config set is complete."""
        if self.hyparams is None:
            raise ValueError("hyparams is required.")
        if self.laser is None:
            raise ValueError("laser is required.")
        if not self.densities:
            raise ValueError("At least one density profile is required.")
        self.configured = True

    def run(self) -> None:
        """Build a fresh `Simulation` from the current config objects and run it.

        Always passes `skip_if_hashed=False` on both `setup_simulation()` and
        `run_simulation()` -- a `run()` call here means "run this now", not "skip if a
        previous run in this working directory already matches this config" (that
        skip-if-hashed default is meant for repeated script invocations, not a live
        model's `set()`).
        """
        if not self.configured:
            self.vprint("not configured to run")
            return
        self.run_simulation()
        self.load_results()

    def run_simulation(self) -> FBPICSimulation:
        """Build a fresh `Simulation` and run it, WITHOUT reading any output back.

        This is the part of `run()` every MPI rank executes (for example under MPI);
        `load_results()` reads the diagnostics afterward, once, wherever they are visible.
        """
        simulation = FBPICSimulation(elements=[self.hyparams, self.laser, *self.densities])
        simulation.setup_simulation(
            working_directory=self.working_directory, skip_if_hashed=False
        )
        simulation.run_simulation(skip_if_hashed=False)
        return simulation

    def _update_output(self, diags_path: Path) -> None:
        """Cache `final_particles` and scalar `stats` from the diagnostics in `diags_path`."""
        ts = LpaDiagnostics(str(diags_path), check_all_files=True)
        iteration = ts.iterations[-1]
        x, y, z, ux, uy, uz, w = ts.get_particle(
            ["x", "y", "z", "ux", "uy", "uz", "w"],
            iteration=iteration,
            species=self.target_species,
            plot=False,
        )
        self.final_particles = ParticleGroup(
            data={
                "x": x,
                "y": y,
                "z": z,
                "px": ux * _MC2_EV,
                "py": uy * _MC2_EV,
                "pz": uz * _MC2_EV,
                "t": np.zeros_like(x),
                "status": np.ones_like(x, dtype=int),
                "weight": w * e,
                "species": "electron",
            }
        )
        if len(w):
            gamma = np.sqrt(1.0 + ux**2 + uy**2 + uz**2)
            energy_mev = (gamma - 1.0) * m_e * c**2 / e / 1e6
            energy_mean = float(np.average(energy_mev, weights=w))
            energy_std = float(
                np.sqrt(np.average((energy_mev - energy_mean) ** 2, weights=w))
            )
            charge_pc = float(np.sum(w)) * e * 1e12
        else:
            energy_mean = energy_std = charge_pc = 0.0
        self.stats = {
            "charge_pc": charge_pc,
            "energy_mean_mev": energy_mean,
            "energy_std_mev": energy_std,
        }

    def reset(self) -> None:
        """Clear cached output/state.

        Does not eagerly rebuild/rerun -- a full fbpic run is far too expensive to trigger
        implicitly from `reset()` (unlike Cheetah's cheap eager re-`track()`). Matches
        `Impact.reset()`'s lightweight semantics: call `configure()` (or `set()`) again to
        actually run.
        """
        self.final_particles = None
        self.stats = {}
        self.configured = False
        self.finished = False

    def fingerprint(self) -> str:
        """Stable hash of the current config.

        Reuses `inversion_fbpic.lib.simulation.Simulation.config_hash()` (built fresh here
        since no `Simulation` object persists between runs) rather than `Base`'s default
        `tools.fingerprint(self.input)`, which assumes a dict-shaped `self.input` this class
        doesn't populate.
        """
        simulation = FBPICSimulation(elements=[self.hyparams, self.laser, *self.densities])
        return simulation.config_hash()

    def archive(self, h5=None, *, save_final_particles: bool = False):
        """Archive the current config (hyparams/laser/densities) into one HDF5 file/group.

        Stores each config object's `to_yaml()` string as an HDF5 attribute -- the same
        data already written to `cfgs/*.yaml` by the demo scripts, just collected into one
        portable file. The time-series diagnostics `fbpic` writes under `diags/` are never
        archived (they persist separately on disk). With `save_final_particles=True` the final
        bunch is stored too, as a `ParticleGroup` in the openPMD-beamphysics format other lume
        libraries read (the archive file itself opens with `ParticleGroup(h5=<path>)`),
        together with `stats`.

        Parameters
        ----------
        h5 : str, Path, or h5py.File, optional
            Destination. If None, a fingerprint-based filename is used.
        save_final_particles : bool, optional
            Also store `final_particles` (group `final_particles`) and `stats` (group `stats`).
            Raises `ValueError` if there are no final particles yet (nothing has run, or
            `load_results()` has not been called). Defaults to False.

        Returns
        -------
        The h5 argument (or generated filename) passed in.
        """
        if h5 is None:
            h5 = f"lume_fbpic_{self.fingerprint()}.h5"

        if isinstance(h5, (str, Path)):
            new_h5file = True
            g = h5py.File(h5, "w")
        else:
            new_h5file = False
            g = h5

        g.attrs["dataType"] = "lume-fbpic"
        g.attrs["software"] = "lume-fbpic"
        g.attrs["target_species"] = self.target_species

        if save_final_particles and self.final_particles is None:
            raise ValueError(
                "save_final_particles=True but there are no final particles; run the "
                "simulation or call load_results() first."
            )

        g.create_group("hyparams").attrs["yaml"] = self.hyparams.to_yaml()
        g.create_group("laser").attrs["yaml"] = self.laser.to_yaml()

        densities_group = g.create_group("densities")
        densities_group.attrs["count"] = len(self.densities)
        for i, density in enumerate(self.densities):
            densities_group.create_group(str(i)).attrs["yaml"] = density.to_yaml()

        if save_final_particles:
            # openPMD root attributes pointing at the `final_particles` group make the archive
            # itself readable as a particle file: `ParticleGroup(h5=<archive path>)`.
            pmd_init(g, basePath="/", particlesPath="final_particles")
            self.final_particles.write(g.create_group("final_particles"))
            stats_group = g.create_group("stats")
            for name, value in self.stats.items():
                stats_group.attrs[name] = value

        if new_h5file:
            g.close()
        return h5

    def load_archive(self, h5, configure: bool = True) -> None:
        """Restore hyparams/laser/densities from an archive written by `archive()`.

        If the archive holds the final particles (`save_final_particles=True`), they and
        `stats` are restored as well; otherwise both are cleared. `configured`/`finished` are
        left False afterward ("must reconfigure to run again"), matching
        `Impact.load_archive()`'s convention, before optionally calling `configure()`.

        Parameters
        ----------
        h5 : str, Path, or h5py.File
        configure : bool
            Whether to call `self.configure()` after loading. Defaults to True.
        """
        contents = _read_archive(h5)
        self.hyparams = contents["hyparams"]
        self.laser = contents["laser"]
        self.densities = contents["densities"]
        if contents["target_species"] is not None:
            self.target_species = contents["target_species"]

        self.final_particles = contents["final_particles"]
        self.stats = contents["stats"]
        self.configured = False
        self.finished = False
        self.vprint("Loaded from archive. Note: must reconfigure to run again.")

        if configure:
            self.configure()

    def load_results(self, working_directory: str | Path | None = None) -> None:
        """Read `final_particles` and `stats` from a finished run's diagnostics.

        Reads `<working_directory>/<save_directory>/hdf5` (default: this simulator's own
        `working_directory`), for runs executed elsewhere -- for example a batch job written by
        a batch job -- or by `run()` itself.
        """
        directory = Path(working_directory) if working_directory else self.working_directory
        self._update_output(directory / self.hyparams.save_directory / "hdf5")
        self.finished = True


def _read_archive(h5) -> dict[str, Any]:
    """Return the contents of an `archive()` file: `hyparams`, `laser`, `densities`,
    `target_species` (None if not stored), and `final_particles` (None) / `stats` ({}) unless
    the archive was written with `save_final_particles=True`."""
    if isinstance(h5, (str, Path)):
        with h5py.File(h5, "r") as g:
            return _read_archive(g)
    count = int(h5["densities"].attrs["count"])
    target_species = h5.attrs.get("target_species")
    final_particles = None
    stats: dict[str, Any] = {}
    if "final_particles" in h5:
        final_particles = ParticleGroup(h5=h5)
        stats = {name: float(value) for name, value in h5["stats"].attrs.items()}
    return {
        "hyparams": SerializableConfig.from_yaml(h5["hyparams"].attrs["yaml"]),
        "laser": SerializableConfig.from_yaml(h5["laser"].attrs["yaml"]),
        "densities": [
            SerializableConfig.from_yaml(h5["densities"][str(i)].attrs["yaml"])
            for i in range(count)
        ],
        "target_species": None if target_species is None else str(target_species),
        "final_particles": final_particles,
        "stats": stats,
    }

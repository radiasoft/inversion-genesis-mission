"""Simulators for `LUMEFBPICModel`: `BaseSimulator` and its subclasses `FBPICSimulator` (LWFA) and
`PWFASimulator` (beam-driven PWFA).

`BaseSimulator` is the abstract base of the simulators. It owns what does not depend on the kind
of simulation: the lifecycle (`run()`, `reset()`), reading `final_particles` and `stats` back from
fbpic's openPMD output, and the HDF5 archive. A subclass supplies its config
(`config()`/`set_config()`/`from_config()`), the validation and the run itself.
`FBPICSimulator` wraps `inversion_fbpic.lib.simulation.Simulation` for LWFA-type simulations
(hyperparameters, laser, density profiles); `PWFASimulator` builds an fbpic `Simulation` directly,
from a grid, a plasma density profile and electron bunches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from scipy.constants import c, e, m_e
import attrs
import h5py
import hashlib
import json
import numpy
import typing

# Config classes register themselves in `SerializableConfig` when their module is imported, which
# is how `from_hdf5()` finds the class of a stored config. The `inversion_fbpic.lib` ones are all
# imported by the `inversion_fbpic.lib` package itself; lume_fbpic's own density profiles are
# imported here, so that a fresh process can load an archive that uses one.
import lume_fbpic.density_profiles  # noqa: F401
from lume_fbpic import archive_config
from inversion_fbpic.lib.simulation import (
    Simulation as FBPICSimulation,
    SimulationHyperparameters,
)
from inversion_fbpic.lib.density_core import _DensityProfile
from inversion_fbpic.lib.laser import _LaserPulse
from lume_fbpic.pwfa_config import PWFAGrid, _ParticleBunch

from openpmd_viewer.addons import LpaDiagnostics

from beamphysics import ParticleGroup
from beamphysics.writers import pmd_init


# Electron rest energy in eV. Used to convert fbpic's normalized momentum (u = gamma*beta)
# into openPMD-beamphysics' px/py/pz convention, which is in eV/c (numerically pc in eV),
# not SI kg*m/s. Verified directly against the installed beamphysics source
# (ParticleGroup.energy = sqrt(px**2+py**2+pz**2+mass**2), mass in eV) and a synthetic
# uz=100 check (expected ~50.6 MeV kinetic energy, confirmed numerically).
_MC2_EV = m_e * c**2 / e

# Groups of an archive that are not config: the model's actions, the input files, and the results.
_RESERVED_GROUPS = {"actions", "final_particles", "inputs", "stats"}


class BaseSimulator(ABC):
    """What every lume-fbpic simulator shares, whatever it simulates.

    A subclass holds its config as SerializableConfig objects and provides `config()`,
    `set_config()`, `from_config()`, `configure()`, `diagnostics_directory()` and
    `run_simulation()`. `configure()` calls `_prepare_inputs()`. It sets `CONFIG_KIND`, the name written to its archives; the subclass is
    registered under it, which is how `BaseSimulator.from_archive()` finds the right class.

    Parameters
    ----------
    target_species : str
        Name of the electron species to read back as `final_particles` after a run.
    working_directory : str | Path, optional
        Directory fbpic writes diagnostics under. Defaults to the current working directory.
    """

    CONFIG_KIND: str = ""
    _registry: dict[str, type["BaseSimulator"]] = {}

    def __init__(
        self,
        *,
        target_species: str,
        working_directory: str | Path | None = None,
    ) -> None:
        self.configured = False
        self.finished = False
        self.target_species = target_species
        self.working_directory = (
            Path(working_directory) if working_directory else Path.cwd()
        )
        self.final_particles: ParticleGroup | None = None
        self.stats: dict[str, typing.Any] = {}
        # (final_particles, selection key, descriptor) -- see actions._descriptor
        self._descriptor_cache: tuple[typing.Any, tuple, dict[str, float]] | None = None
        # The config, final particles and stats `reset()` restores; see `_remember_state()`.
        self._initial_state: (
            tuple[dict[str, typing.Any], ParticleGroup | None, dict] | None
        ) = None
        # Input files embedded in the archive this was loaded from, by (config, index, field);
        # see `archive_config`.
        self._embedded: dict[tuple[str, int | None, str], archive_config.InputFile] = {}

    def __init_subclass__(cls, **kwargs: typing.Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.CONFIG_KIND:
            BaseSimulator._registry[cls.CONFIG_KIND] = cls

    def archive(
        self,
        h5=None,
        *,
        save_final_particles: bool = False,
        input_dirs: str | Path | typing.Sequence[str | Path] | None = None,
    ):
        """Archive the current config into one HDF5 file/group.

        Each config object (see `config()`) is stored as a group, written by
        `SerializableConfig.to_hdf5()`; a list of config objects is a group with one child per
        object. No path is stored. The output paths (`save_directory`, `lasy_file`) are stored as
        the basename of their default (a loaded config has the default), and an input file as its
        basename and md5, in the `inputs` group: the file
        is found, to load the archive, in `input_dirs`. An input file that is not in `input_dirs`
        is embedded in the archive instead (with a warning over 10 MB), and a loaded archive uses
        the embedded copy. The time-series diagnostics `fbpic` writes under `diags/` are never
        archived (they persist separately on disk). With `save_final_particles=True` the final
        bunch is stored too, as a `ParticleGroup` in the openPMD-beamphysics format other lume
        libraries read (the archive file itself opens with `ParticleGroup(h5=<path>)`), together
        with `stats`.

        Parameters
        ----------
        h5 : str, Path, or h5py.File, optional
            Destination. If None, a fingerprint-based filename is used.
        save_final_particles : bool, optional
            Also store `final_particles` (group `final_particles`) and `stats` (group `stats`).
            Raises `ValueError` if there are no final particles yet (nothing has run, or
            `load_results()` has not been called). Defaults to False.
        input_dirs : str, Path or list of them, optional
            Directories that hold the input files, by basename. Defaults to the current
            directory. An input file in one of them with another md5 than the one the config
            names gets a warning, and is referred to, not embedded.

        Returns
        -------
        The h5 argument (or generated filename) passed in.

        Raises
        ------
        FileNotFoundError
            If an input file named by the config does not exist: the simulation could not run.
        """
        if save_final_particles and self.final_particles is None:
            raise ValueError(
                "save_final_particles=True but there are no final particles; run the "
                "simulation or call load_results() first."
            )
        config = self.config()
        refs = archive_config.collect_inputs(config, self._embedded)
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
        g.attrs["config_kind"] = self.CONFIG_KIND
        g.attrs["target_species"] = self.target_species

        for name, value in config.items():
            if isinstance(value, (list, tuple)):
                group = g.create_group(name)
                group.attrs["list"] = True
                for i, item in enumerate(value):
                    archive_config.write_config(group, str(i), item)
            else:
                archive_config.write_config(g, name, value)
        if refs:
            archive_config.store_inputs(g.create_group("inputs"), refs, input_dirs)

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

    @abstractmethod
    def config(self) -> dict[str, typing.Any]:
        """The config, by name: each value a SerializableConfig or a list of them. A new dict
        (and new lists) on every call, so it can be kept and later given to `set_config()`.
        """

    @abstractmethod
    def configure(self) -> None:
        """Validate that the current config set is complete."""

    @abstractmethod
    def diagnostics_directory(self, working_directory: Path) -> Path:
        """The openPMD directory a run executed under `working_directory` writes."""

    def fingerprint(self) -> str:
        """Stable hash of the current config: the same on any machine, and for any
        `save_directory`.

        It is of the config without paths (see `archive()`) and the md5 of each input file.
        """
        configs = {
            name: [
                {
                    key: value
                    for key, value in archive_config.normalized(item).to_dict().items()
                    if key != "git_hash"
                }
                for item in (value if isinstance(value, (list, tuple)) else [value])
            ]
            for name, value in self.config().items()
        }
        inputs = [
            (ref.key, ref.file.md5)
            for ref in archive_config.collect_inputs(self.config(), self._embedded)
        ]
        text = json.dumps([configs, inputs], sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @classmethod
    def from_archive(
        cls,
        h5,
        *,
        working_directory: str | Path | None = None,
        stats: dict[str, typing.Any] | None = None,
        input_dirs: str | Path | typing.Sequence[str | Path] | None = None,
    ) -> "BaseSimulator":
        """Build a simulator from an archive written by `archive()`.

        The class is the one registered for the archive's `config_kind`. Called on a subclass, it
        raises if the archive is of another kind. `stats` is used as the `stats` of an archive that
        holds no final particles (which then has none of its own).

        An input file the archive refers to is looked for by basename in `input_dirs` (default
        the current directory); one with another md5 than at `archive()` is used with a warning.
        An input file embedded in the archive is used as it is, and is written to
        `working_directory/inputs/` for the config to read it.

        What is loaded is the state `reset()` restores.

        Raises:
            FileNotFoundError: If a referenced input file is not in `input_dirs`.
        """
        contents = _read_archive(
            h5,
            input_dirs,
            Path(working_directory) if working_directory else Path.cwd(),
        )
        kind = contents["config_kind"]
        if kind not in BaseSimulator._registry:
            raise ValueError(f"The archive is of an unknown kind {kind!r}")
        simulator_class = BaseSimulator._registry[kind]
        if cls is not BaseSimulator and not issubclass(simulator_class, cls):
            raise ValueError(
                f"The archive is of kind {kind!r}, which {cls.__name__} cannot load"
            )
        simulator = simulator_class.from_config(
            contents["config"],
            target_species=contents["target_species"],
            working_directory=working_directory,
        )
        simulator.final_particles = contents["final_particles"]
        simulator.stats = contents["stats"]
        simulator._embedded = contents["embedded"]
        if simulator.final_particles is None and stats is not None:
            simulator.stats = dict(stats)
        simulator._remember_state()
        return simulator

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: dict[str, typing.Any],
        *,
        target_species: str,
        working_directory: str | Path | None = None,
    ) -> "BaseSimulator":
        """Build a simulator from a `config()`-shaped dict."""

    def load_results(self, working_directory: str | Path | None = None) -> None:
        """Read `final_particles` and `stats` from a finished run's diagnostics.

        Reads the directory `diagnostics_directory()` names for `working_directory` (default:
        this simulator's own), for runs executed elsewhere -- for example a batch job -- or by
        `run()` itself.
        """
        directory = (
            Path(working_directory) if working_directory else self.working_directory
        )
        self._update_output(self.diagnostics_directory(directory))
        self.finished = True

    def reset(self) -> None:
        """Restore the config, final particles and stats the simulator started with.

        That is the state when it was constructed or, for one loaded from an archive, the
        archived state. Nothing is run -- a full fbpic run is far too expensive to trigger
        implicitly from `reset()` (unlike Cheetah's cheap eager re-`track()`) -- so `finished`
        is False afterward; `configured` is left as it is.

        Raises:
            RuntimeError: If the subclass never recorded a starting state (`_remember_state()`).
        """
        if self._initial_state is None:
            raise RuntimeError(
                "the simulator has no starting state; call _remember_state()"
            )
        config, final_particles, stats = self._initial_state
        self.set_config(config)
        self.final_particles = final_particles
        self.stats = dict(stats)
        self.finished = False

    def run(self) -> None:
        """Run the simulation from the current config, then read the results back.

        A `run()` call means "run this now", not "skip if a previous run in this working
        directory already matches this config" (that skip-if-hashed default is meant for
        repeated script invocations, not a live model's `set()`).
        """
        if not self.configured:
            return
        self.run_simulation()
        self.load_results()

    @abstractmethod
    def run_simulation(self) -> typing.Any:
        """Run the simulation WITHOUT reading any output back.

        This is the part of `run()` every MPI rank executes; `load_results()` reads the
        diagnostics afterward, once, wherever they are visible.
        """

    @abstractmethod
    def set_config(self, config: dict[str, typing.Any]) -> None:
        """Replace the config with a `config()`-shaped dict."""

    def _prepare_inputs(self) -> None:
        """Write the embedded input files under `working_directory/inputs/` and point the config
        at them, for those not already there. `configure()` calls it, so that a run in another
        working directory than the one the archive was loaded in finds them."""
        if not self._embedded:
            return
        config = self.config()
        changed = False
        for (name, index, field), file in self._embedded.items():
            item = config[name] if index is None else config[name][index]
            target = file.write_to(self.working_directory / "inputs")
            if Path(getattr(item, field)) == target:
                continue
            item = attrs.evolve(item, **{field: target})
            if index is None:
                config[name] = item
            else:
                config[name][index] = item
            changed = True
        if changed:
            self.set_config(config)

    def _remember_state(self) -> None:
        """Record the current config, final particles and stats as the state `reset()` restores.

        A subclass calls it at the end of its `__init__`; `from_archive()` calls it once the
        archive is loaded.
        """
        self._initial_state = (self.config(), self.final_particles, dict(self.stats))

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
                "t": numpy.zeros_like(x),
                "status": numpy.ones_like(x, dtype=int),
                "weight": w * e,
                "species": "electron",
            }
        )
        if len(w):
            gamma = numpy.sqrt(1.0 + ux**2 + uy**2 + uz**2)
            energy_mev = (gamma - 1.0) * m_e * c**2 / e / 1e6
            energy_mean = float(numpy.average(energy_mev, weights=w))
            energy_std = float(
                numpy.sqrt(numpy.average((energy_mev - energy_mean) ** 2, weights=w))
            )
            charge_pc = float(numpy.sum(w)) * e * 1e12
        else:
            energy_mean = energy_std = charge_pc = 0.0
        self.stats = {
            "charge_pc": charge_pc,
            "energy_mean_mev": energy_mean,
            "energy_std_mev": energy_std,
        }


class FBPICSimulator(BaseSimulator):
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

    CONFIG_KIND = "lwfa"

    def __init__(
        self,
        hyparams: SimulationHyperparameters,
        laser: _LaserPulse,
        densities: list[_DensityProfile],
        *,
        target_species: str,
        working_directory: str | Path | None = None,
    ) -> None:
        super().__init__(
            target_species=target_species, working_directory=working_directory
        )
        self.hyparams = hyparams
        self.laser = laser
        self.densities = list(densities)
        self._remember_state()

    def config(self) -> dict[str, typing.Any]:
        return {
            "hyparams": self.hyparams,
            "laser": self.laser,
            "densities": list(self.densities),
        }

    def configure(self) -> None:
        """Validate that the current config set is complete."""
        if self.hyparams is None:
            raise ValueError("hyparams is required.")
        if self.laser is None:
            raise ValueError("laser is required.")
        if not self.densities:
            raise ValueError("At least one density profile is required.")
        self._prepare_inputs()
        self.configured = True

    def diagnostics_directory(self, working_directory: Path) -> Path:
        return working_directory / self.hyparams.save_directory / "hdf5"

    @classmethod
    def from_config(
        cls,
        config: dict[str, typing.Any],
        *,
        target_species: str,
        working_directory: str | Path | None = None,
    ) -> "FBPICSimulator":
        return cls(
            config["hyparams"],
            config["laser"],
            config["densities"],
            target_species=target_species,
            working_directory=working_directory,
        )

    def run_simulation(self) -> FBPICSimulation:
        """Build a fresh `Simulation` and run it, WITHOUT reading any output back.

        Passes `skip_if_hashed=False` to both `setup_simulation()` and `run_simulation()`.
        """
        simulation = FBPICSimulation(
            elements=[self.hyparams, self.laser, *self.densities]
        )
        simulation.setup_simulation(
            working_directory=self.working_directory, skip_if_hashed=False
        )
        simulation.run_simulation(skip_if_hashed=False)
        return simulation

    def set_config(self, config: dict[str, typing.Any]) -> None:
        self.hyparams = config["hyparams"]
        self.laser = config["laser"]
        self.densities = list(config["densities"])


class PWFASimulator(BaseSimulator):
    """A lab-frame, beam-driven plasma wakefield simulation, built directly on fbpic.

    Holds a `PWFAGrid`, a plasma density profile (`species=None` for bare electrons), a driver
    bunch and optionally a witness bunch, all frozen config objects, and rebuilds a fresh fbpic
    `Simulation` on every run. The plasma, the driver and the witness are written to the
    diagnostics as the species `plasma`, `driver` and `witness`; `target_species` (default
    `driver`) is the one read back as `final_particles`.

    Parameters
    ----------
    grid : PWFAGrid
    plasma : _DensityProfile
    driver : _ParticleBunch
    witness : _ParticleBunch, optional
    target_species : str
        `driver` or `witness`.
    working_directory : str | Path, optional
        Directory fbpic writes diagnostics under. Defaults to the current working directory.
    """

    CONFIG_KIND = "pwfa"

    def __init__(
        self,
        grid: PWFAGrid,
        plasma: _DensityProfile,
        driver: _ParticleBunch,
        witness: _ParticleBunch | None = None,
        *,
        target_species: str = "driver",
        working_directory: str | Path | None = None,
    ) -> None:
        super().__init__(
            target_species=target_species, working_directory=working_directory
        )
        self.grid = grid
        self.plasma = plasma
        self.driver = driver
        self.witness = witness
        self._remember_state()

    def config(self) -> dict[str, typing.Any]:
        config = {"grid": self.grid, "plasma": self.plasma, "driver": self.driver}
        if self.witness is not None:
            config["witness"] = self.witness
        return config

    def configure(self) -> None:
        """Validate that the current config set is complete."""
        for name in ("grid", "plasma", "driver"):
            if getattr(self, name) is None:
                raise ValueError(f"{name} is required.")
        if self.target_species not in self._species_names():
            raise ValueError(
                f"target_species {self.target_species!r} is not one of {self._species_names()}"
            )
        self._prepare_inputs()
        self.configured = True

    def diagnostics_directory(self, working_directory: Path) -> Path:
        return working_directory / self.grid.save_directory / "hdf5"

    @classmethod
    def from_config(
        cls,
        config: dict[str, typing.Any],
        *,
        target_species: str,
        working_directory: str | Path | None = None,
    ) -> "PWFASimulator":
        return cls(
            config["grid"],
            config["plasma"],
            config["driver"],
            config.get("witness"),
            target_species=target_species,
            working_directory=working_directory,
        )

    def run_simulation(self) -> typing.Any:
        """Build a fresh fbpic `Simulation` and run it, WITHOUT reading any output back."""
        from fbpic.fields.smoothing import BinomialSmoother
        from fbpic.main import Simulation
        from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic

        grid = self.grid
        if grid.random_seed is not None:
            numpy.random.seed(grid.random_seed)
        smoother = BinomialSmoother(
            n_passes={"z": grid.smoother_passes, "r": grid.smoother_passes},
            compensator={
                "z": grid.smoother_compensator,
                "r": grid.smoother_compensator,
            },
        )
        simulation = Simulation(
            grid.nz,
            grid.zmax,
            grid.nr,
            grid.rmax,
            grid.nm,
            grid.time_step,
            zmin=grid.zmin,
            boundaries=grid.boundaries,
            use_cuda=grid.use_cuda,
            smoother=smoother,
        )
        species = {}
        species["plasma"], _ = self.plasma.add_to_simulation(
            simulation, is_boosted=False
        )
        species["driver"] = self.driver.add_to_simulation(simulation)
        if self.witness is not None:
            species["witness"] = self.witness.add_to_simulation(simulation)
        simulation.set_moving_window(v=c)

        write_dir = str(self.working_directory / grid.save_directory)
        simulation.diags = [
            FieldDiagnostic(
                grid.write_period,
                simulation.fld,
                comm=simulation.comm,
                write_dir=write_dir,
            ),
            ParticleDiagnostic(
                grid.write_period,
                {
                    k: v
                    for k, v in species.items()
                    if k != "plasma" or grid.write_plasma
                },
                comm=simulation.comm,
                write_dir=write_dir,
            ),
        ]
        simulation.step(grid.n_steps)
        return simulation

    def set_config(self, config: dict[str, typing.Any]) -> None:
        self.grid = config["grid"]
        self.plasma = config["plasma"]
        self.driver = config["driver"]
        self.witness = config.get("witness")

    def _species_names(self) -> list[str]:
        return ["driver", "witness"] if self.witness is not None else ["driver"]


def _read_archive(
    h5, input_dirs: typing.Any = None, working_directory: str | Path | None = None
) -> dict[str, typing.Any]:
    """Return the contents of an `archive()` file: `config_kind`, `config` (name -> object or
    list of objects), `target_species`, `embedded` (its embedded input files), and
    `final_particles` (None) / `stats` ({}) unless the archive was written with
    `save_final_particles=True`."""
    if isinstance(h5, (str, Path)):
        with h5py.File(h5, "r") as g:
            return _read_archive(g, input_dirs, working_directory)
    paths, embedded = archive_config.load_inputs(
        h5.get("inputs"), input_dirs, working_directory or Path.cwd()
    )

    def overrides(name: str, index: int | None) -> dict[str, str]:
        return {
            field: path
            for (config, i, field), path in paths.items()
            if config == name and i == index
        }

    config: dict[str, typing.Any] = {}
    for name in h5:
        if name in _RESERVED_GROUPS:
            continue
        group = h5[name]
        if group.attrs.get("list"):
            config[name] = [
                archive_config.read_config(group[str(i)], overrides(name, i))
                for i in range(len(group))
            ]
        else:
            config[name] = archive_config.read_config(group, overrides(name, None))
    final_particles = None
    stats: dict[str, typing.Any] = {}
    if "final_particles" in h5:
        final_particles = ParticleGroup(h5=h5)
        stats = {name: float(value) for name, value in h5["stats"].attrs.items()}
    return {
        "config_kind": str(h5.attrs["config_kind"]),
        "config": config,
        "embedded": embedded,
        "target_species": str(h5.attrs["target_species"]),
        "final_particles": final_particles,
        "stats": stats,
    }

from typing import ClassVar, Literal, List, Any
from scipy.constants import c, e, epsilon_0, m_e, pi
import hashlib
import json
import numpy as np
from pathlib import Path
import logging
import time
import attrs

from inversion_fbpic.lib.serializable_config import SerializableConfig
from inversion_fbpic.lib.density_core import _DensityProfile
from inversion_fbpic.lib.laser import _LaserPulse

from fbpic.main import Simulation as FBPICSimulation, Particles
from fbpic.utils.random_seed import set_random_seed
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.lpa_utils.laser import add_laser_pulse
from fbpic.openpmd_diag import (
    BackTransformedParticleDiagnostic,
    BackTransformedFieldDiagnostic,
    ParticleDiagnostic,
    FieldDiagnostic,
    restart_from_checkpoint,
    set_periodic_checkpoint,
)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

if MPI is not None:
    MPI_COMM = MPI.COMM_WORLD
    MPI_RANK = MPI_COMM.Get_rank()
    MPI_SIZE = MPI_COMM.Get_size()
else:
    MPI_COMM = None
    MPI_RANK = 0
    MPI_SIZE = 1

HASH_FILE = "sim.sha256"
logger = logging.getLogger(__name__)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _largest_prime_factor(n):
    # Source - https://stackoverflow.com/a/22808285
    # Posted by Stefan, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-08-03, License - CC BY-SA 3.0
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n


@attrs.define(kw_only=True, slots=False, frozen=True)
class _GridParameters(SerializableConfig):
    """
    Grid parameters for the simulation.

    Args:
        dt: (float) [s] Timestep for the simulation.
        dz: (float) [m] Grid spacing along the longitudinal direction.
        dr: (float) [m] Grid spacing along the radial direction.
        nz: (int) Number of gridpoints along the longitudinal direction.
        nr: (int) Number of gridpoints along the radial direction.
        nm: (int) Number of angular modes used. Defaults to 3.
        gamma_boost: (float | None) |OPTIONAL| Gamma boost factor. Defaults to None.
        zmin: (float) [m] Minimum z position of the simulation box.
        zmax: (float) [m] Maximum z position of the simulation box.
        rmax: (float) [m] Maximum radial position of the simulation box.
    """

    CONFIG_TYPE: ClassVar[str] = "grid_parameters"
    _CONCRETE_REGISTRY: ClassVar[dict[str, type["_GridParameters"]]] = {}
    SUBCLASS: ClassVar[str] = "grid_parameters"

    dt: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    dz: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    dr: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    nz: int = attrs.field(converter=int, validator=attrs.validators.ge(1))
    nr: int = attrs.field(converter=int, validator=attrs.validators.ge(1))
    nm: int = attrs.field(converter=int, validator=attrs.validators.ge(1))
    zmin: float = attrs.field(converter=float)
    zmax: float = attrs.field(converter=float)
    rmax: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    gamma_boost: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )


@attrs.define(kw_only=True, slots=False, frozen=True)
class SimulationHyperparameters(SerializableConfig):
    """
    Hyperparameters for FBPIC simulations.

    Args:
        zmin: (float) [m] Minimum z position of the simulation box.
        zmax: (float) [m] Maximum z position of the simulation box.
        rmax: (float) [m] Maximum radial position of the simulation box.
        nz: (int) Number of gridpoints along the longitudinal direction. This should optimally have a small greatest-prime-factor for FFT performance.
        nr: (int) Number of gridpoints along the radial direction.
        nm: (int) Number of angular modes used.
        right_buffer: (float | None) |OPTIONAL| Right buffer of the simulation extent. If None, equal to the window size (zmax - zmin).
        beta_window: (float | None) |OPTIONAL| Beta of the moving window. If None, the moving window velocity is calculated to be the group velocity of
            the first LaserPulse object with a `wavelength` data attribute in conjunction with the combination of all nominal plasma densities.
            If no `wavelength` attribute is found, the moving window velocity is set to 1.0.
        use_mpi: (bool) Whether to use MPI.
        gamma_boost: (float | None) |OPTIONAL| Gamma boost factor.
        number_dumps: (int) Number of diagnostic dumps to save.
        field_diagnostics: (list[Literal["rho", "E", "B", "J"]] | None) |OPTIONAL| List of field diagnostics to save. If None, no field diagnostics are saved.
        save_directory: (str | Path) |OPTIONAL| Directory to save the diagnostics. If not absolute, this is relative to the `working_directory` passed to `Simulation.setup_simulation()`. Defaults to "diags".
        save_checkpoints: (bool) |OPTIONAL| Whether to save checkpoints. Defaults to False.
        checkpoint_period: (int) |OPTIONAL| Period to save checkpoints. Defaults to 100.
        write_period: (int) |OPTIONAL| Period to write diagnostics to disk. Defaults to 50.
        use_restart: (bool) |OPTIONAL| Whether to restart from a checkpoint. Defaults to False.
        track_electrons: (bool) |OPTIONAL| Whether to track and write particle ids. Defaults to False.
        n_order_mpi: (int) |OPTIONAL| Order of the stencil for z derivatives in the Maxwell solver if using MPI parallelization. Defaults to 32.
        random_seed: (int | None) |OPTIONAL| Random seed. Defaults to None (unset).
        r_boundary: (Literal["open", "reflective"]) |OPTIONAL| Boundary condition for the radial direction. Defaults to "reflective".
    """

    CONFIG_TYPE: ClassVar[str] = "simulation_hyperparameters"
    _CONCRETE_REGISTRY: ClassVar[dict[str, type["SimulationHyperparameters"]]] = {}
    SUBCLASS: ClassVar[str] = "simulation_hyperparameters"

    zmin: float = attrs.field(converter=float)
    zmax: float = attrs.field(converter=float)
    rmax: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    nz: int = attrs.field(converter=int)

    @nz.validator
    def _validate_nz(self, attribute: attrs.Attribute, value: int) -> None:
        if value < 1:
            raise ValueError(f"nz must be >= 1, got {value}")
        lpf = _largest_prime_factor(value)
        if lpf > 2:
            logger.warning(
                f"Unoptimal value. The largest prime factor of nz={value} is {lpf}. Smaller greatest-prime-factors are recommended."
            )

    nr: int = attrs.field(converter=int, validator=attrs.validators.ge(1))
    nm: int = attrs.field(converter=int, validator=attrs.validators.ge(1))
    use_mpi: bool = attrs.field(converter=bool)
    number_dumps: int = attrs.field(converter=int, validator=attrs.validators.ge(1))
    right_buffer: float | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=attrs.validators.optional(attrs.validators.gt(0.0)),
    )
    beta_window: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )
    gamma_boost: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )
    field_diagnostics: list[Literal["rho", "E", "B", "J"]] | None = attrs.field(
        default=None
    )
    save_directory: Path = attrs.field(default=Path("diags"), converter=Path)
    save_checkpoints: bool = attrs.field(default=False, converter=bool)
    checkpoint_period: int = attrs.field(
        default=100, converter=int, validator=attrs.validators.ge(1)
    )
    write_period: int = attrs.field(
        default=50, converter=int, validator=attrs.validators.ge(1)
    )
    use_restart: bool = attrs.field(default=False)
    track_electrons: bool = attrs.field(default=False)
    n_order_mpi: int = attrs.field(
        default=32, converter=int, validator=attrs.validators.ge(1)
    )
    random_seed: int | None = attrs.field(
        default=None, converter=attrs.converters.optional(int)
    )
    r_boundary: Literal["open", "reflective"] = attrs.field(default="reflective")

    def grid_parameters_yaml(
        self, file_name: str | None = None, comments: bool = True, indent: int = 2
    ) -> str:
        """
        Grid parameters for the simulation. Returns a YAML string of the grid parameters.
        If a file name is provided, the YAML string is written to the file.

        Args:
            file_name: The file to write the YAML string to. If None, the YAML string is not written to a file. Defaults to None.
            comments: Whether to include comments in the YAML. Defaults to True.
            indent: The indentation level for the YAML. Defaults to 2.

        Returns:
            str: A YAML string of the grid parameters.
        """
        gp = _GridParameters(
            dt=self.dt,
            dz=self.dz,
            dr=self.dr,
            nz=self.nz,
            nr=self.nr,
            nm=self.nm,
            gamma_boost=self.gamma_boost,
            zmin=self.zmin,
            zmax=self.zmax,
            rmax=self.rmax,
        )
        if file_name is not None and self.do_write_to_disk:
            gp.to_yaml_file(file_name, comments=comments, indent=indent)
        return gp.to_yaml(comments=comments, indent=indent)

    @property
    def is_boosted(self) -> bool:
        """Whether this simulation is boosted."""
        return self.gamma_boost is not None and not np.isclose(self.gamma_boost, 1.0)

    @property
    def n_order(self) -> int:
        """Order of the stencil for z derivatives in the Maxwell solver."""
        return self.n_order_mpi if self.use_mpi else -1

    @property
    def dt(self) -> float:
        """Timestep for the simulation."""
        return (
            min(
                self.rmax / (2 * self.gamma_boost * self.nr) / c,
                (self.zmax - self.zmin) / self.nz / c,
            )
            if self.is_boosted
            else (self.zmax - self.zmin) / self.nz / c
        )

    @property
    def v_comoving(self) -> float:
        """Velocity of the Galilean frame (for suppression of the NCI)."""
        if self.gamma_boost is not None and not np.isclose(self.gamma_boost, 1.0):
            return -c * np.sqrt(1.0 - 1.0 / self.gamma_boost**2)
        else:
            return 0.0

    @property
    def dz(self) -> float:
        """Grid spacing along the longitudinal direction."""
        return (self.zmax - self.zmin) / self.nz

    @property
    def dr(self) -> float:
        """Grid spacing along the radial direction."""
        return self.rmax / self.nr

    @property
    def do_write_to_disk(self) -> bool:
        """Determines whether it is safe to handle output to disk when MPI is in use."""
        if not self.use_mpi:
            return True
        else:
            return MPI_RANK == 0


@attrs.define(kw_only=True, slots=False)
class Simulation(SerializableConfig):
    """
    Simulation class for FBPIC simulations.

    Args:
        elements: (str | Path | SerializableConfig | List[str | Path | SerializableConfig]) The elements of the simulation. May be a path to a YAML or JSON file or objects of type SimulationHyperparameters, DensityProfile, or LaserPulse.
        verbosity: (int) |OPTIONAL| The verbosity of the logger. Defaults to 20 (INFO).
    """

    CONFIG_TYPE: ClassVar[str] = "simulation"
    _CONCRETE_REGISTRY: ClassVar[dict[str, type["Simulation"]]] = {}
    SUBCLASS: ClassVar[str] = "simulation"

    # Only serialized objects.
    elements: (
        str
        | Path
        | dict[str, Any]
        | SerializableConfig
        | List[str | Path | dict[str, Any] | SerializableConfig]
    ) = attrs.field()
    verbosity: int = attrs.field(default=logging.INFO, converter=int)

    hyparams: SimulationHyperparameters | None = attrs.field(default=None, init=False)
    densities: List[_DensityProfile] = attrs.field(factory=list, init=False)
    lasers: List[_LaserPulse] = attrs.field(factory=list, init=False)

    _logger: logging.Logger = attrs.field(init=False)
    simulation: FBPICSimulation = attrs.field(init=False)

    working_directory: Path = attrs.field(default=Path.cwd(), init=False)
    is_setup: bool = attrs.field(default=False, init=False)
    T_interact: float = attrs.field(default=0.0, init=False)

    def __attrs_post_init__(self) -> None:
        logging.basicConfig(level=self.verbosity)
        self._logger = logging.getLogger("Simulation")
        self._logger.setLevel(self.verbosity)

        # load and sort the simulation elements
        if isinstance(self.elements, str | Path):
            self._load_and_sort_from_path(self.elements)
        elif isinstance(self.elements, dict):
            self._sort_component(SerializableConfig.from_dict(self.elements))
        elif issubclass(type(self.elements), SerializableConfig):
            self._sort_component(self.elements)
        elif isinstance(self.elements, list):
            for input in self.elements:
                if issubclass(type(input), SerializableConfig):
                    self._sort_component(input)
                elif isinstance(input, dict):
                    self._sort_component(SerializableConfig.from_dict(input))
                elif isinstance(input, str | Path):
                    self._load_and_sort_from_path(input)
                else:
                    self._logger.critical(
                        f"Unsupported input type: {type(input).__name__}. Must be SerializableConfig, str, or Path."
                    )
                    raise ValueError(
                        f"Unsupported input type: {type(input).__name__}. Must be SerializableConfig, str, or Path."
                    )
        else:
            self._logger.critical(
                f"Unsupported inputs type: {type(self.elements).__name__}. Must be str, Path, or list of str, Path, or SerializableConfig."
            )
            raise ValueError(
                f"Unsupported inputs type: {type(self.elements).__name__}. Must be str, Path, or list of str, Path, or SerializableConfig."
            )

        if self.hyparams is None:
            self._logger.critical(
                "No SimulationHyperparameters object found. Please provide a SimulationHyperparameters object or a path to a file or directory containing a SimulationHyperparameters object."
            )
            raise ValueError(
                "No SimulationHyperparameters object found. Please provide a SimulationHyperparameters object or a path to a file or directory containing a SimulationHyperparameters object."
            )
        if len(self.densities) == 0:
            self._logger.critical(
                "No DensityProfile objects found. Please provide a DensityProfile object or a path to a file or directory containing a DensityProfile object."
            )
            raise ValueError(
                "No DensityProfile objects found. Please provide a DensityProfile object or a path to a file or directory containing a DensityProfile object."
            )
        if len(self.lasers) == 0:
            self._logger.critical(
                "No LaserPulse objects found. Please provide a LaserPulse object or a path to a file or directory containing a LaserPulse object."
            )
            raise ValueError(
                "No LaserPulse objects found. Please provide a LaserPulse object or a path to a file or directory containing a LaserPulse object."
            )

    @property
    def z_extent(self) -> tuple[float, float]:
        """The extent of the simulation."""
        minz, maxz = None, None
        for density in self.densities:
            c_minz, c_maxz = density.get_z_extent()
            if minz is None or c_minz < minz:
                minz = c_minz
            if maxz is None or c_maxz > maxz:
                maxz = c_maxz
        return minz, maxz

    @property
    def do_write_to_disk(self) -> bool:
        """Determines whether it is safe to handle output to disk when MPI is in use."""
        return self.hyparams.do_write_to_disk

    def config_hash(self) -> str:
        """Return a stable SHA-256 hex digest of this simulation's configuration.

        The hash is independent of the order in which elements were provided.
        """
        payload = {
            "simulation_hyperparameters": self.hyparams.to_dict(),
            "density_profiles": sorted(
                (den.to_dict() for den in self.densities),
                key=_canonical_json,
            ),
            "laser_pulses": sorted(
                (las.to_dict() for las in self.lasers),
                key=_canonical_json,
            ),
        }
        digest_input = _canonical_json(payload)
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()

    def _sort_component(
        self,
        payload: SerializableConfig,
        path: str | Path | None = None,
        skip_incompatible: bool = False,
    ) -> None:
        """
        Sort the component into the collections.

        Args:
            payload: The SerializableConfig object to sort.
            path: The path to the file or directory that the payload was loaded from.
            skip_incompatible: Whether to skip incompatible objects rather than raising an error. Defaults to False.

        Returns:
            None
        """
        if isinstance(payload, SimulationHyperparameters):
            if self.hyparams is None:
                self.hyparams = payload
            else:
                self._logger.critical(
                    "Multiple SimulationHyperparameters objects found. Please provide only one SimulationHyperparameters object."
                )
                raise ValueError(
                    "Multiple SimulationHyperparameters objects found. Please provide only one SimulationHyperparameters object."
                )
        elif isinstance(payload, _DensityProfile):
            self.densities.append(payload)
        elif isinstance(payload, _LaserPulse):
            self.lasers.append(payload)
        else:
            if skip_incompatible:
                self._logger.warning(
                    f"Skipping incompatible object: {type(payload).__name__}. Must be SimulationHyperparameters, DensityProfile, or LaserPulse."
                )
                return
            else:
                self._logger.critical(
                    f"Unsupported object type: {type(payload).__name__}. Must be SimulationHyperparameters, DensityProfile, or LaserPulse."
                )
                raise ValueError(
                    f"Unsupported object type: {type(payload).__name__}. Must be SimulationHyperparameters, DensityProfile, or LaserPulse."
                )

        if path is not None:
            self._logger.info(f"Loaded {type(payload).__name__} from {str(path)}")
        else:
            self._logger.info(f"Loaded {type(payload).__name__}")
        self._logger.debug(f"{type(payload).__name__}: {payload.to_dict()}")

    def _element_config_anchor(self) -> Path | None:
        """Directory for resolving ``elements`` path strings."""
        if self.source_file is not None:
            return self.source_file.parent
        return SerializableConfig._nested_path_relative_to()

    def _load_and_sort_from_path(
        self, path: str | Path, skip_incompatible: bool = False
    ) -> None:
        """
        Load the object from a file or directory and sort it into the collections. If a directory is provided, all files in the directory are loaded and sorted.

        Args:
            path: The path to the file or directory to load.
            skip_incompatible: Whether to skip incompatible objects rather than raising an error. This is ignored and automatically set to True for directories. Defaults to False.
        Returns:
            None
        """
        relative_to = self._element_config_anchor()
        resolved = SerializableConfig._resolve_config_path(
            path, relative_to=relative_to, accept_dir=True
        )

        if resolved.is_dir():
            for file in sorted(resolved.glob("*.yaml")):
                self._load_and_sort_from_path(file, skip_incompatible=True)
            for file in sorted(resolved.glob("*.json")):
                self._load_and_sort_from_path(file, skip_incompatible=True)
            return

        payload = SerializableConfig.from_file(path, relative_to=relative_to)
        self._sort_component(payload, resolved, skip_incompatible=skip_incompatible)

    # TODO: refactor this to some other module
    @staticmethod
    def calculate_group_beta(wavelength: float, plasma_density: float) -> float:
        """
        Calculate the group velocity (normalized to the speed of light) of electromagnetic waves in a plasma.

        Args:
            wavelength: (float) [m] The wavelength of the EM wave.
            plasma_density: (float) [m^-3] The density of the plasma.

        Returns:
            (float) [c] The group velocity of the EM wave in the plasma normalized to c.
        """

        return (
            1.0
            - wavelength**2
            * e**2
            / (8.0 * pi**2 * c**2 * m_e * epsilon_0)
            * plasma_density
        )

    def calculate_mean_group_beta(
        self, wavelength: float | None = None, num_samples: int = 1000
    ) -> float:
        r"""
        Calculate the mean group velocity of EM waves in this `Simulation`'s on-axis plasma profile.
        This is the total simulation length (maximal extent) divided by the time to traverse.

        Args:
            wavelength: (float) [m] |OPTIONAL| The wavelength of the EM wave. If None, the wavelength is calculated from the mean group velocity of the first laser to have a `wavelength` attribute.
            num_samples: (int) |OPTIONAL| Number of samples to take along the plasma profile. Defaults to 1000.
        Returns:
            (float) [c] The mean group velocity of the EM wave in the plasma normalized to c.
        """

        if wavelength is None:
            for laser in self.lasers:
                if hasattr(laser, "wavelength"):
                    wavelength = laser.wavelength
                    break

        if wavelength is None:
            self._logger.warning(
                "No wavelength found for any LaserPulse objects. Setting beta_window to 1.0."
            )
            return 1.0

        zs: np.ndarray = np.linspace(
            self.z_extent[0], self.z_extent[1], num_samples, endpoint=False
        )
        total_axial_density: np.ndarray = sum(
            [
                density.build_density_function()(zs, 0.0) * density.nominal_density
                for density in self.densities
            ]
        )
        betas = self.calculate_group_beta(wavelength, total_axial_density)
        mean_beta = float(len(betas)) / sum(1.0 / betas)
        return mean_beta

    @staticmethod
    def _normalize_working_directory(
        working_directory: Path | str | None,
    ) -> Path:
        """Resolve the simulation run directory for runtime paths such as ``save_directory``."""
        if working_directory is None:
            return Path.cwd()
        return Path(working_directory).resolve()

    @property
    def _save_directory(self) -> Path:
        """Resolve the absolute diagnostics directory for this simulation."""
        if self.hyparams.save_directory.is_absolute():
            return self.hyparams.save_directory
        return self.working_directory / self.hyparams.save_directory

    @property
    def _hash_path(self) -> Path:
        """Path where the completed simulation configuration hash is stored."""
        return self._save_directory / HASH_FILE

    def _is_hashed(self) -> bool:
        """
        Check if the simulation has been hashed.
        This should only be run once self.working_directory is set.

        Returns:
            (bool) Whether the simulation has been hashed.
        """
        hash = None
        try:
            with open(self._hash_path, "r") as f:
                hash = f.read()
        except FileNotFoundError:
            pass
        return hash == self.config_hash()

    def _save_hash(self) -> None:
        """
        Save the hash of the simulation to the working directory.
        This should only be run once self.working_directory is set.

        Returns:
            None
        """
        if self.do_write_to_disk:
            self._hash_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._hash_path, "w") as f:
                f.write(self.config_hash())

    @property
    def num_steps(self) -> int:
        """Number of FBPIC iterations required for the interaction."""
        return int(self.T_interact / self.simulation.dt)

    @staticmethod
    def _particle_diagnostic_period_steps(num_steps: int, number_dumps: int) -> int:
        """Return an diagnostic period measured in steps that does not exceed a dump limit."""
        if number_dumps < 2:
            return num_steps

        period = max(1, (num_steps - 1) // (number_dumps - 1))
        while (num_steps - 1) // period + 1 > number_dumps:
            period += 1
        return period

    def setup_simulation(
        self,
        working_directory: Path | str | None = None,
        skip_if_hashed: bool = True,
        **kwargs,
    ) -> None:
        """
        Setup the simulation.

        Args:
            working_directory: (str | Path | None) |OPTIONAL| The working directory to save the simulation. If None, the working directory is the current working directory.
            skip_if_hashed: (bool) |OPTIONAL| Whether to skip the setup if there is a recorded hash that matches the current simulation configuration. Defaults to True.
            **kwargs: Additional keyword arguments for the simulation setup, passed to FBPIC's `Simulation` constructor.

        Returns:
            None
        """

        working_directory = self._normalize_working_directory(working_directory)
        self.working_directory = working_directory
        save_directory_absolute = self._save_directory

        if self._is_hashed() and skip_if_hashed:
            self._logger.warning(
                "Simulation already run and hashed with this configuration. Skipping setup."
            )
            return

        if self.is_setup:
            self._logger.warning("Simulation already setup. Skipping setup.")
            return

        # boosted frame converter
        boost = None
        if self.hyparams.is_boosted:
            boost = BoostConverter(self.hyparams.gamma_boost)

        # The interaction length of the simulation (meters)
        # Buffer length + full extent of the plasma
        right_buffer = self.hyparams.right_buffer
        if right_buffer is None:
            right_buffer = self.hyparams.zmax - self.hyparams.zmin
        L_interact = self.z_extent[1] - self.z_extent[0] + right_buffer

        # calculate the moving window velocity
        v_window = None
        beta_window = self.hyparams.beta_window
        if beta_window is None:
            beta_window = self.calculate_mean_group_beta()
        if np.isclose(beta_window, 0.0):
            self._logger.error(
                f"Window velocity is near zero. beta_window: {beta_window}"
            )
            raise ValueError(
                f"Window velocity is near zero. beta_window: {beta_window}"
            )
        v_window = c * beta_window

        # Velocity of the Galilean frame (for suppression of the NCI)
        if boost is not None:
            v_comoving = -c * np.sqrt(1.0 - 1.0 / boost.gamma0**2)
        else:
            v_comoving = 0.0

        # Interaction time, in the boosted frame (seconds)
        if boost is not None:
            self.T_interact = boost.interaction_time(
                L_interact, (self.hyparams.zmax - self.hyparams.zmin), v_window
            )
        else:
            self.T_interact = L_interact / v_window

        # Set the random seed
        if self.hyparams.random_seed is not None:
            set_random_seed(self.hyparams.random_seed)

        # Initialize the simulation object
        self.simulation = FBPICSimulation(
            Nz=self.hyparams.nz,
            Nr=self.hyparams.nr,
            Nm=self.hyparams.nm,
            dt=self.hyparams.dt,
            zmin=self.hyparams.zmin,
            zmax=self.hyparams.zmax,
            rmax=self.hyparams.rmax,
            v_comoving=v_comoving if boost is not None else None,
            gamma_boost=boost.gamma0 if boost is not None else None,
            n_order=self.hyparams.n_order_mpi if self.hyparams.use_mpi else -1,
            boundaries={"z": "open", "r": self.hyparams.r_boundary},
            use_all_mpi_ranks=self.hyparams.use_mpi,
            use_cuda=True,
            **kwargs,
        )

        # Add the particles and particle diagnostics
        if self.hyparams.number_dumps < 2:
            dt_lab_diag_period = (
                L_interact + (self.hyparams.zmax - self.hyparams.zmin)
            ) / v_window
            dn_lab_diag_period = self.num_steps
        else:
            # Avoid an extra lab-frame dump from floating-point boundary rounding.
            dt_lab_diag_period = np.nextafter(
                (L_interact + (self.hyparams.zmax - self.hyparams.zmin))
                / v_window
                / (max(1, self.hyparams.number_dumps - 1)),
                -np.inf,
            )
            dn_lab_diag_period = self._particle_diagnostic_period_steps(
                self.num_steps, self.hyparams.number_dumps
            )

        # check for duplicate elec names in the un-boosted case (FBPIC glitch)
        if boost is None:
            elec_names = [
                density.elec_name
                for density in self.densities
                if density.elec_name is not None
            ]
            ion_names = [
                density.ion_name
                for density in self.densities
                if density.ion_name is not None
            ]
            if len(elec_names) != len(set(elec_names)) or len(ion_names) != len(
                set(ion_names)
            ):
                self._logger.critical(
                    "Duplicate species names found in un-boosted case. The unboosted FBPIC particle diagnostics does not properly handle overlapping names. Please ensure that all species have unique names."
                )
                raise ValueError(
                    "Duplicate species names found in un-boosted case. The unboosted FBPIC particle diagnostics does not properly handle overlapping names. Please ensure that all species have unique names."
                )

        elecs: List[Particles] = []
        ions: List[Particles] = []
        for density in self.densities:
            elec: Particles
            ion: Particles | None
            elec, ion = density.add_to_simulation(
                self.simulation, is_boosted=boost is not None
            )
            elecs.append(elec)
            if ion is not None:
                ions.append(ion)

            # add diags
            for name, select, spec in [
                (density.elec_name, density.elec_select, elec),
                (density.ion_name, density.ion_select, ion),
            ]:
                if name is not None:
                    if boost is not None:
                        self.simulation.diags.append(
                            BackTransformedParticleDiagnostic(
                                self.hyparams.zmin,
                                self.hyparams.zmax,
                                v_window,
                                dt_lab_diag_period,
                                self.hyparams.number_dumps,
                                boost.gamma0,
                                self.hyparams.write_period,
                                self.simulation.fld,
                                species={name: spec},
                                select=select,
                                comm=self.simulation.comm,
                                write_dir=save_directory_absolute,
                            ),
                        )
                    else:
                        self.simulation.diags.append(
                            ParticleDiagnostic(
                                period=dn_lab_diag_period,
                                species={name: spec},
                                select=select,
                                comm=self.simulation.comm,
                                write_dir=save_directory_absolute,
                            ),
                        )
            # /add diags
            # add tracking
            if self.hyparams.track_electrons and not self.hyparams.use_restart:
                elec.track(self.simulation.comm)
        # /Add the particles and particle diagnostics

        # Add the laser pulses
        for laser in self.lasers:
            add_laser_pulse(
                self.simulation,
                laser.build_laser_profile(),
                gamma_boost=boost.gamma0 if boost is not None else None,
                method=laser.method if laser.method is not None else "direct",
                z0_antenna=laser.z0_antenna if laser.method == "antenna" else None,
                v_antenna=laser.v_antenna if laser.method == "antenna" else 0,
            )
        # /Add the laser pulses

        # Moving window
        if boost is not None:
            (v_window_boosted,) = boost.velocity([v_window])
            self.simulation.set_moving_window(v=v_window_boosted)
        else:
            self.simulation.set_moving_window(v=v_window)

        # Add field diagnostics
        if self.hyparams.field_diagnostics is not None:
            if boost is not None:
                self.simulation.diags.append(
                    BackTransformedFieldDiagnostic(
                        self.hyparams.zmin,
                        self.hyparams.zmax,
                        v_window,
                        dt_lab_diag_period,
                        self.hyparams.number_dumps,
                        boost.gamma0,
                        fieldtypes=self.hyparams.field_diagnostics,
                        period=self.hyparams.write_period,
                        fldobject=self.simulation.fld,
                        comm=self.simulation.comm,
                        write_dir=save_directory_absolute,
                    )
                )
            else:
                self.simulation.diags.append(
                    FieldDiagnostic(
                        period=dn_lab_diag_period,
                        fldobject=self.simulation.fld,
                        comm=self.simulation.comm,
                        write_dir=save_directory_absolute,
                    )
                )
        # /Add field diagnostics

        # Checkpoints
        if self.hyparams.save_checkpoints:
            set_periodic_checkpoint(self.simulation, self.hyparams.checkpoint_period)
        if self.hyparams.use_restart:
            restart_from_checkpoint(self.simulation)

        self.is_setup = True

    # /setup_simulation

    def run_simulation(
        self,
        show_progress: bool = True,
        logger_friendly_progress: bool = False,
        record_hash: bool = True,
        skip_if_hashed: bool = True,
    ) -> None:
        """
        Run the simulation. `setup_simulation()` must be called before this function.

        Args:
            show_progress: (bool) Whether to show the progress bar. Defaults to True.
            logger_friendly_progress: (bool) Whether to use logger-friendly output for indicating progress. Defaults to False.
            record_hash: (bool) Whether to record the hash of the simulation configuration upon successful completion. Defaults to True.
            skip_if_hashed: (bool) Whether to skip the simulation if the configuration hash is the same as a previous run. Defaults to True.

        Returns:
            None
        """

        if self._is_hashed() and skip_if_hashed:
            self._logger.warning(
                "Simulation already run and hashed with this configuration. Skipping run."
            )
            return

        if not self.is_setup:
            self._logger.error(
                "Simulation not setup. Run `Simulation.setup_simulation()` before `Simulation.run_simulation()`."
            )
            raise ValueError(
                "Simulation not setup. Run `Simulation.setup_simulation()` before `Simulation.run_simulation()`."
            )

        if not logger_friendly_progress:
            self.simulation.step(self.num_steps, show_progress=show_progress)
        else:
            # figure out how many sub-steps to run
            num_steps = self.num_steps
            if num_steps < 50:
                num_sub_steps = 1
            elif num_steps < 500:
                num_sub_steps = 10
            else:
                num_sub_steps = 100

            # distribute total number of steps across num_sub_steps
            steps = [num_steps // num_sub_steps] * num_sub_steps
            # distribute the remainder evenly across the sub-steps
            remainder = num_steps % num_sub_steps
            for i in range(remainder):
                steps[i] += 1

            # run the simulation
            self._logger.info(f"Running {num_steps} steps...")
            time0 = time.time()
            time00 = time0
            for i in range(num_sub_steps):
                self.simulation.step(steps[i], show_progress=False)
                time1 = time.time()
                self._logger.info(
                    f" {round(float(i+1)/num_sub_steps*100):02d}%, {round((num_sub_steps-i-1) * (time1-time0)):d} seconds remaining"
                )
                time0 = time1

            self._logger.info(f"Completed in {round((time0-time00)):d} seconds.")

        # TODO: read simulation output to see if it actually was successful
        if record_hash and self.do_write_to_disk:
            self._save_hash()

    # TODO: Add a method for plotting the constituent DensityProfiles like in Chris' results. Include lineouts of gas pressure and the total plasma density.
    # TODO: Add a method for plotting the laser pulse profiles (probably just calling the LaserProfile.plot() method)

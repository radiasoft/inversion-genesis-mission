"""Class-based boosted-frame nitrogen ionization-injection FBPIC simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from scipy.constants import c, e, m_e, m_p

from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.lpa_utils.laser import FromLasyFileLaser, add_laser_pulse
from fbpic.main import Simulation
from fbpic.openpmd_diag import (
    BackTransformedFieldDiagnostic,
    BackTransformedParticleDiagnostic,
    restart_from_checkpoint,
    set_periodic_checkpoint,
)
from fbpic.utils.random_seed import set_random_seed

try:
    from inversion_fbpic.density_profiles.downramp_injection import (
        build_generalized_gaussian_profile,
    )
    from inversion_fbpic.utils.input_params import InputParameters
    from inversion_fbpic.utils.laser import HighOrderLasyLaser
    from inversion_fbpic.utils.simulation_setup_tools import start_simulation_single
except ImportError:
    from downramp_injection import build_generalized_gaussian_profile  # type: ignore
    from input_params import InputParameters  # type: ignore
    from laser import HighOrderLasyLaser  # type: ignore
    from simulation_setup_tools import start_simulation_single  # type: ignore


class IonizationInjectionSimulation:
    """Configure and run a LASY-driven nitrogen ionization-injection case.

    ``physical_parameters`` contains flat, required laser and gas properties.
    ``hyperparameters`` holds FBPIC grid, particle-loading, diagnostic, and
    execution settings. Laser-prefixed physical parameters are passed directly
    to :class:`HighOrderLasyLaser`.
    """

    _PHYSICAL_PARAMETER_KEYS = frozenset(
        {
            "peak_plasma_density_cm3",
            "nitrogen_dopant_fraction",
            "density_peak",
            "density_alpha_m",
            "density_beta",
            "density_center_location_m",
            *HighOrderLasyLaser.PHYSICAL_PARAMETER_KEYS,
        }
    )
    _HYPERPARAMETER_DEFAULTS: dict[str, Any] = {
        "use_cuda": True,
        "n_order": 32,
        "gamma_boost": 1.5,
        "nz": 600,
        "nr": 300,
        "zmax": 0.0,
        "zmin": -70.0e-6,
        "rmax": 140.0e-6,
        "n_azimuthal_modes": 5,
        "p_zmin": 0.0,
        "p_rmax": 90e-6,
        "p_nz": 4,
        "p_nr": 4,
        "p_nt": 30,
        "number_dumps_lab": 100,
        "write_period": 50,
        "save_checkpoints": False,
        "checkpoint_period": 100,
        "use_restart": False,
        "track_electrons": False,
        "random_seed": 0,
        "lasy_file": Path(__file__).with_name("experimental_laser"),
        "lasy_t_start": 0.0,
        "lasy_antenna_position": 0.0,
        "lab_diagnostic_directory": "lab_diags",
        "laser_polarization": (1, 0),
        "laser_n_azimuthal_modes": 5,
        "laser_num_points": (600, 900),
        "laser_hi_range": 8.0,
        "laser_center_and_remove_tilt": True,
        "laser_centering_angles": 72,
    }

    def __init__(
        self,
        physical_parameters: Mapping[str, Any],
        hyperparameters: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.physical_parameters = dict(physical_parameters)
        self.hyperparameters = {
            **self._HYPERPARAMETER_DEFAULTS,
            **(hyperparameters or {}),
        }
        self._validate_parameters()
        self.lasy_laser_file = self._laser_output_path()

    def _validate_parameters(self) -> None:
        missing_physical = self._PHYSICAL_PARAMETER_KEYS - set(self.physical_parameters)
        unknown_physical = set(self.physical_parameters) - self._PHYSICAL_PARAMETER_KEYS
        unknown_hyperparameters = set(self.hyperparameters) - set(
            self._HYPERPARAMETER_DEFAULTS
        )
        if missing_physical:
            raise ValueError(f"Missing physical parameters: {sorted(missing_physical)}")
        if unknown_physical:
            raise ValueError(f"Unknown physical parameters: {sorted(unknown_physical)}")
        if unknown_hyperparameters:
            raise ValueError(
                f"Unknown hyperparameters: {sorted(unknown_hyperparameters)}"
            )
        if not 0.0 <= self.physical_parameters["nitrogen_dopant_fraction"] <= 1.0:
            raise ValueError("nitrogen_dopant_fraction must be in the interval [0, 1].")
    def _laser_output_path(self) -> Path:
        requested_path = Path(self.hyperparameters["lasy_file"])
        return requested_path.parent / f"{requested_path.stem}_00000.h5"

    def _prepare_laser(self, sim: Simulation) -> Path:
        """Build the LASY file once, then synchronize ranks before loading it."""
        if sim.comm.rank == 0:
            high_order_laser = HighOrderLasyLaser(
                {
                    key: self.physical_parameters[key]
                    for key in HighOrderLasyLaser.PHYSICAL_PARAMETER_KEYS
                },
                {
                    "polarization": self.hyperparameters["laser_polarization"],
                    "n_azimuthal_modes": self.hyperparameters[
                        "laser_n_azimuthal_modes"
                    ],
                    "num_points": self.hyperparameters["laser_num_points"],
                    "hi_range": self.hyperparameters["laser_hi_range"],
                    "center_and_remove_tilt": self.hyperparameters[
                        "laser_center_and_remove_tilt"
                    ],
                    "centering_angles": self.hyperparameters[
                        "laser_centering_angles"
                    ],
                },
            )
            self.lasy_laser_file = high_order_laser.save(
                self.hyperparameters["lasy_file"]
            )

        if sim.comm.mpi_comm is not None:
            sim.comm.mpi_comm.barrier()
        if not self.lasy_laser_file.is_file():
            raise FileNotFoundError(
                f"LASY laser file was not created: {self.lasy_laser_file}"
            )
        return self.lasy_laser_file

    def setup_simulation(self) -> tuple[Simulation, float]:
        """Create the FBPIC simulation and return it with its interaction time."""
        hyperparameters = self.hyperparameters
        physical_parameters = self.physical_parameters
        boost = BoostConverter(hyperparameters["gamma_boost"])
        dt = min(
            hyperparameters["rmax"] / (2 * boost.gamma0 * hyperparameters["nr"]) / c,
            (hyperparameters["zmax"] - hyperparameters["zmin"]) / hyperparameters["nz"] / c,
        )
        n_plasma = physical_parameters["peak_plasma_density_cm3"] * 1.0e6
        n_gas = n_plasma / 2.0
        v_window = c * (
            1.0
            - 0.5
            * physical_parameters["peak_plasma_density_cm3"]
            / 2.0
            * 1e6
            / 1.75e27
        )
        v_comoving = -c * np.sqrt(1.0 - 1.0 / boost.gamma0**2)
        density = build_generalized_gaussian_profile(
            gauss_peak=physical_parameters["density_peak"],
            gauss_alpha=physical_parameters["density_alpha_m"],
            gauss_beta=physical_parameters["density_beta"],
            gauss_z0=physical_parameters["density_center_location_m"],
        )
        interaction_length = (
            2.8 * physical_parameters["density_alpha_m"]
            + physical_parameters["density_center_location_m"]
        )
        interaction_time = boost.interaction_time(
            interaction_length,
            (hyperparameters["zmax"] - hyperparameters["zmin"]),
            v_window,
        )

        set_random_seed(hyperparameters["random_seed"])
        sim = Simulation(
            hyperparameters["nz"],
            hyperparameters["zmax"],
            hyperparameters["nr"],
            hyperparameters["rmax"],
            hyperparameters["n_azimuthal_modes"],
            dt,
            zmin=hyperparameters["zmin"],
            v_comoving=v_comoving,
            gamma_boost=boost.gamma0,
            n_order=hyperparameters["n_order"],
            use_cuda=hyperparameters["use_cuda"],
            boundaries={"z": "open", "r": "reflective"},
        )
        atoms_he = sim.add_new_species(
            q=0,
            m=4.0 * m_p,
            n=n_gas * (1.0 - physical_parameters["nitrogen_dopant_fraction"]),
            dens_func=density,
            boost_positions_in_dens_func=True,
            p_nz=hyperparameters["p_nz"],
            p_nr=hyperparameters["p_nr"],
            p_nt=hyperparameters["p_nt"],
            p_zmin=hyperparameters["p_zmin"],
            p_rmax=hyperparameters["p_rmax"],
        )
        atoms_n = sim.add_new_species(
            q=0,
            m=14.0 * m_p,
            n=n_gas * physical_parameters["nitrogen_dopant_fraction"],
            dens_func=density,
            boost_positions_in_dens_func=True,
            p_nz=hyperparameters["p_nz"],
            p_nr=hyperparameters["p_nr"],
            p_nt=hyperparameters["p_nt"],
            p_zmin=hyperparameters["p_zmin"],
            p_rmax=hyperparameters["p_rmax"],
        )
        elec_he = sim.add_new_species(q=-e, m=m_e)
        elec_n = sim.add_new_species(q=-e, m=m_e)
        atoms_he.make_ionizable("He", target_species=elec_he, level_start=0)
        atoms_n.make_ionizable("N", target_species=elec_n, level_start=0)

        laser_file = self._prepare_laser(sim)
        laser_profile = FromLasyFileLaser(
            str(laser_file),
            t_start=hyperparameters["lasy_t_start"],
        )
        add_laser_pulse(
            sim,
            laser_profile,
            gamma_boost=boost.gamma0,
            method="antenna",
            z0_antenna=hyperparameters["lasy_antenna_position"],
        )

        if hyperparameters["use_restart"]:
            restart_from_checkpoint(sim)
        elif hyperparameters["track_electrons"]:
            elec_n.track(sim.comm)

        (v_window_boosted,) = boost.velocity([v_window])
        sim.set_moving_window(v=v_window_boosted)
        self._configure_diagnostics(
            sim,
            elec_n,
            v_window,
            interaction_length,
            boost,
        )
        if hyperparameters["save_checkpoints"]:
            set_periodic_checkpoint(sim, hyperparameters["checkpoint_period"])
        if sim.comm.rank == 0:
            self._save_input_parameters(dt)
        return sim, interaction_time

    def _configure_diagnostics(
        self,
        sim: Simulation,
        nitrogen_electrons: Any,
        v_window: float,
        interaction_length: float,
        boost: BoostConverter,
    ) -> None:
        hyperparameters = self.hyperparameters
        dt_lab_diag_period = (
            (interaction_length + (hyperparameters["zmax"] - hyperparameters["zmin"]))
            / v_window
            / (hyperparameters["number_dumps_lab"] - 1)
        )
        sim.diags = [
            BackTransformedParticleDiagnostic(
                hyperparameters["zmin"],
                hyperparameters["zmax"],
                v_window,
                dt_lab_diag_period,
                hyperparameters["number_dumps_lab"],
                boost.gamma0,
                hyperparameters["write_period"],
                sim.fld,
                select={"uz": [10.0, None]},
                species={"nitrogen_electrons": nitrogen_electrons},
                comm=sim.comm,
                write_dir=hyperparameters["lab_diagnostic_directory"],
            ),
            BackTransformedFieldDiagnostic(
                hyperparameters["zmin"],
                hyperparameters["zmax"],
                v_window,
                dt_lab_diag_period,
                hyperparameters["number_dumps_lab"],
                boost.gamma0,
                fieldtypes=["rho", "E", "B"],
                period=hyperparameters["write_period"],
                fldobject=sim.fld,
                comm=sim.comm,
                write_dir=hyperparameters["lab_diagnostic_directory"],
            ),
        ]

    def _save_input_parameters(self, dt: float) -> None:
        """Persist the resolved laser, gas, and FBPIC settings beside diagnostics."""
        hyperparameters = self.hyperparameters
        physical_parameters = self.physical_parameters
        InputParameters.clear()
        InputParameters.add("PhysicalParameters", physical_parameters)
        InputParameters.add(
            "Laser",
            {
                "lasy_file": str(self.lasy_laser_file),
                "lasy_t_start_s": hyperparameters["lasy_t_start"],
                "antenna_position_m": hyperparameters["lasy_antenna_position"],
                "azimuthal_modes": hyperparameters["n_azimuthal_modes"],
                **{
                    key: physical_parameters[key]
                    for key in HighOrderLasyLaser.PHYSICAL_PARAMETER_KEYS
                },
            },
        )
        InputParameters.add(
            "Gas",
            {
                "peak_plasma_density_cm-3": physical_parameters["peak_plasma_density_cm3"],
                "nitrogen_dopant_fraction": physical_parameters["nitrogen_dopant_fraction"],
                "profile": "generalized_gaussian",
                "gaussian_peak": physical_parameters["density_peak"],
                "gaussian_alpha_m": physical_parameters["density_alpha_m"],
                "gaussian_beta": physical_parameters["density_beta"],
                "gaussian_center_m": physical_parameters["density_center_location_m"],
            },
        )
        InputParameters.add(
            "Simulation",
            {
                "gamma_boost": hyperparameters["gamma_boost"],
                "nz": hyperparameters["nz"],
                "dz": (hyperparameters["zmax"] - hyperparameters["zmin"])
                / hyperparameters["nz"],
                "nr": hyperparameters["nr"],
                "dr": hyperparameters["rmax"] / hyperparameters["nr"],
                "dt": dt,
            },
        )
        InputParameters.save_to_ini(
            Path(hyperparameters["lab_diagnostic_directory"]).parent
        )

    def run(self) -> None:
        """Launch the configured simulation through the shared FBPIC runner."""
        start_simulation_single(simulation_setup=self.setup_simulation)

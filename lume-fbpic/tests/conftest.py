"""Shared fixtures for lume_fbpic tests.

Minimal fixtures (small grid) -- mirrors the style of
Simulation_FBPIC/tests/test_lib/conftest.py's minimal-kwargs fixtures. The `model` fixture uses `dummy_run=True`, so
`set()` never runs a real simulation, and these tests never pay real fbpic execution cost
unless they explicitly call `simulator.run()`.
"""

from __future__ import annotations

import pytest

from inversion_fbpic.lib.simulation import SimulationHyperparameters
from inversion_fbpic.lib.laser import GaussianLaserPulse
from inversion_fbpic.lib.density_profiles import SmoothSineFlattop

from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.simulator import FBPICSimulator

HYPERPARAMETERS_KWARGS: dict = {
    "zmin": -1.0e-5,
    "zmax": 0.0,
    "rmax": 1.0e-5,
    "nz": 8,
    "nr": 8,
    "nm": 1,
    "use_mpi": False,
    "number_dumps": 2,
}

LASER_KWARGS: dict = {
    "z0": -3.0e-5,
    "wavelength": 8.0e-7,
    "tau_fwhm": 3.8e-14,
    "cep": 0.0,
    "waist": 2.8e-5,
    "focal_position": 3.0e-3,
    "polarization": 0.0,
}

FLATTOP_KWARGS: dict = {
    "nominal_density": 1.0e24,
    "p_nz": 1,
    "p_nr": 1,
    "p_nt": 1,
    "flattop_width": 1.0e-3,
    "upramp_length": 0.5e-3,
    "downramp_length": 0.5e-3,
    "elec_name": "electrons_flattop",
}

DOWNRAMP_KWARGS: dict = {
    "nominal_density": 0.3e24,
    "p_nz": 1,
    "p_nr": 1,
    "p_nt": 1,
    "flattop_width": 1.0e-3,
    "upramp_length": 0.5e-3,
    "downramp_length": 0.5e-3,
    "elec_name": "electrons_downramp",
}


@pytest.fixture()
def hyparams() -> SimulationHyperparameters:
    return SimulationHyperparameters(**HYPERPARAMETERS_KWARGS)


@pytest.fixture()
def laser() -> GaussianLaserPulse:
    return GaussianLaserPulse(energy=5.0, **LASER_KWARGS)


@pytest.fixture()
def densities() -> list[SmoothSineFlattop]:
    return [SmoothSineFlattop(**FLATTOP_KWARGS), SmoothSineFlattop(**DOWNRAMP_KWARGS)]


@pytest.fixture()
def simulator(hyparams, laser, densities, tmp_path) -> FBPICSimulator:
    return FBPICSimulator(
        hyparams,
        laser,
        densities,
        target_species="electrons_flattop",
        working_directory=tmp_path,
    )


@pytest.fixture()
def model(simulator) -> LUMEFBPICModel:
    return LUMEFBPICModel.from_simulator(simulator, dummy_run=True)


@pytest.fixture()
def particle_group():
    """A small synthetic electron bunch (px/py/pz in eV/c, weights in coulombs)."""
    import numpy as np
    from scipy.constants import c, e, m_e

    try:
        from beamphysics import ParticleGroup
    except ImportError:
        from pmd_beamphysics import ParticleGroup

    mc2_ev = m_e * c**2 / e
    n = 400
    rng = np.random.default_rng(3)
    return ParticleGroup(
        data={
            "x": rng.normal(1.0e-6, 3.0e-6, n),
            "y": rng.normal(0.0, 1.5e-6, n),
            "z": rng.normal(0.003, 4.0e-6, n),
            "px": rng.normal(0.5, 2.0, n) * mc2_ev,
            "py": rng.normal(0.0, 1.0, n) * mc2_ev,
            "pz": rng.normal(150.0, 30.0, n) * mc2_ev,
            "t": np.zeros(n),
            "status": np.ones(n, dtype=int),
            "weight": np.full(n, 1.0e3 * e),
            "species": "electron",
        }
    )

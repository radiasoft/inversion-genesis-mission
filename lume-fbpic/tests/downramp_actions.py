"""The HTU downramp-injection action set: `make_actions(simulator)`.

An action list for an `FBPICSimulator` shaped like `demo_downramp_simulation`: one
`GaussianLaserPulse` and two `SmoothSineFlattop` density profiles (index 0 the flattop, index 1 the
downramp). Hardcoded to that shape; `docs/examples/lwfa.py` and `ionization_injection.py` show lists
built for other shapes from the same building-block action classes (`lume_fbpic.actions`). A test
helper: the fixtures in `conftest.py` use this set.
"""

from __future__ import annotations

from lume.actions import Action

from lume_fbpic.actions import (
    DensityFieldAction,
    FinalParticlesAction,
    LaserFieldAction,
    StatAction,
)
from lume_fbpic.simulator import FBPICSimulator


def make_actions(simulator: FBPICSimulator) -> list[Action]:
    """Build the HTU downramp-injection control/output variable set.

    Matches `demo_downramp_simulation`'s config shape: one `GaussianLaserPulse` and two
    `SmoothSineFlattop` density profiles (index 0 = flattop, index 1 = downramp).

    Control variables (writable): laser energy, focal position, temporal width; flattop and
    downramp nominal densities; downramp length.
    Outputs (read-only): charge, mean/std energy, and `final_particles`.
    """
    return [
        LaserFieldAction(name="laser_energy", field_name="energy", unit="J"),
        LaserFieldAction(
            name="laser_focal_position", field_name="focal_position", unit="m"
        ),
        LaserFieldAction(
            name="laser_temporal_width", field_name="tau_fwhm", unit="s"
        ),
        DensityFieldAction(
            name="flattop_density",
            density_index=0,
            field_name="nominal_density",
            unit="m^-3",
        ),
        DensityFieldAction(
            name="downramp_density",
            density_index=1,
            field_name="nominal_density",
            unit="m^-3",
        ),
        DensityFieldAction(
            name="downramp_length",
            density_index=1,
            field_name="downramp_length",
            unit="m",
        ),
        StatAction(
            name="charge_pc", stat_name="charge_pc", unit="pC", read_only=True
        ),
        StatAction(
            name="energy_mean_mev",
            stat_name="energy_mean_mev",
            unit="MeV",
            read_only=True,
        ),
        StatAction(
            name="energy_std_mev",
            stat_name="energy_std_mev",
            unit="MeV",
            read_only=True,
        ),
        FinalParticlesAction(name="final_particles", read_only=True),
    ]

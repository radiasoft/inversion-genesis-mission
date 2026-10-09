"""Tests for the HTU-twin helpers in `docs/htu-twin/twin/twin_stage.py`: the bunch frame, bunch moments, ballistic
drift and the plasma-exit plane. They need neither `htu` nor `lume-pva`, but `twin_stage` imports torch,
so they are skipped without it."""

from __future__ import annotations

import numpy
import pytest
from scipy.constants import c, e
from lume_fbpic.simulator import ELECTRON_MC2_EV

from inversion_fbpic.utils import distributions

pytest.importorskip("torch")

from twin_stage import (  # noqa: E402
    beam_moments,
    bunch_frame_particles,
    drift_particles,
    plasma_exit_z,
)

from beamphysics import ParticleGroup  # noqa: E402


def _twiss_bunch(
    n=200_000,
    energy_mev=100.0,
    spread=0.05,
    beta=5.0e-3,
    alpha=0.4,
    emit=2.0e-9,
    charge_pc=30.0,
):
    """A Gaussian bunch with the given Twiss in both planes (geometric emittance `emit`)."""
    rng = numpy.random.default_rng(1)
    energy = energy_mev * 1.0e6 * (1.0 + spread * rng.normal(size=n))
    p0c = numpy.sqrt((energy_mev * 1.0e6) ** 2 - ELECTRON_MC2_EV**2)
    data = {
        "species": "electron",
        "t": numpy.zeros(n),
        "status": numpy.ones(n, dtype=int),
    }
    for plane in ("x", "y"):
        a, b = rng.normal(size=n), rng.normal(size=n)
        data[plane] = numpy.sqrt(emit * beta) * a
        data[f"p{plane}"] = numpy.sqrt(emit / beta) * (b - alpha * a) * p0c
    data["z"] = numpy.full(n, 0.003) + 4.0e-6 * rng.normal(size=n)
    data["pz"] = numpy.sqrt(
        energy**2 - ELECTRON_MC2_EV**2 - data["px"] ** 2 - data["py"] ** 2
    )
    data["weight"] = numpy.full(n, charge_pc * 1.0e-12 / n)
    return ParticleGroup(data=data)


def _lab_snapshot(n=4000, seed=0) -> ParticleGroup:
    rng = numpy.random.default_rng(seed)
    return ParticleGroup(
        data={
            "x": rng.normal(0.0, 3.0e-6, n),
            "y": rng.normal(0.0, 1.5e-6, n),
            "z": rng.normal(0.003, 4.0e-6, n),
            "px": rng.normal(0.0, 2.0, n) * ELECTRON_MC2_EV,
            "py": rng.normal(0.0, 1.0, n) * ELECTRON_MC2_EV,
            "pz": rng.normal(150.0, 40.0, n) * ELECTRON_MC2_EV,
            "t": numpy.zeros(n),
            "status": numpy.ones(n, dtype=int),
            "weight": numpy.full(n, 1.0e3 * e),
            "species": "electron",
        }
    )


def test_moments_recover_the_twiss_energy_and_charge():
    pg = _twiss_bunch(
        beta=5.0e-3, alpha=0.4, emit=2.0e-9, energy_mev=100.0, spread=0.05
    )

    moments = beam_moments(pg)

    gamma_beta = numpy.sqrt((100.0e6 / ELECTRON_MC2_EV) ** 2 - 1.0)
    for plane in ("x", "y"):
        assert moments[f"beta_{plane}_mm"] == pytest.approx(5.0, rel=0.02)
        assert moments[f"alpha_{plane}"] == pytest.approx(0.4, abs=0.02)
        assert moments[f"norm_emit_{plane}_um"] == pytest.approx(
            2.0e-9 * gamma_beta * 1e6, rel=0.02
        )
        assert moments[f"{plane}_um"] == pytest.approx(0.0, abs=0.05)
        assert moments[f"{plane}p_mrad"] == pytest.approx(0.0, abs=0.01)
    assert moments["energy_mev"] == pytest.approx(100.0, rel=1e-3)
    assert moments["energy_spread_pct"] == pytest.approx(5.0, rel=0.02)
    assert moments["charge_pc"] == pytest.approx(30.0)
    assert moments["num_particles"] == 200_000


def test_moments_report_centroid_and_pointing():
    pg = _twiss_bunch(n=20_000)
    shifted = ParticleGroup(
        data={
            **{
                n: numpy.asarray(getattr(pg, n))
                for n in ("y", "z", "py", "pz", "t", "status", "weight")
            },
            "x": numpy.asarray(pg.x) + 50.0e-6,
            "px": numpy.asarray(pg.px)
            + 1.0e-3 * numpy.sqrt((100.0e6) ** 2 - ELECTRON_MC2_EV**2),
            "species": "electron",
        }
    )

    moments = beam_moments(shifted)

    assert moments["x_um"] == pytest.approx(50.0, abs=0.5)
    assert moments["xp_mrad"] == pytest.approx(1.0, abs=0.05)
    assert moments["y_um"] == pytest.approx(0.0, abs=0.5)


def test_a_zero_emittance_plane_has_zero_twiss():
    pg = _twiss_bunch(n=100, emit=0.0)

    moments = beam_moments(pg)

    assert moments["beta_x_mm"] == 0.0 and moments["alpha_x"] == 0.0
    assert moments["norm_emit_x_um"] == 0.0


def test_moments_need_at_least_two_charged_particles():
    one = ParticleGroup(
        data={
            "x": numpy.zeros(1),
            "y": numpy.zeros(1),
            "z": numpy.zeros(1),
            "px": numpy.zeros(1),
            "py": numpy.zeros(1),
            "pz": numpy.full(1, 1e8),
            "t": numpy.zeros(1),
            "status": numpy.ones(1, dtype=int),
            "weight": numpy.ones(1) * 1e-15,
            "species": "electron",
        }
    )
    with pytest.raises(ValueError, match="at least two"):
        beam_moments(one)


@pytest.mark.parametrize("length", [0.052, -0.00088])
def test_drift_matches_cheetahs_drift_kick_drift(length):
    import torch

    cheetah = pytest.importorskip("cheetah")
    pg = bunch_frame_particles(_lab_snapshot(), uz_min=30.0, central_fraction=None)
    energy = float(numpy.average(pg.energy, weights=pg.weight))
    beam = cheetah.ParticleBeam.from_openpmd_particlegroup(
        pg, energy=torch.tensor(energy, dtype=torch.float64), dtype=torch.float64
    )
    out = cheetah.Drift(
        length=torch.tensor(length, dtype=torch.float64),
        tracking_method="drift_kick_drift",
    ).track(beam)

    drifted = drift_particles(pg, length)

    numpy.testing.assert_allclose(out.x.numpy(), numpy.asarray(drifted.x), atol=1e-15)
    numpy.testing.assert_allclose(out.y.numpy(), numpy.asarray(drifted.y), atol=1e-15)
    cheetah_dtau = out.tau.numpy() - beam.tau.numpy()
    ours_dtau = c * (numpy.asarray(drifted.t) - numpy.asarray(pg.t))
    numpy.testing.assert_allclose(cheetah_dtau, ours_dtau, atol=1e-14)


def test_zero_drift_changes_nothing():
    pg = bunch_frame_particles(_lab_snapshot(), uz_min=30.0, central_fraction=None)

    out = drift_particles(pg, 0.0)

    for name in ("x", "y", "z", "t", "px", "py", "pz", "weight"):
        numpy.testing.assert_allclose(
            numpy.asarray(getattr(out, name)), numpy.asarray(getattr(pg, name))
        )


def test_drifting_forward_then_back_restores_the_bunch():
    pg = bunch_frame_particles(_lab_snapshot(), uz_min=30.0, central_fraction=None)

    out = drift_particles(drift_particles(pg, 0.05), -0.05)

    numpy.testing.assert_allclose(numpy.asarray(out.x), numpy.asarray(pg.x), atol=1e-15)
    numpy.testing.assert_allclose(numpy.asarray(out.t), numpy.asarray(pg.t), atol=1e-18)
    numpy.testing.assert_allclose(numpy.asarray(out.z), numpy.asarray(pg.z), atol=1e-15)


def test_drift_leaves_momenta_charge_and_status_alone():
    pg = bunch_frame_particles(_lab_snapshot(), uz_min=30.0, central_fraction=None)

    out = drift_particles(pg, 0.01)

    for name in ("px", "py", "pz", "weight", "status"):
        assert numpy.array_equal(
            numpy.asarray(getattr(out, name)), numpy.asarray(getattr(pg, name))
        )
    assert numpy.allclose(numpy.asarray(out.z), numpy.asarray(pg.z) + 0.01)


def test_on_axis_higher_energy_particles_gain_time_over_a_forward_drift():
    pg = bunch_frame_particles(_lab_snapshot(), uz_min=30.0, central_fraction=None)
    on_axis = ParticleGroup(
        data={
            **{
                n: numpy.asarray(getattr(pg, n))
                for n in ("x", "y", "z", "t", "pz", "status", "weight")
            },
            "px": numpy.zeros(len(pg)),
            "py": numpy.zeros(len(pg)),
            "species": "electron",
        }
    )

    out = drift_particles(on_axis, 0.05)

    dt = numpy.asarray(out.t) - numpy.asarray(on_axis.t)
    by_energy = numpy.argsort(numpy.asarray(on_axis.energy))
    assert numpy.all(
        numpy.diff(dt[by_energy]) <= 1e-21
    )  # more energy, earlier (the gain is ~ 1/gamma^2)
    assert dt[by_energy[0]] > dt[by_energy[-1]]
    assert numpy.array_equal(
        numpy.asarray(out.x), numpy.asarray(on_axis.x)
    )  # no angle, no transverse move


def test_off_axis_particles_are_delayed_by_their_longer_path():
    pg = bunch_frame_particles(_lab_snapshot(), uz_min=30.0, central_fraction=None)
    px = numpy.asarray(pg.px)
    out = drift_particles(pg, 0.05)

    dt = numpy.asarray(out.t) - numpy.asarray(pg.t)
    larger_angle = numpy.abs(px / numpy.asarray(pg.pz)) > numpy.median(
        numpy.abs(px / numpy.asarray(pg.pz))
    )
    # the angle term (path ~ 1 + theta^2/2) outweighs the energy term at gamma ~ 150
    assert dt[larger_angle].mean() > dt[~larger_angle].mean()


def test_plasma_exit_is_the_largest_profile_end():
    class Profile:
        def __init__(self, extent):
            self._extent = extent

        def get_z_extent(self):
            return self._extent

    assert plasma_exit_z([Profile((0.0, 1.0e-3)), Profile((0.5e-3, 5.4e-3))]) == 5.4e-3


def test_arrival_time_is_minus_the_offset_from_the_mean_z_over_c():
    pg = _lab_snapshot()

    out = bunch_frame_particles(pg, uz_min=None, central_fraction=None)

    z = numpy.asarray(out.z)
    assert numpy.allclose(numpy.asarray(out.t), -(z - z.mean()) / c)
    assert numpy.corrcoef(z, numpy.asarray(out.t))[0, 1] == pytest.approx(-1.0)
    assert numpy.average(
        numpy.asarray(out.t), weights=numpy.asarray(out.weight)
    ) == pytest.approx(0.0, abs=1e-18)


def test_selection_matches_the_descriptor_library():
    pg = _lab_snapshot()
    phase_space = numpy.stack(
        [
            pg.x,
            pg.px / ELECTRON_MC2_EV,
            pg.y,
            pg.py / ELECTRON_MC2_EV,
            pg.z,
            pg.pz / ELECTRON_MC2_EV,
        ],
        axis=-1,
    )
    weights = numpy.asarray(pg.weight) / e
    expected, expected_w = distributions.crop_central_particles(
        *distributions.select_by_uz(phase_space, weights, uz_min=120.0),
        central_fraction=0.9,
    )

    out = bunch_frame_particles(pg, uz_min=120.0, central_fraction=0.9)

    assert len(out) == len(expected)
    assert numpy.asarray(out.weight).sum() / e == pytest.approx(expected_w.sum())
    assert numpy.sort(numpy.asarray(out.pz) / ELECTRON_MC2_EV) == pytest.approx(
        numpy.sort(expected[:, 5])
    )


def test_other_fields_and_weights_are_unchanged_and_status_is_one():
    pg = _lab_snapshot()

    out = bunch_frame_particles(pg, uz_min=None, central_fraction=None)

    for name in ("x", "y", "z", "px", "py", "pz", "weight"):
        assert numpy.array_equal(
            numpy.asarray(getattr(out, name)), numpy.asarray(getattr(pg, name))
        )
    assert numpy.all(numpy.asarray(out.status) == 1)
    assert out.species == "electron"


def test_selection_that_leaves_too_few_particles_is_an_error():
    with pytest.raises(ValueError, match="fewer than two"):
        bunch_frame_particles(_lab_snapshot(), uz_min=1.0e6)


def test_central_fraction_must_be_a_fraction():
    with pytest.raises(ValueError, match="central_fraction"):
        bunch_frame_particles(_lab_snapshot(), central_fraction=1.5)

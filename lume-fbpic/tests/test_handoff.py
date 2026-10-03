"""Tests for `lume_fbpic.handoff`: bunch-frame conversion and the bunch moments."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import c, e, m_e

from lume_fbpic.handoff import (
    _slice_charge_fractions,
    beam_moments,
    bunch_frame_particles,
    drift_particles,
    particles_from_descriptor,
    plasma_exit_z,
)

from inversion_fbpic.utils import distributions

try:
    from beamphysics import ParticleGroup
except ImportError:
    from pmd_beamphysics import ParticleGroup

_MC2_EV = m_e * c**2 / e


def _twiss_bunch(
    n=200_000, energy_mev=100.0, spread=0.05, beta=5.0e-3, alpha=0.4, emit=2.0e-9, charge_pc=30.0
):
    """A Gaussian bunch with the given Twiss in both planes (geometric emittance `emit`)."""
    rng = np.random.default_rng(1)
    energy = energy_mev * 1.0e6 * (1.0 + spread * rng.normal(size=n))
    p0c = np.sqrt((energy_mev * 1.0e6) ** 2 - _MC2_EV**2)
    data = {"species": "electron", "t": np.zeros(n), "status": np.ones(n, dtype=int)}
    for plane in ("x", "y"):
        a, b = rng.normal(size=n), rng.normal(size=n)
        data[plane] = np.sqrt(emit * beta) * a
        data[f"p{plane}"] = np.sqrt(emit / beta) * (b - alpha * a) * p0c
    data["z"] = np.full(n, 0.003) + 4.0e-6 * rng.normal(size=n)
    data["pz"] = np.sqrt(energy**2 - _MC2_EV**2 - data["px"] ** 2 - data["py"] ** 2)
    data["weight"] = np.full(n, charge_pc * 1.0e-12 / n)
    return ParticleGroup(data=data)


def _lab_snapshot(n=4000, seed=0) -> ParticleGroup:
    rng = np.random.default_rng(seed)
    return ParticleGroup(
        data={
            "x": rng.normal(0.0, 3.0e-6, n),
            "y": rng.normal(0.0, 1.5e-6, n),
            "z": rng.normal(0.003, 4.0e-6, n),
            "px": rng.normal(0.0, 2.0, n) * _MC2_EV,
            "py": rng.normal(0.0, 1.0, n) * _MC2_EV,
            "pz": rng.normal(150.0, 40.0, n) * _MC2_EV,
            "t": np.zeros(n),
            "status": np.ones(n, dtype=int),
            "weight": np.full(n, 1.0e3 * e),
            "species": "electron",
        }
    )


def test_arrival_time_is_minus_the_offset_from_the_mean_z_over_c():
    pg = _lab_snapshot()

    out = bunch_frame_particles(pg, uz_min=None, central_fraction=None)

    z = np.asarray(out.z)
    assert np.allclose(np.asarray(out.t), -(z - z.mean()) / c)
    assert np.corrcoef(z, np.asarray(out.t))[0, 1] == pytest.approx(-1.0)
    assert np.average(np.asarray(out.t), weights=np.asarray(out.weight)) == pytest.approx(0.0, abs=1e-18)


def test_selection_matches_the_descriptor_library():
    pg = _lab_snapshot()
    phase_space = np.stack(
        [pg.x, pg.px / _MC2_EV, pg.y, pg.py / _MC2_EV, pg.z, pg.pz / _MC2_EV], axis=-1
    )
    weights = np.asarray(pg.weight) / e
    expected, expected_w = distributions.crop_central_particles(
        *distributions.select_by_uz(phase_space, weights, uz_min=120.0), central_fraction=0.9
    )

    out = bunch_frame_particles(pg, uz_min=120.0, central_fraction=0.9)

    assert len(out) == len(expected)
    assert np.asarray(out.weight).sum() / e == pytest.approx(expected_w.sum())
    assert np.sort(np.asarray(out.pz) / _MC2_EV) == pytest.approx(np.sort(expected[:, 5]))


def test_other_fields_and_weights_are_unchanged_and_status_is_one():
    pg = _lab_snapshot()

    out = bunch_frame_particles(pg, uz_min=None, central_fraction=None)

    for name in ("x", "y", "z", "px", "py", "pz", "weight"):
        assert np.array_equal(np.asarray(getattr(out, name)), np.asarray(getattr(pg, name)))
    assert np.all(np.asarray(out.status) == 1)
    assert out.species == "electron"


def test_selection_that_leaves_too_few_particles_is_an_error():
    with pytest.raises(ValueError, match="fewer than two"):
        bunch_frame_particles(_lab_snapshot(), uz_min=1.0e6)


def test_central_fraction_must_be_a_fraction():
    with pytest.raises(ValueError, match="central_fraction"):
        bunch_frame_particles(_lab_snapshot(), central_fraction=1.5)


def test_moments_recover_the_twiss_energy_and_charge():
    pg = _twiss_bunch(beta=5.0e-3, alpha=0.4, emit=2.0e-9, energy_mev=100.0, spread=0.05)

    moments = beam_moments(pg)

    gamma_beta = np.sqrt((100.0e6 / _MC2_EV) ** 2 - 1.0)
    for plane in ("x", "y"):
        assert moments[f"beta_{plane}_mm"] == pytest.approx(5.0, rel=0.02)
        assert moments[f"alpha_{plane}"] == pytest.approx(0.4, abs=0.02)
        assert moments[f"norm_emit_{plane}_um"] == pytest.approx(2.0e-9 * gamma_beta * 1e6, rel=0.02)
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
            **{n: np.asarray(getattr(pg, n)) for n in ("y", "z", "py", "pz", "t", "status", "weight")},
            "x": np.asarray(pg.x) + 50.0e-6,
            "px": np.asarray(pg.px) + 1.0e-3 * np.sqrt((100.0e6) ** 2 - _MC2_EV**2),
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
            "x": np.zeros(1), "y": np.zeros(1), "z": np.zeros(1), "px": np.zeros(1),
            "py": np.zeros(1), "pz": np.full(1, 1e8), "t": np.zeros(1),
            "status": np.ones(1, dtype=int), "weight": np.ones(1) * 1e-15, "species": "electron",
        }
    )
    with pytest.raises(ValueError, match="at least two"):
        beam_moments(one)


@pytest.mark.parametrize("length", [0.052, -0.00088])
def test_drift_matches_cheetahs_drift_kick_drift(length):
    torch = pytest.importorskip("torch")
    cheetah = pytest.importorskip("cheetah")
    pg = bunch_frame_particles(_lab_snapshot(), uz_min=30.0, central_fraction=None)
    energy = float(np.average(pg.energy, weights=pg.weight))
    beam = cheetah.ParticleBeam.from_openpmd_particlegroup(
        pg, energy=torch.tensor(energy, dtype=torch.float64), dtype=torch.float64
    )
    out = cheetah.Drift(
        length=torch.tensor(length, dtype=torch.float64), tracking_method="drift_kick_drift"
    ).track(beam)

    drifted = drift_particles(pg, length)

    np.testing.assert_allclose(out.x.numpy(), np.asarray(drifted.x), atol=1e-15)
    np.testing.assert_allclose(out.y.numpy(), np.asarray(drifted.y), atol=1e-15)
    cheetah_dtau = out.tau.numpy() - beam.tau.numpy()
    ours_dtau = c * (np.asarray(drifted.t) - np.asarray(pg.t))
    np.testing.assert_allclose(cheetah_dtau, ours_dtau, atol=1e-14)


def test_zero_drift_changes_nothing():
    pg = bunch_frame_particles(_lab_snapshot(), uz_min=30.0, central_fraction=None)

    out = drift_particles(pg, 0.0)

    for name in ("x", "y", "z", "t", "px", "py", "pz", "weight"):
        np.testing.assert_allclose(np.asarray(getattr(out, name)), np.asarray(getattr(pg, name)))


def test_drifting_forward_then_back_restores_the_bunch():
    pg = bunch_frame_particles(_lab_snapshot(), uz_min=30.0, central_fraction=None)

    out = drift_particles(drift_particles(pg, 0.05), -0.05)

    np.testing.assert_allclose(np.asarray(out.x), np.asarray(pg.x), atol=1e-15)
    np.testing.assert_allclose(np.asarray(out.t), np.asarray(pg.t), atol=1e-18)
    np.testing.assert_allclose(np.asarray(out.z), np.asarray(pg.z), atol=1e-15)


def test_drift_leaves_momenta_charge_and_status_alone():
    pg = bunch_frame_particles(_lab_snapshot(), uz_min=30.0, central_fraction=None)

    out = drift_particles(pg, 0.01)

    for name in ("px", "py", "pz", "weight", "status"):
        assert np.array_equal(np.asarray(getattr(out, name)), np.asarray(getattr(pg, name)))
    assert np.allclose(np.asarray(out.z), np.asarray(pg.z) + 0.01)


def test_on_axis_higher_energy_particles_gain_time_over_a_forward_drift():
    pg = bunch_frame_particles(_lab_snapshot(), uz_min=30.0, central_fraction=None)
    on_axis = ParticleGroup(
        data={
            **{n: np.asarray(getattr(pg, n)) for n in ("x", "y", "z", "t", "pz", "status", "weight")},
            "px": np.zeros(len(pg)),
            "py": np.zeros(len(pg)),
            "species": "electron",
        }
    )

    out = drift_particles(on_axis, 0.05)

    dt = np.asarray(out.t) - np.asarray(on_axis.t)
    by_energy = np.argsort(np.asarray(on_axis.energy))
    assert np.all(np.diff(dt[by_energy]) <= 1e-21)  # more energy, earlier (the gain is ~ 1/gamma^2)
    assert dt[by_energy[0]] > dt[by_energy[-1]]
    assert np.array_equal(np.asarray(out.x), np.asarray(on_axis.x))  # no angle, no transverse move


def test_off_axis_particles_are_delayed_by_their_longer_path():
    pg = bunch_frame_particles(_lab_snapshot(), uz_min=30.0, central_fraction=None)
    px = np.asarray(pg.px)
    out = drift_particles(pg, 0.05)

    dt = np.asarray(out.t) - np.asarray(pg.t)
    larger_angle = np.abs(px / np.asarray(pg.pz)) > np.median(np.abs(px / np.asarray(pg.pz)))
    # the angle term (path ~ 1 + theta^2/2) outweighs the energy term at gamma ~ 150
    assert dt[larger_angle].mean() > dt[~larger_angle].mean()


def test_plasma_exit_is_the_largest_profile_end():
    class Profile:
        def __init__(self, extent):
            self._extent = extent

        def get_z_extent(self):
            return self._extent

    assert plasma_exit_z([Profile((0.0, 1.0e-3)), Profile((0.5e-3, 5.4e-3))]) == 5.4e-3


def _descriptor_of(pg) -> dict[str, float]:
    phase_space = np.stack(
        [pg.x, pg.px / _MC2_EV, pg.y, pg.py / _MC2_EV, pg.z, pg.pz / _MC2_EV], axis=-1
    )
    return distributions.compute_moment_descriptor(phase_space, np.asarray(pg.weight) / e)


def _phase_space(pg):
    return np.stack([pg.x, pg.px / _MC2_EV, pg.y, pg.py / _MC2_EV, pg.z, pg.pz / _MC2_EV], axis=-1)


def test_a_synthetic_bunch_reproduces_the_descriptors_moments():
    descriptor = _descriptor_of(_lab_snapshot(n=20_000))

    synthetic = particles_from_descriptor(descriptor, n_particles=100_000, seed=3)

    again = distributions.compute_moment_descriptor(
        _phase_space(synthetic), np.asarray(synthetic.weight) / e
    )
    assert again["total_beam_charge_c"] == pytest.approx(descriptor["total_beam_charge_c"], rel=1e-9)
    assert again["mean_uz"] == pytest.approx(descriptor["mean_uz"], rel=0.01)
    for key in ("cov_uz_uz", "cov_x_x", "cov_y_y", "cov_ux_ux", "cov_uy_uy", "cov_z_z"):
        assert again[key] == pytest.approx(descriptor[key], rel=0.05), key
    for a, b, key in (("x", "ux", "cov_x_ux"), ("y", "uy", "cov_y_uy")):
        scale = np.sqrt(descriptor[f"cov_{a}_{a}"] * descriptor[f"cov_{b}_{b}"])
        assert abs(again[key] - descriptor[key]) < 0.02 * scale, key  # correlation within 0.02


def test_a_synthetic_bunch_is_in_the_bunch_frame_with_equal_charges():
    descriptor = _descriptor_of(_lab_snapshot())

    synthetic = particles_from_descriptor(descriptor, n_particles=1000)

    assert len(synthetic) == 1000
    assert np.allclose(np.asarray(synthetic.t), -np.asarray(synthetic.z) / c)
    assert np.all(np.asarray(synthetic.status) == 1)
    weight = np.asarray(synthetic.weight)
    assert np.allclose(weight, weight[0])
    assert weight.sum() == pytest.approx(descriptor["total_beam_charge_c"])
    assert np.asarray(synthetic.pz).min() / _MC2_EV >= 1.0  # no non-physical negative energies


def test_a_synthetic_bunch_depends_on_the_seed_only():
    descriptor = _descriptor_of(_lab_snapshot())

    a = particles_from_descriptor(descriptor, n_particles=500, seed=1)
    b = particles_from_descriptor(descriptor, n_particles=500, seed=1)
    other = particles_from_descriptor(descriptor, n_particles=500, seed=2)

    assert np.array_equal(np.asarray(a.x), np.asarray(b.x))
    assert not np.array_equal(np.asarray(a.x), np.asarray(other.x))


def test_energy_rises_along_z_when_the_descriptors_bins_do():
    descriptor = {**_descriptor_of(_lab_snapshot()), **{
        f"longitudinal_mean_uz_{i:02d}": m for i, m in enumerate((60.0, 90.0, 200.0, 400.0))
    }, **{f"longitudinal_rms_uz_{i:02d}": 25.0 for i in range(4)}}
    descriptor["mean_uz"] = 180.0
    descriptor["cov_uz_uz"] = 17_000.0

    synthetic = particles_from_descriptor(descriptor, n_particles=40_000, seed=0)

    z = np.asarray(synthetic.z)
    uz = np.asarray(synthetic.pz) / _MC2_EV
    assert np.corrcoef(z, uz)[0, 1] > 0.6  # the head (larger z) carries the high energies


def test_a_descriptor_missing_a_feature_is_an_error():
    descriptor = _descriptor_of(_lab_snapshot())
    del descriptor["cov_x_ux"]

    with pytest.raises(ValueError, match="cov_x_ux"):
        particles_from_descriptor(descriptor)


def test_slice_charge_fractions_reproduce_the_mean_and_variance():
    means, rms = np.array([62.0, 84.0, 205.0, 404.0]), np.array([23.0, 29.0, 72.0, 52.0])

    fractions = _slice_charge_fractions(means, rms, mean=171.6, variance=17_148.0)

    assert fractions.sum() == pytest.approx(1.0)
    assert np.all(fractions >= 0)
    assert fractions @ means == pytest.approx(171.6)
    assert fractions @ (rms**2 + means**2) - (fractions @ means) ** 2 == pytest.approx(17_148.0)


def test_inconsistent_bins_fall_back_to_equal_charge_with_a_warning():
    means, rms = np.array([10.0, 20.0, 30.0, 40.0]), np.array([1.0, 1.0, 1.0, 1.0])

    with pytest.warns(UserWarning, match="equal charge"):
        fractions = _slice_charge_fractions(means, rms, mean=500.0, variance=1.0)

    assert np.allclose(fractions, 0.25)

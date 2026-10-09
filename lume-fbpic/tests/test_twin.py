"""Tests for `TwinStage` (docs/htu-twin/twin/twin.py) against the real HTU twin.

Need `htu` (geecs-lume-twin) importable, for example with `PYTHONPATH` pointing at its checkout;
they are skipped otherwise. The twin builds slowly (about a second), so one is shared per module.
"""

from __future__ import annotations

import numpy
import pytest
from scipy.constants import c, e, m_e

pytest.importorskip("htu")
pytest.importorskip("lume_pva")

from lume.exceptions import ReadOnlyError  # noqa: E402
from lume.staged_model import StagedModel  # noqa: E402

from htu.model import build_htu_model  # noqa: E402
from tests.downramp_actions import make_actions  # noqa: E402
from lume_fbpic.model import LUMEFBPICModel  # noqa: E402
from twin import (  # noqa: E402
    TwinStage,
    beam_moments,
    binned_screen_geometries,
    build_chain,
    bunch_frame_particles,
    drift_particles,
)

from beamphysics import ParticleGroup

_MC2_EV = m_e * c**2 / e


def _lab_snapshot(n=4000, seed=0, uz_mean=150.0, uz_std=20.0) -> ParticleGroup:
    rng = numpy.random.default_rng(seed)
    return ParticleGroup(
        data={
            "x": rng.normal(0.0, 3.0e-6, n),
            "y": rng.normal(0.0, 1.5e-6, n),
            "z": rng.normal(0.003, 4.0e-6, n),
            "px": rng.normal(0.0, 2.0, n) * _MC2_EV,
            "py": rng.normal(0.0, 1.0, n) * _MC2_EV,
            "pz": rng.normal(uz_mean, uz_std, n) * _MC2_EV,
            "t": numpy.zeros(n),
            "status": numpy.ones(n, dtype=int),
            "weight": numpy.full(n, 1.0e3 * e),
            "species": "electron",
        }
    )


@pytest.fixture(scope="module")
def twin_model():
    return build_htu_model()


@pytest.fixture()
def stage(twin_model):
    stage = TwinStage(twin_model)
    yield stage
    stage.reset()  # the twin model is shared: put it back for the next test


def _get(stage, name):
    return float(stage.get([name])[name])


def test_a_lab_snapshot_is_selected_and_injected(stage):
    pg = _lab_snapshot()
    expected = bunch_frame_particles(pg, uz_min=30.0, central_fraction=0.95)

    stage.initial_particles = pg

    assert stage.external_beam is True
    assert len(stage.initial_particles) == len(expected)
    assert _get(stage, "Source_Charge_pC") == pytest.approx(expected.charge * 1e12)
    assert _get(stage, "Source_NumParticles") == len(expected)


def test_source_pvs_read_the_injected_bunchs_moments(stage):
    pg = _lab_snapshot(uz_mean=150.0)

    stage.initial_particles = pg

    gamma = numpy.sqrt(1 + 150.0**2)
    assert _get(stage, "Source_Energy_MeV") == pytest.approx(
        gamma * _MC2_EV / 1e6, rel=0.02
    )
    assert _get(stage, "Source_EnergySpread_pct") == pytest.approx(13.0, abs=2.0)


def test_screen_images_carry_the_injected_charge(stage):
    stage.initial_particles = _lab_snapshot()

    image = numpy.asarray(stage.get(["TCPhosphor_image"])["TCPhosphor_image"])

    assert image.sum() > 0
    assert image.sum() == pytest.approx(
        _get(stage, "Source_Charge_pC") * 1e-12, rel=0.3
    )


def test_particles_that_already_have_arrival_times_are_used_as_they_are(stage):
    bunch = bunch_frame_particles(_lab_snapshot(), uz_min=30.0, central_fraction=0.9)

    stage.initial_particles = bunch

    assert len(stage.initial_particles) == len(bunch)  # no second selection


def test_a_failed_injection_leaves_the_twin_untouched(stage):
    before = (_get(stage, "Source_Energy_MeV"), _get(stage, "Source_Charge_pC"))

    with pytest.raises(ValueError):
        stage.initial_particles = _lab_snapshot(
            uz_mean=2.0, uz_std=0.2
        )  # nothing passes uz >= 30

    assert (_get(stage, "Source_Energy_MeV"), _get(stage, "Source_Charge_pC")) == before
    assert stage.external_beam is False


def test_an_unlocked_source_put_replaces_the_injected_beam(twin_model):
    stage = TwinStage(twin_model, lock_source=False)
    try:
        stage.initial_particles = _lab_snapshot()
        assert stage.external_beam is True

        stage.set({"Source_Energy_MeV": 90.0})

        assert stage.external_beam is False
        assert _get(stage, "Source_Energy_MeV") == 90.0
    finally:
        stage.reset()


def test_a_locked_source_cannot_be_set_and_the_injected_beam_stays(stage):
    stage.initial_particles = _lab_snapshot()
    charge = _get(stage, "Source_Charge_pC")
    assert charge == pytest.approx(
        stage.initial_particles.charge * 1e12
    )  # it reads the bunch

    with pytest.raises(ReadOnlyError):
        stage.set({"Source_Charge_pC": 5.0})
    stage.set({"EMQ1H_Current": 0.71})  # the other twin variables stay writable

    assert stage.external_beam is True
    assert _get(stage, "Source_Charge_pC") == charge
    assert _get(stage, "EMQ1H_Current") == pytest.approx(0.71, rel=1e-3)


def test_the_source_can_be_left_writable(twin_model):
    stage = TwinStage(twin_model, lock_source=False)

    assert not stage.supported_variables["Source_Energy_MeV"].read_only


def test_reset_restores_the_original_source_beam_and_forgets_the_drift(stage):
    original = (_get(stage, "Source_Energy_MeV"), _get(stage, "Source_Charge_pC"))
    stage.plasma_exit_z = 0.0029
    stage.initial_particles = _lab_snapshot()
    assert stage.drift_length is not None

    stage.reset()

    assert stage.external_beam is False
    assert stage.initial_particles is None
    assert stage.drift_length is None
    assert (
        _get(stage, "Source_Energy_MeV"),
        _get(stage, "Source_Charge_pC"),
    ) == pytest.approx(original)


def test_wrapped_variables_are_the_twins_apart_from_the_locked_source(
    stage, twin_model
):
    twin = twin_model.supported_variables
    assert list(stage.supported_variables) == list(twin)
    sources = [name for name in stage.supported_variables if name.startswith("Source_")]
    assert len(sources) == 14
    assert all(stage.supported_variables[name].read_only for name in sources)
    for name, variable in stage.supported_variables.items():
        if name.startswith("Source_"):
            assert (
                variable.model_copy(update={"read_only": twin[name].read_only})
                == twin[name]
            )
        else:
            assert (
                variable is twin[name]
            )  # the rest is the twin's own, writable as it is


def test_the_stage_follows_an_lpa_model_in_a_staged_chain(stage, simulator):
    simulator.final_particles = _lab_snapshot()
    lpa = LUMEFBPICModel(simulator, make_actions(simulator), dummy_run=True)
    chain = StagedModel([lpa, stage])

    with pytest.raises(ReadOnlyError):  # the LPA is the source: a source put is refused
        chain.set({"Source_Energy_MeV": 50.0})
    assert chain.supported_variables["Source_Energy_MeV"].read_only

    chain.set(
        {"EMQ1H_Current": 0.7}
    )  # only a twin variable: the LPA bunch is still passed on

    assert stage.external_beam is True
    assert _get(stage, "EMQ1H_Current") == pytest.approx(0.7, rel=1e-3)
    assert chain.get(["Source_Charge_pC"])["Source_Charge_pC"] > 0


def test_without_a_plasma_exit_plane_the_bunch_is_not_drifted(stage):
    stage.initial_particles = _lab_snapshot()

    assert stage.drift_length is None


def test_the_bunch_is_drifted_to_the_plasma_exit_plane(twin_model):
    stage = TwinStage(
        twin_model, plasma_exit_z=0.0029
    )  # the snapshot's mean z is 0.003
    try:
        pg = _lab_snapshot()
        expected_selection = bunch_frame_particles(
            pg, uz_min=30.0, central_fraction=0.95
        )
        z_mean = float(
            numpy.average(expected_selection.z, weights=expected_selection.weight)
        )

        stage.initial_particles = pg

        assert stage.drift_length == pytest.approx(0.0029 - z_mean)
        assert stage.drift_length < 0  # the bunch was past the exit: moved back
        injected = stage.initial_particles
        assert float(
            numpy.average(injected.z, weights=injected.weight)
        ) == pytest.approx(0.0029)
        expected = drift_particles(expected_selection, stage.drift_length)
        numpy.testing.assert_allclose(
            numpy.asarray(injected.x), numpy.asarray(expected.x)
        )
        moments = beam_moments(injected)  # the source PVs read the bunch at the plane
        assert _get(stage, "Source_BetaX_mm") == pytest.approx(
            moments["beta_x_mm"], rel=1e-4
        )
        assert _get(stage, "Source_X_um") == pytest.approx(moments["x_um"], abs=1e-3)
    finally:
        stage.reset()


def test_a_bunch_that_has_not_reached_the_plane_is_drifted_with_a_warning(twin_model):
    stage = TwinStage(
        twin_model, plasma_exit_z=0.0031
    )  # beyond the snapshot's z of 0.003
    try:
        with pytest.warns(UserWarning, match="short of the plasma exit"):
            stage.initial_particles = _lab_snapshot()

        assert stage.drift_length > 0
    finally:
        stage.reset()


def test_the_served_chain_exposes_the_twin_controls_and_keeps_the_source_read_only(
    simulator, twin_model
):
    from lume_pva.runner import PutMode

    from serve import build_config

    simulator.final_particles = _lab_snapshot()
    chain = build_chain(
        LUMEFBPICModel(simulator, make_actions(simulator), dummy_run=True), twin_model
    )
    stage = chain.lume_model_instances[1]
    try:
        config = build_config(
            chain,
            prefix="HTU:SIM:",
            serve_always=set(stage.supported_variables),
            wait_for_puts=True,
        )
        modes = {
            name: str(entry["mode"]) for name, entry in config["variables"].items()
        }

        assert modes["EMQ1H_Current"] == "rw"  # a twin control, served read-write
        assert modes["Source_Energy_MeV"] == "ro"  # the LPA is the source
        assert modes["charge_pc"] == "ro"  # an LPA output
        assert "laser_energy" not in modes  # an LPA input is not served
        assert config["put_mode"] == PutMode.Complete
        assert config["prefix"] == "HTU:SIM:"
    finally:
        stage.reset()


def test_tracking_after_an_injection_is_deferred_until_the_twin_is_read(
    stage, twin_model, mocker
):
    track = mocker.spy(twin_model.simulator, "track")

    stage.initial_particles = _lab_snapshot()
    assert track.call_count == 0

    _get(stage, "Source_Charge_pC")
    assert track.call_count == 1
    _get(stage, "Source_Energy_MeV")
    assert track.call_count == 1  # tracked once, not on every read


def test_a_set_after_an_injection_tracks_once(stage, twin_model, mocker):
    track = mocker.spy(twin_model.simulator, "track")

    stage.initial_particles = _lab_snapshot()
    stage.set({"EMQ1H_Current": 0.7})

    assert (
        track.call_count == 1
    )  # the twin's own re-track; not a second one for the injection
    assert stage.external_beam is True


def test_screens_are_current_when_read_right_after_an_injection(stage):
    stage.initial_particles = _lab_snapshot()

    image = numpy.asarray(stage.get(["TCPhosphor_image"])["TCPhosphor_image"])

    assert image.sum() > 0


def test_switching_the_lpa_run_changes_the_twins_source(
    simulator, twin_model, tmp_path
):
    from tests.downramp_actions import make_actions
    from lume_fbpic.actions import make_descriptor_actions
    from selector import ArchiveSelector
    from inversion_fbpic.utils import distributions

    models = {}
    descriptors = {}
    for name, uz, charge_pc in (("low", 100.0, 200.0), ("high", 200.0, 400.0)):
        pg = _lab_snapshot(uz_mean=uz)
        phase_space = numpy.stack(
            [pg.x, pg.px / _MC2_EV, pg.y, pg.py / _MC2_EV, pg.z, pg.pz / _MC2_EV],
            axis=-1,
        )
        weights = numpy.full(len(pg), charge_pc * 1e-12 / len(pg)) / e
        descriptor = distributions.compute_moment_descriptor(phase_space, weights)
        descriptors[name] = descriptor
        source = LUMEFBPICModel(
            simulator,
            [*make_actions(simulator), *make_descriptor_actions()],
            dummy_run=True,
        )
        source.simulator.stats = {f"descriptor_{k}": v for k, v in descriptor.items()}
        source.archive(tmp_path / f"{name}.h5")
        models[name] = LUMEFBPICModel.from_archive(
            tmp_path / f"{name}.h5", dummy_run=True
        )
    chain = build_chain(
        ArchiveSelector(models, synthetic_bunch_particles=4000), twin_model
    )
    stage = chain.lume_model_instances[1]
    try:
        chain.set({"EMQ1H_Current": 0.7})
        low = (_get(stage, "Source_Charge_pC"), _get(stage, "Source_Energy_MeV"))
        assert len(stage.initial_particles) == 4000  # the selected bunch: no second cut
        assert low[1] == pytest.approx(
            numpy.sqrt(1 + descriptors["low"]["mean_uz"] ** 2) * _MC2_EV / 1e6, rel=0.05
        )

        chain.set({"LPA_Archive": "high"})  # only the selector: the twin must follow
        high = (_get(stage, "Source_Charge_pC"), _get(stage, "Source_Energy_MeV"))

        assert low[0] == pytest.approx(200.0, rel=1e-3) and high[0] == pytest.approx(
            400.0, rel=1e-3
        )
        assert high[1] > 1.5 * low[1]
        assert chain.get(["LPA_Archive"])["LPA_Archive"] == "high"
        assert chain.get(["descriptor_total_beam_charge_pc"])[
            "descriptor_total_beam_charge_pc"
        ] == pytest.approx(400.0)
        assert stage.external_beam is True
    finally:
        stage.reset()


def test_binned_geometries_keep_the_field_of_view_and_coarsen_the_pixels():
    from htu.lattice import DEFAULT_SCREEN_GEOMETRY_OVERRIDES, SCREEN_NAMES
    from htu.screens import DEFAULT_GEOMETRY

    geometries = binned_screen_geometries(4)

    assert set(geometries) == set(SCREEN_NAMES)
    for name, geometry in geometries.items():
        base = DEFAULT_SCREEN_GEOMETRY_OVERRIDES.get(name, DEFAULT_GEOMETRY)
        assert (geometry.size_x, geometry.size_y) == (
            base.size_x // 4,
            base.size_y // 4,
        )
        assert geometry.calibration_m == pytest.approx(base.calibration_m * 4)
        assert geometry.size_x * geometry.calibration_m == pytest.approx(
            base.size_x * base.calibration_m
        )
    assert geometries["ChicaneSlit"].calibration_m == pytest.approx(
        80e-6
    )  # its own 20 um x 4


def test_a_binning_that_is_not_a_positive_divisor_is_refused():
    for factor in (0, -2, 1.5, 3):
        with pytest.raises(ValueError):
            binned_screen_geometries(factor)


def test_a_given_model_cannot_be_binned(twin_model):
    with pytest.raises(ValueError, match="builds the twin model"):
        TwinStage(twin_model, screen_binning=4)


def test_a_binned_stage_serves_smaller_images_with_the_same_charge(stage):
    binned = TwinStage(screen_binning=4)
    try:
        pg = _lab_snapshot()
        stage.initial_particles = pg
        binned.initial_particles = pg

        full = numpy.asarray(stage.get(["TCPhosphor_image"])["TCPhosphor_image"])
        coarse = numpy.asarray(binned.get(["TCPhosphor_image"])["TCPhosphor_image"])

        assert full.shape == (1024, 1024) and coarse.shape == (256, 256)
        assert coarse.sum() == pytest.approx(
            full.sum(), rel=1e-3
        )  # same charge in view
        occupied = lambda image: numpy.count_nonzero(image)  # noqa: E731
        assert occupied(coarse) < occupied(full)  # fewer, fuller pixels
        assert coarse.max() > full.max()
    finally:
        binned.reset()

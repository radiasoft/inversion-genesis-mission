"""
Tests for laser profile physics: extents, energy/a0 round-trips,
build_laser_profile polarization variants, and YAML round-trips.

Energy/a0 XOR validation tests live in ``test_serializable_config.py``.

Run from Simulation_FBPIC::

    pytest tests/test_lib/test_laser.py -v
"""

from __future__ import annotations

from math import pi

import numpy as np
import pytest
from scipy.constants import c, epsilon_0

LASER_BASE_KWARGS: dict = {
    "z0": -3.0e-5,
    "wavelength": 8.0e-7,
    "tau_fwhm": 3.8e-14,
    "cep": 0.0,
    "waist": 2.8e-5,
    "focal_position": 3.0e-3,
    "polarization": 0.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_laser(**overrides):
    from inversion_fbpic.lib.laser import GaussianLaserPulse

    kw = {**LASER_BASE_KWARGS, **overrides}
    if "energy" not in kw and "a0" not in kw:
        kw["energy"] = 5.0
    return GaussianLaserPulse(**kw)


def _integrate_laser_energy_from_fields(
    pulse,
    *,
    n_r: int = 128,
    n_z: int = 256,
    n_t_optical: int = 32,
    num_sigma: float = 6.0,
) -> float:
    """Integrate cycle-averaged EM energy density from ``E_field`` profiles.

    At each spatial sample point the squared field magnitude is averaged over
    one optical cycle, then integrated over cylindrical volume using
    ``epsilon_0 * <E^2>`` (total electric plus magnetic energy in vacuum).
    """
    from inversion_fbpic.lib.laser import _ensure_profile_list

    profiles = _ensure_profile_list(pulse.build_laser_profile())
    z_min, z_max = pulse.get_z_extent(num_sigma=num_sigma)
    r_max = pulse.get_r_extent((z_min, z_max), num_sigma=num_sigma)

    r = np.linspace(0.0, r_max, n_r)
    z = np.linspace(z_min, z_max, n_z)
    dr = float(np.diff(r).mean())
    dz = float(np.diff(z).mean())
    optical_period = pulse.wavelength / c
    t_samples = np.linspace(0.0, optical_period, n_t_optical, endpoint=False)

    r_grid, z_grid = np.meshgrid(r, z, indexing="ij")
    x_grid = r_grid
    y_grid = np.zeros_like(r_grid)

    e2_sum = np.zeros_like(r_grid)
    for ti in t_samples:
        ex = np.zeros_like(r_grid)
        ey = np.zeros_like(r_grid)
        for profile in profiles:
            ex_p, ey_p = profile.E_field(x=x_grid, y=y_grid, z=z_grid, t=ti)
            ex += ex_p
            ey += ey_p
        e2_sum += ex**2 + ey**2
    e2_avg = e2_sum / n_t_optical

    volume_element = 2.0 * pi * r_grid * dr * dz
    volume_element[0, :] = pi * dr**2 / 4.0 * dz

    return float(epsilon_0 * np.sum(e2_avg * volume_element))


# Well-behaved laser parameters: long pulse and wide waist vs wavelength.
_ENERGY_INTEGRATION_KWARGS: dict = {
    "z0": 0.0,
    "wavelength": 8.0e-7,
    "tau_fwhm": 200.0e-15,
    "cep": 0.0,
    "waist": 50.0e-6,
    "focal_position": 0.0,
}

# ===================================================================
# Energy / a0 round-trip consistency
# ===================================================================


class TestEnergyA0:
    def test_energy_to_a0_round_trip(self) -> None:
        pulse = _make_laser(energy=5.0)
        from inversion_fbpic.lib.laser import GaussianLaserPulse

        pulse2 = GaussianLaserPulse(a0=pulse.a0, **LASER_BASE_KWARGS)
        assert pulse2.energy == pytest.approx(5.0, rel=1e-6)

    def test_a0_to_energy_round_trip(self) -> None:
        pulse = _make_laser(a0=2.0)
        from inversion_fbpic.lib.laser import GaussianLaserPulse

        pulse2 = GaussianLaserPulse(energy=pulse.energy, **LASER_BASE_KWARGS)
        assert pulse2.a0 == pytest.approx(2.0, rel=1e-6)

    def test_negative_energy_raises(self) -> None:
        from inversion_fbpic.lib.laser import GaussianLaserPulse

        with pytest.raises(ValueError, match="energy must be > 0"):
            GaussianLaserPulse(energy=-1.0, **LASER_BASE_KWARGS)

    def test_negative_a0_raises(self) -> None:
        from inversion_fbpic.lib.laser import GaussianLaserPulse

        with pytest.raises(ValueError, match="a0 must be > 0"):
            GaussianLaserPulse(a0=-1.0, **LASER_BASE_KWARGS)


# ===================================================================
# Field-intensity energy integration
# ===================================================================


class TestEnergyFromFieldIntegration:
    @pytest.mark.parametrize(
        ("polarization", "label"),
        [
            (0.0, "linear"),
            (pi / 4, "linear_45deg"),
            ("left", "left_circular"),
            ("right", "right_circular"),
            ([pi / 4, pi / 2], "elliptical"),
        ],
    )
    def test_energy_from_field_integration(self, polarization, label) -> None:
        target_energy = 5.0
        pulse = _make_laser(
            energy=target_energy,
            polarization=polarization,
            **_ENERGY_INTEGRATION_KWARGS,
        )
        integrated_energy = _integrate_laser_energy_from_fields(pulse)
        assert integrated_energy == pytest.approx(target_energy, rel=2e-2), label


# ===================================================================
# Longitudinal extent
# ===================================================================


class TestGetZExtent:
    def test_symmetric_about_z0(self) -> None:
        pulse = _make_laser(energy=5.0)
        z_min, z_max = pulse.get_z_extent()
        center = (z_min + z_max) / 2.0
        assert center == pytest.approx(pulse.z0, rel=1e-6)

    def test_wider_with_more_sigma(self) -> None:
        pulse = _make_laser(energy=5.0)
        extent_3 = pulse.get_z_extent(num_sigma=3.0)
        extent_5 = pulse.get_z_extent(num_sigma=5.0)
        width_3 = extent_3[1] - extent_3[0]
        width_5 = extent_5[1] - extent_5[0]
        assert width_5 > width_3

    def test_scales_with_tau_fwhm(self) -> None:
        pulse_short = _make_laser(energy=5.0, tau_fwhm=3.0e-14)
        pulse_long = _make_laser(energy=5.0, tau_fwhm=6.0e-14)
        width_short = pulse_short.get_z_extent()[1] - pulse_short.get_z_extent()[0]
        width_long = pulse_long.get_z_extent()[1] - pulse_long.get_z_extent()[0]
        assert width_long > width_short


# ===================================================================
# Radial extent
# ===================================================================


class TestGetRExtent:
    def test_positive(self) -> None:
        pulse = _make_laser(energy=5.0)
        sim_extent = (-1e-4, 0.0)
        r_ext = pulse.get_r_extent(sim_extent)
        assert r_ext > 0.0

    def test_wider_when_focal_far_from_box(self) -> None:
        pulse_near = _make_laser(energy=5.0, focal_position=0.0)
        pulse_far = _make_laser(energy=5.0, focal_position=0.01)
        sim_extent = (-1e-4, 0.0)
        r_near = pulse_near.get_r_extent(sim_extent)
        r_far = pulse_far.get_r_extent(sim_extent)
        assert r_far > r_near

    def test_rayleigh_scaling(self) -> None:
        pulse = _make_laser(energy=5.0)
        rayleigh = pi * pulse.waist**2 / pulse.wavelength
        assert rayleigh > 0.0
        r_ext = pulse.get_r_extent((-1e-4, 0.0))
        assert r_ext > pulse.waist / 2.0


# ===================================================================
# build_laser_profile — polarization variants
# ===================================================================


class TestBuildLaserProfile:
    def test_linear_returns_single(self) -> None:
        pulse = _make_laser(energy=5.0, polarization=0.0)
        profile = pulse.build_laser_profile()
        from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser

        assert isinstance(profile, GaussianLaser)

    def test_linear_45_returns_single(self) -> None:
        pulse = _make_laser(energy=5.0, polarization=pi / 4)
        profile = pulse.build_laser_profile()
        from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser

        assert isinstance(profile, GaussianLaser)

    def test_left_circular_returns_two_profiles(self) -> None:
        pulse = _make_laser(energy=5.0, polarization="left")
        profiles = pulse.build_laser_profile()
        assert isinstance(profiles, list)
        assert len(profiles) == 2
        total_E0_sq = sum(p.E0x**2 + p.E0y**2 for p in profiles)
        assert total_E0_sq > 0

    def test_right_circular_returns_two_profiles(self) -> None:
        pulse = _make_laser(energy=5.0, polarization="right")
        profiles = pulse.build_laser_profile()
        assert isinstance(profiles, list)
        assert len(profiles) == 2
        total_E0_sq = sum(p.E0x**2 + p.E0y**2 for p in profiles)
        assert total_E0_sq > 0

    def test_elliptical_returns_two_profiles(self) -> None:
        pulse = _make_laser(energy=5.0, polarization=[pi / 4, pi / 2])
        profiles = pulse.build_laser_profile()
        assert isinstance(profiles, list)
        assert len(profiles) == 2

    def test_profile_has_correct_waist(self) -> None:
        pulse = _make_laser(energy=5.0, polarization=0.0)
        profile = pulse.build_laser_profile()
        assert profile.transverse_profile.w0 == pytest.approx(pulse.waist)


# ===================================================================
# YAML round-trip
# ===================================================================


class TestLaserYamlRoundTrip:
    def test_energy_based_round_trip(self) -> None:
        pulse = _make_laser(energy=5.0)
        yaml_str = pulse.to_yaml(comments=False)

        from inversion_fbpic.lib.laser import GaussianLaserPulse

        reloaded = GaussianLaserPulse.from_yaml(yaml_str)
        assert reloaded.energy == pytest.approx(pulse.energy, rel=1e-6)
        assert reloaded.a0 == pytest.approx(pulse.a0, rel=1e-6)

    def test_a0_based_round_trip(self) -> None:
        pulse = _make_laser(a0=2.0)
        yaml_str = pulse.to_yaml(comments=False)

        from inversion_fbpic.lib.laser import GaussianLaserPulse

        reloaded = GaussianLaserPulse.from_yaml(yaml_str)
        assert reloaded.a0 == pytest.approx(pulse.a0, rel=1e-6)
        assert reloaded.energy == pytest.approx(pulse.energy, rel=1e-6)

    def test_circular_polarization_round_trip(self) -> None:
        pulse = _make_laser(energy=5.0, polarization="left")
        yaml_str = pulse.to_yaml(comments=False)

        from inversion_fbpic.lib.laser import GaussianLaserPulse

        reloaded = GaussianLaserPulse.from_yaml(yaml_str)
        assert reloaded.polarization == "left"

    def test_elliptical_polarization_round_trip(self) -> None:
        pulse = _make_laser(energy=5.0, polarization=[pi / 4, pi / 2])
        yaml_str = pulse.to_yaml(comments=False)

        from inversion_fbpic.lib.laser import GaussianLaserPulse

        reloaded = GaussianLaserPulse.from_yaml(yaml_str)
        assert reloaded.polarization == pytest.approx([pi / 4, pi / 2])


# ===================================================================
# LasyLaserPulse
# ===================================================================

LASY_BASE_KWARGS: dict = {
    "energy": 1.0,
    "z0": -1.0e-4,
    "z0_antenna": 0.0,
    "wavelength": 0.8e-6,
    "tau_fwhm": 30e-15,
    "waist": 20e-6,
    "focal_position": 1.0e-3,
    "super_gaussian_order": 2.0,
}

# Small LASY grid so integration tests build in well under a second.
LASY_TINY_GRID: dict = {
    "n_azimuthal_modes": 1,
    "num_points": (48, 96),
    "centering_angles": 8,
}


def _make_lasy(**overrides):
    from inversion_fbpic.lib.laser import LasyLaserPulse

    return LasyLaserPulse(**{**LASY_BASE_KWARGS, **overrides})


class TestLasyLaserPulse:
    def test_registered_in_concrete_registry(self) -> None:
        from inversion_fbpic.lib.laser import _LaserPulse

        assert "lasy" in _LaserPulse._CONCRETE_REGISTRY

    def test_defaults_map_to_high_order_parameters(self) -> None:
        from inversion_fbpic.utils.laser import ZERNIKE_OSA_INDICES, HighOrderLasyLaser

        pulse = _make_lasy()
        physical = pulse.physical_parameters
        assert set(physical) == HighOrderLasyLaser.PHYSICAL_PARAMETER_KEYS
        assert physical["laser_wavelength_m"] == 0.8e-6
        assert physical["laser_energy_J"] == 1.0
        assert physical["laser_pulse_duration_fwhm_s"] == 30e-15
        assert physical["laser_spot_size_m"] == 20e-6
        assert physical["laser_super_gaussian_order"] == 2.0
        assert physical["laser_focal_position_m"] == 1.0e-3
        for name in ZERNIKE_OSA_INDICES:
            assert physical[f"zernike_{name}"] == 0.0
        assert pulse.hyperparameters == HighOrderLasyLaser._HYPERPARAMETER_DEFAULTS

    def test_post_init_forces_antenna_and_defers_a0(self) -> None:
        pulse = _make_lasy()
        assert pulse.method == "antenna"
        assert pulse.v_antenna == 0.0
        assert pulse._amplitude_source == "energy"
        assert pulse.a0 is None
        assert pulse.out_a0 is None
        assert pulse.out_energy is None
        assert not pulse.is_prepared
        assert isinstance(hash(pulse), int)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"a0": 1.0}, "do not pass a0"),
            ({"method": "direct"}, "antenna"),
            ({"v_antenna": 1.0}, "stationary antenna"),
            ({"z0_antenna": None}, "z0_antenna is required"),
            ({"energy": None}, "energy must be provided"),
            ({"energy": -1.0}, "energy must be provided"),
            ({"zernike_coefficients": {"bogus": 1.0}}, "Unknown zernike_coefficients"),
            ({"polarization": [1.0, 0.0, 0.0]}, "polarization"),
            ({"num_points": [10]}, "num_points"),
        ],
    )
    def test_invalid_inputs_raise(self, overrides: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _make_lasy(**overrides)

    def test_optional_spectral_fields_are_forwarded_only_when_set(self) -> None:
        from inversion_fbpic.utils.laser import HighOrderLasyLaser

        pulse = _make_lasy()
        assert not (
            set(pulse.physical_parameters)
            & HighOrderLasyLaser.OPTIONAL_PHYSICAL_PARAMETER_KEYS
        )

        pulse = _make_lasy(
            spectral_bandwidth="auto",
            cep=0.3,
            gdd_relative=0.5,
            fod=1e-56,
            peak_delay_from_file_start=100e-15,
            maximum_pulse_duration_fwhm=200e-15,
        )
        physical = pulse.physical_parameters
        assert physical["laser_spectral_bandwidth_rad_s"] == "auto"
        assert physical["laser_cep_phase_rad"] == 0.3
        assert physical["laser_gdd_relative"] == 0.5
        assert physical["laser_fod_s4"] == 1e-56
        assert "laser_gdd_s2" not in physical and "laser_tod_s3" not in physical
        hyper = pulse.hyperparameters
        assert hyper["peak_delay_from_file_start_s"] == 100e-15
        assert hyper["maximum_pulse_duration_fwhm_s"] == 200e-15
        assert hyper["spectral_time_window_factor"] == 6.0

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"centering_angles": 3, "n_azimuthal_modes": 5}, "centering_angles"),
            (
                {"gdd": 1e-28, "gdd_relative": 0.5},
                "either laser_gdd_s2 or laser_gdd_relative",
            ),
            ({"gdd": 1e-28}, "requires a nonzero laser_spectral_bandwidth"),
        ],
    )
    def test_physical_validation_is_delegated_to_high_order_laser(
        self, tmp_path, overrides: dict, match: str
    ) -> None:
        """These are HighOrderLasyLaser's rules; the wrapper surfaces them at prepare()."""
        pulse = _make_lasy(lasy_file=tmp_path / "x", **overrides)
        with pytest.raises(ValueError, match=match):
            pulse.prepare(None)

    def test_partial_zernike_dict_is_completed(self) -> None:
        from inversion_fbpic.utils.laser import ZERNIKE_OSA_INDICES

        pulse = _make_lasy(zernike_coefficients={"coma_x": -0.25})
        assert set(pulse.zernike_coefficients) == set(ZERNIKE_OSA_INDICES)
        assert pulse.zernike_coefficients["coma_x"] == -0.25
        assert all(
            v == 0.0 for k, v in pulse.zernike_coefficients.items() if k != "coma_x"
        )
        assert pulse.physical_parameters["zernike_coma_x"] == -0.25

    def test_resolve_before_prepare(self) -> None:
        pulse = _make_lasy()
        assert pulse.resolve_laser_energy() == 1.0
        with pytest.raises(RuntimeError, match="prepare"):
            pulse.resolve_laser_a0()

    def test_yaml_round_trip_before_build(self) -> None:
        from pathlib import Path

        from inversion_fbpic.lib.laser import LasyLaserPulse
        from inversion_fbpic.lib.serializable_config import SerializableConfig

        pulse = _make_lasy(
            zernike_coefficients={"coma_x": -0.25}, polarization=[0.0, 1.0]
        )
        yaml_str = pulse.to_yaml(comments=False)
        assert "out_a0: null" in yaml_str
        assert "\n  a0:" not in yaml_str

        reloaded = LasyLaserPulse.from_yaml(yaml_str)
        assert reloaded == pulse
        assert reloaded.polarization == (0.0, 1.0)
        assert reloaded.num_points == (600, 900)
        assert reloaded.lasy_file == Path("diags/lasy_laser")
        assert reloaded.zernike_coefficients == pulse.zernike_coefficients
        assert hash(reloaded) == hash(pulse)

        generic = SerializableConfig.from_dict(pulse.to_dict())
        assert isinstance(generic, LasyLaserPulse)
        assert generic == pulse

        compact = pulse.to_dict(include_nones=False)["parameters"]
        assert "out_a0" not in compact
        assert "a0" not in compact

    def test_lasy_file_is_written_verbatim_in_config_files(self, tmp_path) -> None:
        """A relative output prefix must survive saving into a sub-directory."""
        from pathlib import Path

        from inversion_fbpic.lib.laser import LasyLaserPulse

        pulse = _make_lasy(lasy_file=Path("diags/lasy_laser"))
        cfg = tmp_path / "cfgs" / "laser.yaml"
        pulse.to_yaml_file(cfg, comments=False)
        assert "lasy_file: diags/lasy_laser" in cfg.read_text()
        reloaded = LasyLaserPulse.from_file(cfg)
        assert reloaded.lasy_file == Path("diags/lasy_laser")

        absolute = _make_lasy(lasy_file=tmp_path / "abs_prefix")
        assert (
            absolute.to_dict()["parameters"]["lasy_file"]
            == (tmp_path / "abs_prefix").as_posix()
        )

    def test_extents(self) -> None:
        from inversion_fbpic.lib.laser import _gaussian_r_extent

        pulse = _make_lasy()
        z_min, z_max = pulse.get_z_extent()
        assert z_min < pulse.z0 < z_max
        assert z_max - pulse.z0 == pytest.approx(pulse.z0 - z_min)

        extent = (-1.0e-4, 0.0)
        assert pulse.get_r_extent(extent) == pytest.approx(
            _gaussian_r_extent(20e-6, 0.8e-6, 1.0e-3, extent, 3.0)
        )

    def test_resolve_comm_accepts_none_fbpic_and_mpi4py_forms(self) -> None:
        from types import SimpleNamespace

        from inversion_fbpic.lib._laser_implementations.lasy_laser import _resolve_comm

        assert _resolve_comm(None) == (0, None)

        fbpic_serial = SimpleNamespace(rank=0, mpi_comm=None)
        assert _resolve_comm(fbpic_serial) == (0, None)
        mpi = SimpleNamespace(Get_rank=lambda: 2, Get_size=lambda: 4)
        fbpic_parallel = SimpleNamespace(rank=2, mpi_comm=mpi)
        assert _resolve_comm(fbpic_parallel) == (2, mpi)
        assert _resolve_comm(mpi) == (2, mpi)

        with pytest.raises(TypeError, match="comm must be"):
            _resolve_comm(object())

    def test_base_prepare_is_noop(self) -> None:
        pulse = _make_laser(energy=5.0)
        assert pulse.prepare(None) is None
        assert pulse.build_laser_profile() is not None


class _FakeComm:
    """mpi4py-shaped stand-in: records broadcasts; non-root ranks get ``incoming``."""

    def __init__(self, rank: int = 0, incoming=None) -> None:
        self.rank = rank
        self.incoming = incoming
        self.broadcasts: list = []

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return 2

    def bcast(self, obj, root=0):
        self.broadcasts.append((obj, root))
        return obj if self.rank == root else self.incoming


@pytest.mark.integration
class TestLasyLaserPulseBuild:
    def test_prepare_writes_file_and_measures_a0(self, tmp_path) -> None:
        from fbpic.lpa_utils.laser.laser_profiles import FromLasyFileLaser

        from inversion_fbpic.utils.simulation_setup_tools import (
            calculate_laser_a0_from_energy,
        )

        pulse = _make_lasy(lasy_file=tmp_path / "lasy_laser", **LASY_TINY_GRID)
        hash_before = hash(pulse)

        pulse.prepare(None)

        expected_file = tmp_path / "lasy_laser_00000.h5"
        assert expected_file.is_file()
        assert pulse.lasy_file_path == expected_file.resolve()
        assert pulse.is_prepared
        assert pulse.out_a0 is not None and pulse.out_a0 > 0
        assert pulse.a0 == pulse.out_a0 == pulse.resolve_laser_a0()
        # Order 2 is Gaussian, so the numeric focus a0 should match the analytic one.
        assert pulse.out_a0 == pytest.approx(
            calculate_laser_a0_from_energy(1.0, 0.8, 20e-6, 30e-15), rel=0.05
        )
        assert hash(pulse) == hash_before

        mtime = expected_file.stat().st_mtime_ns
        pulse.prepare(None)
        assert expected_file.stat().st_mtime_ns == mtime

        profile = pulse.build_laser_profile()
        assert isinstance(profile, FromLasyFileLaser)
        assert profile.t_start == 0.0

        yaml_str = pulse.to_yaml(comments=False)
        assert "out_a0: null" not in yaml_str
        assert f"out_a0: {pulse.out_a0}" in yaml_str

    def test_relative_lasy_file_anchors_at_relative_to(
        self, tmp_path, monkeypatch
    ) -> None:
        from pathlib import Path

        pulse = _make_lasy(lasy_file=Path("diags/lasy_laser"), **LASY_TINY_GRID)
        assert (
            pulse.resolve_lasy_file(tmp_path)
            == (tmp_path / "diags/lasy_laser").resolve()
        )

        # Without an anchor, the CWD is used.
        cwd_anchor = tmp_path / "cwd_anchor"
        cwd_anchor.mkdir()
        monkeypatch.chdir(cwd_anchor)
        assert pulse.resolve_lasy_file() == (cwd_anchor / "diags/lasy_laser").resolve()

        work_dir = tmp_path / "run"
        pulse.prepare(None, relative_to=work_dir)
        assert (
            pulse.lasy_file_path == (work_dir / "diags/lasy_laser_00000.h5").resolve()
        )
        assert pulse.lasy_file_path.is_file()
        assert not (tmp_path / "cwd_anchor/diags").exists()

    def test_absolute_lasy_file_ignores_relative_to(self, tmp_path) -> None:
        pulse = _make_lasy(lasy_file=tmp_path / "abs_laser", **LASY_TINY_GRID)
        assert pulse.resolve_lasy_file(tmp_path / "elsewhere") == tmp_path / "abs_laser"

    def test_prepare_with_mpi4py_like_comm(self, tmp_path) -> None:
        """A raw mpi4py-style communicator receives (payload, error) by broadcast."""
        comm = _FakeComm(rank=0)
        pulse = _make_lasy(lasy_file=tmp_path / "mpi", **LASY_TINY_GRID)
        pulse.prepare(comm)
        assert pulse.is_prepared and pulse.out_a0 > 0
        assert comm.broadcasts == [
            (((str(pulse.lasy_file_path), pulse.out_a0), None), 0)
        ]

    def test_rank0_build_failure_is_broadcast_and_reraised(self, tmp_path) -> None:
        """Rank 0 re-raises the original error after telling the other ranks."""
        comm = _FakeComm(rank=0)
        # GDD without a spectral bandwidth is rejected by HighOrderLasyLaser.
        pulse = _make_lasy(lasy_file=tmp_path / "bad", gdd=1e-28, **LASY_TINY_GRID)
        with pytest.raises(
            ValueError, match="requires a nonzero laser_spectral_bandwidth"
        ):
            pulse.prepare(comm)
        assert not pulse.is_prepared
        assert len(comm.broadcasts) == 1
        (payload, message), root = comm.broadcasts[0]
        assert payload is None and root == 0
        assert message.startswith("ValueError: ")
        assert "laser_spectral_bandwidth" in message

    def test_non_root_rank_raises_instead_of_hanging_on_rank0_failure(self) -> None:
        """A non-root rank that receives an error payload raises immediately."""
        comm = _FakeComm(rank=1, incoming=(None, "ValueError: bad grid"))
        pulse = _make_lasy()
        with pytest.raises(
            RuntimeError, match="LASY build failed on rank 0: ValueError: bad grid"
        ):
            pulse.prepare(comm)
        assert not pulse.is_prepared
        assert comm.broadcasts == [((None, None), 0)]

    def test_non_root_rank_adopts_broadcast_result(self, tmp_path) -> None:
        """A non-root rank never builds; it takes the path and a0 from rank 0."""
        existing = tmp_path / "from_rank0_00000.h5"
        existing.write_bytes(b"")
        comm = _FakeComm(rank=1, incoming=((str(existing), 1.23), None))
        pulse = _make_lasy()
        pulse.prepare(comm)
        assert pulse.is_prepared
        assert pulse.lasy_file_path == existing
        assert pulse.a0 == pulse.out_a0 == 1.23
        assert pulse._high_order_laser is None

    def test_peak_delay_and_spectral_phase_reach_the_lasy_build(self, tmp_path) -> None:
        pulse = _make_lasy(
            lasy_file=tmp_path / "spectral",
            spectral_bandwidth="auto",
            gdd_relative=0.3,
            peak_delay_from_file_start=120e-15,
            center_and_remove_tilt=False,
            **LASY_TINY_GRID,
        )
        pulse.prepare(None)
        laser = pulse._high_order_laser
        assert laser.hyperparameters["peak_delay_from_file_start_s"] == 120e-15
        assert laser.physical_parameters["laser_gdd_s2"] != 0.0
        time, intensity = laser._on_axis_intensity()
        measured_delay = time[int(np.argmax(intensity))] - time[0]
        assert measured_delay == pytest.approx(120e-15, abs=2 * (time[1] - time[0]))
        assert pulse.out_a0 > 0

    def test_build_laser_profile_auto_prepares(self, tmp_path) -> None:
        from fbpic.lpa_utils.laser.laser_profiles import FromLasyFileLaser

        pulse = _make_lasy(
            lasy_file=tmp_path / "auto", t_start=5.0e-15, **LASY_TINY_GRID
        )
        profile = pulse.build_laser_profile()
        assert isinstance(profile, FromLasyFileLaser)
        assert profile.t_start == 5.0e-15
        assert pulse.is_prepared

    @pytest.mark.parametrize("mode", ["lineout", "lineout_and_2d"])
    def test_plot(self, tmp_path, mode: str) -> None:
        import matplotlib.pyplot as plt

        pulse = _make_lasy(lasy_file=tmp_path / "plot", **LASY_TINY_GRID)
        out = tmp_path / f"{mode}.png"
        _, ax = plt.subplots()
        fig = pulse.plot(mode=mode, ax=ax, output_path=out)
        assert out.is_file()
        assert fig is not None
        assert len(ax.get_lines()) == 1
        plt.close("all")


# ===================================================================
# _format_polarization helper
# ===================================================================


class TestFormatPolarization:
    def test_linear(self) -> None:
        from inversion_fbpic.lib.laser import _format_polarization

        result = _format_polarization(0.0)
        assert "linear" in result

    def test_circular(self) -> None:
        from inversion_fbpic.lib.laser import _format_polarization

        assert "circular" in _format_polarization("left")
        assert "circular" in _format_polarization("right")

    def test_elliptical(self) -> None:
        from inversion_fbpic.lib.laser import _format_polarization

        result = _format_polarization([pi / 4, pi / 2])
        assert "elliptical" in result

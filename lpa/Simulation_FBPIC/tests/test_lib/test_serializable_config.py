"""
Regression tests for SerializableConfig (attrs): Python construction, YAML/JSON I/O,
minimal inputs, example templates, and composite Simulation loading.

YAML load fixtures are inline multiline strings below. Tests that need on-disk
files write only under pytest ``tmp_path`` / ``yaml_out`` (auto-removed).

Edge-case and known-limitation coverage lives in sibling modules:

* ``test_config_yaml.py`` — comment extraction/injection, round-trip flags,
  docstring ``Args:`` parsing (see module docstring for the full test catalog).
* ``test_config_yaml_limitations.py`` — tabs, anchors, block scalars, unquoted
  ``#`` in scalars, simulation aggregate comment maps.

Run from Simulation_FBPIC::

    pytest tests/test_lib/test_serializable_config.py \\
           tests/test_lib/test_config_yaml.py \\
           tests/test_lib/test_config_yaml_limitations.py -v
"""

from __future__ import annotations

import functools
import json
import logging
import runpy
import subprocess
import warnings
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml


@pytest.fixture(autouse=True)
def isolated_git_hash_cache(monkeypatch) -> None:
    """Use a fresh test cache without changing the real import-time snapshot."""
    from inversion_fbpic.lib import serializable_config as module

    monkeypatch.setattr(
        module, "_git_hash", functools.cache(module._git_hash.__wrapped__)
    )


@pytest.fixture
def git_hash_file(monkeypatch, tmp_path: Path) -> Path:
    from inversion_fbpic.lib import serializable_config as module

    path = tmp_path / "git_hash.txt"
    monkeypatch.setattr(module, "_GIT_HASH_FILE", path)
    return path


def test_git_hash_file_is_package_relative() -> None:
    from inversion_fbpic.lib import serializable_config as module

    assert module._GIT_HASH_FILE == (
        Path(module.__file__).resolve().parents[1] / "git_hash.txt"
    )


@pytest.mark.parametrize(
    "content",
    ["a" * 40, "", " \n", FileNotFoundError(), PermissionError(), UnicodeError()],
)
def test_git_hash_is_captured_during_import(monkeypatch, content) -> None:
    from inversion_fbpic.lib import serializable_config as module

    revision = (content.strip() or None) if isinstance(content, str) else None
    read = (
        Mock(return_value=content)
        if isinstance(content, str)
        else Mock(side_effect=content)
    )
    monkeypatch.setattr(Path, "read_text", read)
    # Execute in a separate namespace without re-registering the real domains.
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        imported = runpy.run_path(module.__file__)
        read.assert_called_once_with(encoding="utf-8")
        read.side_effect = None
        read.return_value = "b" * 40
        assert imported["_git_hash"]() == revision
        assert imported["_git_hash"]() == revision
    assert len(emitted) == (0 if revision is not None else 1)
    if emitted:
        assert emitted[0].category is RuntimeWarning
        assert "git_hash: null for this process" in str(emitted[0].message)
        assert "restart Python" in str(emitted[0].message)
    assert read.call_count == 1


def test_git_hash_reads_recorded_file_without_git(
    monkeypatch, git_hash_file: Path, tmp_path: Path
) -> None:
    from inversion_fbpic.lib import serializable_config as module

    revision = "a" * 40
    git_hash_file.write_text(revision + "\n", encoding="utf-8")
    run = Mock(side_effect=AssertionError("Serialization must not run Git"))
    monkeypatch.setattr(subprocess, "run", run)
    other_directory = tmp_path / "unrelated"
    other_directory.mkdir()
    monkeypatch.chdir(other_directory)

    assert module._git_hash() == revision
    run.assert_not_called()


@pytest.mark.parametrize("content", [None, b" \n", b"\xff"])
def test_git_hash_missing_empty_or_invalid(git_hash_file: Path, content) -> None:
    from inversion_fbpic.lib import serializable_config as module

    if content is not None:
        git_hash_file.write_bytes(content)
    assert module._git_hash() is None


def test_git_hash_unreadable(monkeypatch) -> None:
    from inversion_fbpic.lib import serializable_config as module

    path = Mock()
    path.read_text.side_effect = PermissionError("Revision file is unreadable")
    monkeypatch.setattr(module, "_GIT_HASH_FILE", path)
    assert module._git_hash() is None


@pytest.mark.parametrize("revision", ["a" * 40, None])
@pytest.mark.parametrize("include_nones", [True, False])
def test_git_hash_in_all_serialization_formats(
    git_hash_file: Path, minimal_density, tmp_path: Path, revision, include_nones: bool
) -> None:
    if revision is not None:
        git_hash_file.write_text(revision, encoding="utf-8")
    config = minimal_density
    config_cls = type(config)
    json_path = tmp_path / "config.json"
    yaml_path = tmp_path / "config.yaml"
    example_json_path = tmp_path / "example.json"
    example_yaml_path = tmp_path / "example.yaml"

    config.to_json_file(json_path, include_nones=include_nones)
    config.to_yaml_file(yaml_path, include_nones=include_nones)
    config_cls.example_json(example_json_path, include_nones=include_nones)
    config_cls.example_yaml(example_yaml_path, include_nones=include_nones)
    payloads = [
        config.to_dict(include_nones=include_nones),
        json.loads(config.to_json(include_nones=include_nones)),
        yaml.safe_load(config.to_yaml(include_nones=include_nones)),
        json.loads(json_path.read_text()),
        yaml.safe_load(yaml_path.read_text()),
        config_cls.example_dict(include_nones=include_nones),
        json.loads(config_cls.example_json(include_nones=include_nones)),
        yaml.safe_load(config_cls.example_yaml(include_nones=include_nones)),
        json.loads(example_json_path.read_text()),
        yaml.safe_load(example_yaml_path.read_text()),
    ]
    for payload in payloads:
        assert payload["git_hash"] == revision
        assert "git_hash" not in payload["parameters"]
    for path in (json_path, yaml_path):
        assert config_cls.from_file(path).to_dict()["git_hash"] == revision


def test_git_hash_in_nested_configs_and_laser_override(
    git_hash_file: Path, minimal_simulation_elements
) -> None:
    from inversion_fbpic.lib import serializable_config as module

    revision = "a" * 40
    git_hash_file.write_text(revision, encoding="utf-8")
    # Discard fixture-construction logging's read of the not-yet-written test file.
    module._git_hash.cache_clear()
    simulation = minimal_simulation_elements
    for payload in (
        simulation.to_dict(),
        json.loads(simulation.to_json()),
        yaml.safe_load(simulation.to_yaml()),
    ):
        assert payload["git_hash"] == revision
        assert "git_hash" not in payload["parameters"]
        elements = payload["parameters"]["elements"]
        assert len(elements) == 3
        for element in elements:
            assert element["git_hash"] == revision
            assert "git_hash" not in element["parameters"]
        laser = next(e for e in elements if e["config_type"] == "laser_pulse")
        assert "out_a0" in laser["parameters"]
        assert "a0" not in laser["parameters"]


@pytest.mark.parametrize("incoming_hash", ["b" * 40, None, "missing"])
def test_loaded_git_hash_is_informational(
    git_hash_file: Path, minimal_density, incoming_hash
) -> None:
    from inversion_fbpic.lib import serializable_config as module

    revision = "a" * 40
    git_hash_file.write_text(revision, encoding="utf-8")
    payload = minimal_density.to_dict()
    if incoming_hash == "missing":
        payload.pop("git_hash")
    else:
        payload["git_hash"] = incoming_hash
    original = json.dumps(payload)

    for loaded in (
        module.SerializableConfig.from_dict(payload),
        module.SerializableConfig.from_json(json.dumps(payload)),
        module.SerializableConfig.from_yaml(yaml.safe_dump(payload)),
    ):
        assert loaded.to_dict()["git_hash"] == revision
        assert loaded.length == minimal_density.length
    assert json.dumps(payload) == original


@pytest.mark.parametrize("revision", ["a" * 40, None])
def test_git_hash_stays_cached_during_serialization(
    git_hash_file: Path, minimal_density, revision
) -> None:
    if revision is not None:
        git_hash_file.write_text(revision, encoding="utf-8")
    assert minimal_density.to_dict()["git_hash"] == revision
    git_hash_file.write_text("b" * 40, encoding="utf-8")
    assert minimal_density.to_dict()["git_hash"] == revision
    assert type(minimal_density).example_dict()["git_hash"] == revision
    git_hash_file.unlink()
    assert minimal_density.to_dict()["git_hash"] == revision


def test_from_any_parses_json_after_path_oserror(monkeypatch, minimal_density) -> None:
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    monkeypatch.setattr(
        SerializableConfig,
        "from_file",
        Mock(side_effect=OSError("File name too long")),
    )
    loaded = SerializableConfig.from_any(minimal_density.to_json())
    assert loaded.to_dict() == minimal_density.to_dict()


@pytest.mark.parametrize("revision", ["a" * 40, None])
def test_simulation_config_hash_includes_git_revision(
    git_hash_file: Path, minimal_simulation_elements, revision
) -> None:
    from inversion_fbpic.lib import serializable_config as module

    if revision is not None:
        git_hash_file.write_text(revision, encoding="utf-8")
    module._git_hash.cache_clear()
    first = minimal_simulation_elements.config_hash()
    assert minimal_simulation_elements.config_hash() == first
    git_hash_file.write_text("b" * 40, encoding="utf-8")
    assert minimal_simulation_elements.config_hash() == first
    # A fresh import in a new process captures the new revision.
    module._git_hash.cache_clear()
    assert minimal_simulation_elements.config_hash() != first


# Inline YAML fixtures (formerly under tests/test_lib/yaml_in/).
EXAMPLE_DENSITY_MINIMAL_YAML = """\
# Minimal ExampleDensityProfile: only required parameters plus length.
# Omitted keys (ionization, p_rmax, elec_*, ion_*, start_position) use attrs
# defaults (e.g. species='H', ionization=0). species is omitted here on purpose.
config_type: density_profile
subclass: sine_squared_bump
parameters:
  nominal_density: 1.0e+24
  length: 1.0e-05
  p_nz: 1
  p_nr: 1
  p_nt: 1
"""

EXAMPLE_DENSITY_WITH_OPTIONS_YAML = """\
# ExampleDensityProfile with optional diagnostics and species filled in.
config_type: density_profile
subclass: sine_squared_bump
parameters:
  nominal_density: 1.0e+24
  p_nz: 4
  p_nr: 4
  p_nt: 8
  species: H
  ionization: 1
  p_rmax: 5.0e-05
  length: 1.0e-03
  start_position: 0.0
  elec_name: electrons
  elec_select:
    uz: [10.0, null]
    z: [-1.0e-06, 1.0e-06]
  ion_name: null
  ion_select: null
"""

EXAMPLE_LASER_MINIMAL_YAML = """\
# Minimal GaussianLaserPulse: optional method / antenna fields omitted (default None).
config_type: laser_pulse
subclass: gaussian
parameters:
  energy: 5.0
  z0: -3.0e-05
  wavelength: 8.0e-07
  tau_fwhm: 3.8e-14
  cep: 0.0
  waist: 2.8e-05
  focal_position: 3.0e-03
  polarization: 0.0
"""

EXAMPLE_SIMULATION_HYPERPARAMETERS_YAML = """\
# SimulationHyperparameters with only required fields; optional buffers/diagnostics omitted.
config_type: simulation_hyperparameters
subclass: simulation_hyperparameters
parameters:
  zmin: -7.0e-05
  zmax: 0.0
  rmax: 1.4e-04
  nz: 2048
  nr: 300
  nm: 3
  use_mpi: false
  number_dumps: 100
"""

YAML_FIXTURES: dict[str, str] = {
    "example_density_minimal.yaml": EXAMPLE_DENSITY_MINIMAL_YAML,
    "example_density_with_options.yaml": EXAMPLE_DENSITY_WITH_OPTIONS_YAML,
    "example_laser_minimal.yaml": EXAMPLE_LASER_MINIMAL_YAML,
    "example_simulation_hyperparameters.yaml": EXAMPLE_SIMULATION_HYPERPARAMETERS_YAML,
}


@pytest.fixture
def yaml_out(tmp_path: Path) -> Path:
    """Writable output directory under pytest's auto-cleaned tmp_path."""
    out = tmp_path / "yaml_out"
    out.mkdir()
    return out


def test_yaml_fixtures_defined() -> None:
    assert all(text.strip() for text in YAML_FIXTURES.values())
    assert "config_type:" in EXAMPLE_DENSITY_MINIMAL_YAML


def test_parameter_descriptions_ignore_nested_yaml_examples() -> None:
    """Args parsing must not split on example keys deeper than param indent."""
    from inversion_fbpic.lib.density_core import _DensityProfile

    lines = _DensityProfile._parameter_descriptions()["elec_select"]
    joined = " ".join(lines)
    assert "Filters for the electron species" in joined
    assert "Example Python" in joined
    assert "Example YAML" in joined
    assert joined.index("Example YAML") < joined.index("Defaults to None")
    assert "[dict]" in lines[0]


def test_yaml_round_trip_preserves_comment_indentation() -> None:
    """Nested indentation inside user comment text is preserved on load and dump."""
    from inversion_fbpic.lib.commented_yaml import extract_yaml_comments
    from inversion_fbpic.lib.density_profiles import ExampleDensityProfile
    from inversion_fbpic.lib.serializable_config import (
        CONFIG_TYPE_STR,
        PARAMETERS_STR,
    )

    source = """config_type: density_profile
subclass: sine_squared_bump
parameters:
  nominal_density: 1.0e+24
  elec_select: null                    # first line
                                       #     Example YAML:
                                       #         elec_select:
                                       #           uz:
                                       #           - 10.0
  length: 1.0
  p_nz: 1
  p_nr: 1
  p_nt: 1
"""
    profile = ExampleDensityProfile.from_yaml(source)
    lines = profile.yaml_parameter_descriptions["elec_select"]
    assert lines[0] == "first line"
    assert lines[1] == "    Example YAML:"
    assert lines[2] == "        elec_select:"
    assert lines[3] == "          uz:"
    assert lines[4] == "          - 10.0"

    round_trip = profile.to_yaml(comments=True, round_trip_comments=True)
    assert "#         elec_select:" in round_trip
    assert "#           uz:" in round_trip
    assert "#           - 10.0" in round_trip

    extracted = extract_yaml_comments(
        round_trip, header_key=CONFIG_TYPE_STR, block_key=PARAMETERS_STR
    )[()][1]["elec_select"]
    assert extracted[2] == "        elec_select:"
    assert extracted[3] == "          uz:"


def test_yaml_round_trip_user_comments() -> None:
    """Comments from a YAML file are stored and re-emitted on to_yaml."""
    from inversion_fbpic.lib.commented_yaml import extract_yaml_comments
    from inversion_fbpic.lib.density_profiles import ExampleDensityProfile
    from inversion_fbpic.lib.serializable_config import (
        CONFIG_TYPE_STR,
        PARAMETERS_STR,
    )

    source = """# My custom class note.
config_type: density_profile
subclass: sine_squared_bump
parameters:
  nominal_density: 1.0e+24  # user nominal comment
  length: 1.0e-05
  p_nz: 1
  p_nr: 1
  p_nt: 1
"""
    profile = ExampleDensityProfile.from_yaml(source)
    assert profile.yaml_class_description == "My custom class note."
    assert profile.yaml_parameter_descriptions is not None
    assert profile.yaml_parameter_descriptions["nominal_density"] == [
        "user nominal comment"
    ]
    assert profile.yaml_parameter_descriptions.get("length") is None

    round_trip = profile.to_yaml(comments=True, round_trip_comments=True)
    assert "# My custom class note." in round_trip
    assert "user nominal comment" in round_trip

    extracted = extract_yaml_comments(
        round_trip, header_key=CONFIG_TYPE_STR, block_key=PARAMETERS_STR
    )[()]
    assert extracted[0] == "My custom class note."
    assert extracted[1]["nominal_density"] == ["user nominal comment"]


def test_yaml_round_trip_falls_back_to_docstrings() -> None:
    """Instances without stored comments use docstrings when dumping YAML."""
    from inversion_fbpic.lib.density_profiles import ExampleDensityProfile

    profile = ExampleDensityProfile(
        nominal_density=1.0e24,
        length=1.0e-5,
        p_nz=1,
        p_nr=1,
        p_nt=1,
    )
    assert not profile._has_round_trip_yaml_comments()
    yaml_text = profile.to_yaml(comments=True, round_trip_comments=True)
    assert "Finite sine-squared" in yaml_text
    assert "[m]" in yaml_text


def test_write_example_templates(yaml_out: Path) -> None:
    from inversion_fbpic.lib.density_profiles import ExampleDensityProfile
    from inversion_fbpic.lib.laser import GaussianLaserPulse
    from inversion_fbpic.lib.simulation import SimulationHyperparameters

    ExampleDensityProfile.example_json(yaml_out / "example_density_profile.json")
    ExampleDensityProfile.example_yaml(
        yaml_out / "example_density_profile.yaml", comments=True, include_nones=True
    )
    GaussianLaserPulse.example_json(yaml_out / "example_gaussian_laser_pulse.json")
    GaussianLaserPulse.example_yaml(
        yaml_out / "example_gaussian_laser_pulse.yaml",
        comments=True,
        include_nones=True,
    )
    SimulationHyperparameters.example_yaml(
        yaml_out / "example_simulation_hyperparameters_full.yaml",
        comments=True,
        include_nones=True,
    )
    SimulationHyperparameters.example_json(
        yaml_out / "example_simulation_hyperparameters_full.json"
    )

    for name in (
        "example_density_profile.json",
        "example_density_profile.yaml",
        "example_gaussian_laser_pulse.json",
        "example_gaussian_laser_pulse.yaml",
        "example_simulation_hyperparameters_full.yaml",
        "example_simulation_hyperparameters_full.json",
    ):
        assert (yaml_out / name).is_file()


def test_python_minimal_constructors() -> None:
    from inversion_fbpic.lib.density_profiles import ExampleDensityProfile
    from inversion_fbpic.lib.laser import GaussianLaserPulse
    from inversion_fbpic.lib.simulation import SimulationHyperparameters

    profile = ExampleDensityProfile(
        nominal_density=1.0e24,
        length=1.0e-5,
        p_nz=1,
        p_nr=1,
        p_nt=1,
    )
    assert profile.species == "H"
    assert profile.ionization == 0

    pulse = GaussianLaserPulse(
        energy=5.0,
        z0=-30e-6,
        wavelength=800e-9,
        tau_fwhm=38e-15,
        cep=0.0,
        waist=28e-6,
        focal_position=3e-3,
        polarization=0.0,
    )
    assert pulse.method is None
    assert pulse.a0 > 0.0

    hy = SimulationHyperparameters(
        zmin=-70e-6,
        zmax=0.0,
        rmax=140e-6,
        nz=2048,
        nr=300,
        nm=3,
        use_mpi=False,
        number_dumps=100,
    )
    assert hy.right_buffer is None
    assert hy.field_diagnostics is None


@pytest.mark.parametrize(
    "fixture_name,out_name",
    [
        ("example_density_minimal.yaml", "loaded_density_minimal_roundtrip.yaml"),
        ("example_density_with_options.yaml", "loaded_density_with_options.yaml"),
        ("example_laser_minimal.yaml", "loaded_laser_minimal_roundtrip.yaml"),
        ("example_simulation_hyperparameters.yaml", "loaded_hyperparameters.yaml"),
    ],
)
def test_yaml_minimal_load_roundtrip_from_temp_file(
    tmp_path: Path, fixture_name: str, out_name: str
) -> None:
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    src_path = tmp_path / fixture_name
    src_path.write_text(YAML_FIXTURES[fixture_name], encoding="utf-8")

    obj = SerializableConfig.from_file(src_path)
    out_path = tmp_path / out_name
    no_nones_path = tmp_path / out_name.replace(".yaml", "_no_nones.yaml")
    obj.to_yaml_file(out_path, comments=True, include_nones=True)
    obj.to_yaml_file(no_nones_path, comments=False, include_nones=False)

    assert out_path.is_file()
    assert no_nones_path.is_file()


def test_round_trip_yaml_string(yaml_out: Path) -> None:
    from inversion_fbpic.lib.density_profiles import ExampleDensityProfile
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    p = ExampleDensityProfile(
        nominal_density=1.0e18 * 1e6,
        length=10e-6,
        p_nz=2,
        p_nr=2,
        p_nt=2,
        elec_name="electrons",
    )
    y = p.to_yaml(comments=False)
    p2 = SerializableConfig.from_yaml(y)
    assert type(p2) is ExampleDensityProfile
    assert p2.length == p.length
    assert p2.elec_name == "electrons"
    (yaml_out / "roundtrip_density_from_string.yaml").write_text(
        p2.to_yaml(comments=True), encoding="utf-8"
    )


def test_round_trip_json(yaml_out: Path) -> None:
    from inversion_fbpic.lib.laser import GaussianLaserPulse
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    pulse = GaussianLaserPulse(
        energy=5.0,
        z0=-30e-6,
        wavelength=800e-9,
        tau_fwhm=38e-15,
        cep=0.0,
        waist=28e-6,
        focal_position=0.0,
        polarization=0.0,
    )
    blob = pulse.to_json(indent=2)
    pulse2 = SerializableConfig.from_json(blob)
    assert type(pulse2) is GaussianLaserPulse
    (yaml_out / "roundtrip_laser.json").write_text(blob, encoding="utf-8")
    json.loads(blob)


def test_unknown_parameter_rejected() -> None:
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    bad = """
config_type: density_profile
subclass: sine_squared_bump
parameters:
  nominal_density: 1.0e+20
  length: 1.0e-06
  p_nz: 1
  p_nr: 1
  p_nt: 1
  not_a_real_field: 1
"""
    with pytest.raises(ValueError, match="Unknown parameter"):
        SerializableConfig.from_yaml(bad)


def test_hyperparameters_grid_yaml(yaml_out: Path) -> None:
    from inversion_fbpic.lib.simulation import SimulationHyperparameters

    hy = SimulationHyperparameters(
        zmin=-70e-6,
        zmax=0.0,
        rmax=140e-6,
        nz=2048,
        nr=300,
        nm=3,
        use_mpi=False,
        number_dumps=100,
        field_diagnostics=["rho", "E", "B"],
    )
    grid_yaml = hy.grid_parameters_yaml(
        file_name=yaml_out / "derived_grid_parameters.yaml",
        comments=True,
    )
    assert "dt:" in grid_yaml
    assert "dr:" in grid_yaml
    assert "dz:" in grid_yaml
    assert (yaml_out / "derived_grid_parameters.yaml").is_file()


def test_simulation_python_then_paths(yaml_out: Path) -> None:
    from inversion_fbpic.lib.density_profiles import ExampleDensityProfile
    from inversion_fbpic.lib.laser import GaussianLaserPulse
    from inversion_fbpic.lib.simulation import Simulation, SimulationHyperparameters

    hy = SimulationHyperparameters(
        zmin=-70e-6,
        zmax=0.0,
        rmax=140e-6,
        nz=2048,
        nr=300,
        nm=3,
        use_mpi=False,
        number_dumps=100,
        field_diagnostics=["rho", "E", "B"],
    )
    d1 = ExampleDensityProfile(
        nominal_density=1.0e18 * 1e6,
        p_nz=4,
        p_nr=4,
        p_nt=8,
        length=1e-3,
        elec_name="electrons",
        elec_select={"uz": [10.0, None], "z": [-1e-6, 1e-6]},
    )
    d2 = ExampleDensityProfile(
        nominal_density=4.0e18 * 1e6,
        p_nz=4,
        p_nr=4,
        p_nt=8,
        length=1e-6,
        start_position=0.5e-3,
    )
    laser = GaussianLaserPulse(
        energy=5.0,
        z0=-30e-6,
        wavelength=800e-9,
        tau_fwhm=38e-15,
        cep=0.0,
        waist=30e-6,
        focal_position=0.5e-3,
        polarization=0.0,
    )

    p_hy = yaml_out / "sim_hyperparameters.yaml"
    p_d1 = yaml_out / "sim_density_1.yaml"
    p_d2 = yaml_out / "sim_density_2.yaml"
    p_laser = yaml_out / "sim_laser.yaml"
    hy.to_yaml_file(p_hy, comments=True)
    d1.to_yaml_file(p_d1, comments=True)
    d2.to_yaml_file(p_d2, comments=True)
    laser.to_yaml_file(p_laser, comments=True)

    sim = Simulation(elements=[hy, d1, d2, laser], verbosity=logging.WARNING)
    sim.to_yaml_file(yaml_out / "simulation_aggregate_python.yaml", comments=True)

    sim2 = Simulation(
        elements=[p_hy, p_d1, p_d2, p_laser],
        verbosity=logging.WARNING,
    )
    sim2.to_yaml_file(yaml_out / "simulation_reloaded_from_paths.yaml", comments=True)

    sim3 = Simulation.from_file(yaml_out / "simulation_aggregate_python.yaml")
    sim3.to_yaml_file(
        yaml_out / "simulation_roundtrip_from_aggregate.yaml",
        comments=False,
        include_nones=False,
    )
    assert len(sim3.elements) == 4


def test_serializable_config_dispatch() -> None:
    from inversion_fbpic.lib.laser import GaussianLaserPulse
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    obj = SerializableConfig.from_yaml(EXAMPLE_LASER_MINIMAL_YAML)
    d = obj.to_dict()
    obj2 = SerializableConfig.from_dict(d)
    assert type(obj) is type(obj2)
    assert type(obj) is GaussianLaserPulse


def test_laser_polarization_yaml_snippets(yaml_out: Path) -> None:
    from scipy.constants import pi

    from inversion_fbpic.lib.laser import GaussianLaserPulse

    base = dict(
        a0=1.0,
        z0=0.0,
        wavelength=800e-9,
        tau_fwhm=38e-15,
        cep=0.0,
        waist=28e-6,
        focal_position=0.0,
    )
    cases = [
        ("example_laser_polarization_linear.yaml", pi / 4),
        ("example_laser_polarization_left.yaml", "left"),
        ("example_laser_polarization_elliptical.yaml", (pi / 8, pi / 2)),
    ]
    for fname, pol in cases:
        pulse = GaussianLaserPulse(**base, polarization=pol)
        path = yaml_out / fname
        path.write_text(pulse.to_yaml(comments=True), encoding="utf-8")
        assert path.is_file()


def test_from_file_resolves_relative_paths_against_source_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nested file references resolve relative to the loading config, not cwd."""
    from inversion_fbpic.lib.density_core import _DensityProfile
    from inversion_fbpic.lib.density_modifiers import ModifiedDensityProfile
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "base.yaml").write_text(
        """\
config_type: density_profile
subclass: smooth_sine_flattop
parameters:
  nominal_density: 1.0e+24
  flattop_width: 3.0e-03
  upramp_length: 0.5e-03
  downramp_length: 1.0e-03
  offset_length: -0.5e-03
  p_nz: 1
  p_nr: 1
  p_nt: 1
""",
        encoding="utf-8",
    )
    (cfg_dir / "modified.yaml").write_text(
        """\
config_type: density_profile
subclass: modified_density_profile
parameters:
  base_density_profile: base.yaml
  modifiers: []
""",
        encoding="utf-8",
    )

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    loaded = SerializableConfig.from_file(cfg_dir / "modified.yaml")
    assert isinstance(loaded, ModifiedDensityProfile)
    assert loaded.source_file == (cfg_dir / "modified.yaml").resolve()
    assert loaded.resolved_base_density_profile.SUBCLASS == "smooth_sine_flattop"
    assert isinstance(loaded.resolved_base_density_profile, _DensityProfile)
    assert (
        loaded.nominal_density == loaded.resolved_base_density_profile.nominal_density
    )
    assert loaded.p_nz == loaded.resolved_base_density_profile.p_nz


def _write_linked_simulation_cfg(project_root: Path) -> Path:
    """Write a simulation.yaml plus sibling element configs under ``project_root/cfg``."""
    cfg_dir = project_root / "cfg"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "simulation_hyperparameters.yaml").write_text(
        EXAMPLE_SIMULATION_HYPERPARAMETERS_YAML, encoding="utf-8"
    )
    (cfg_dir / "density.yaml").write_text(
        EXAMPLE_DENSITY_MINIMAL_YAML, encoding="utf-8"
    )
    (cfg_dir / "laser.yaml").write_text(EXAMPLE_LASER_MINIMAL_YAML, encoding="utf-8")
    simulation_yaml = cfg_dir / "simulation.yaml"
    simulation_yaml.write_text(
        """\
config_type: simulation
subclass: simulation
parameters:
  elements:
  - simulation_hyperparameters.yaml
  - density.yaml
  - laser.yaml
  verbosity: 20
""",
        encoding="utf-8",
    )
    return simulation_yaml


def test_simulation_from_file_with_relative_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Top-level simulation YAML and nested element paths resolve from *relative_to*."""
    from inversion_fbpic.lib.simulation import Simulation

    project_root = tmp_path / "project"
    _write_linked_simulation_cfg(project_root)

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    sim = Simulation.from_file("cfg/simulation.yaml", relative_to=project_root)
    assert sim.source_file == (project_root / "cfg" / "simulation.yaml").resolve()
    assert sim.hyparams is not None
    assert len(sim.densities) == 1
    assert len(sim.lasers) == 1


def test_simulation_elements_resolve_with_path_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Programmatic ``Simulation(elements=[...])`` resolves paths via context anchor."""
    from inversion_fbpic.lib.serializable_config import SerializableConfig
    from inversion_fbpic.lib.simulation import Simulation

    project_root = tmp_path / "project"
    cfg_dir = project_root / "cfg"
    _write_linked_simulation_cfg(project_root)

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    with SerializableConfig.resolving_paths_relative_to(cfg_dir):
        sim = Simulation(
            elements=[
                "simulation_hyperparameters.yaml",
                "density.yaml",
                "laser.yaml",
            ],
            verbosity=logging.WARNING,
        )

    assert sim.hyparams is not None
    assert len(sim.densities) == 1
    assert len(sim.lasers) == 1


def test_normalize_working_directory_honors_argument(tmp_path: Path) -> None:
    from inversion_fbpic.lib.simulation import Simulation

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert Simulation._normalize_working_directory(run_dir) == run_dir.resolve()
    assert Simulation._normalize_working_directory(str(run_dir)) == run_dir.resolve()


def test_from_file_delegates_to_from_file(tmp_path: Path) -> None:
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    path = tmp_path / "laser.yaml"
    path.write_text(EXAMPLE_LASER_MINIMAL_YAML, encoding="utf-8")
    loaded = SerializableConfig.from_file(path)
    assert loaded.source_file == path.resolve()


def test_from_any_returns_existing_instance() -> None:
    from inversion_fbpic.lib.laser import GaussianLaserPulse
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    pulse = GaussianLaserPulse(
        energy=5.0,
        z0=-30e-6,
        wavelength=800e-9,
        tau_fwhm=38e-15,
        cep=0.0,
        waist=28e-6,
        focal_position=3e-3,
        polarization=0.0,
    )
    assert SerializableConfig.from_any(pulse) is pulse


def test_from_any_loads_from_path(tmp_path: Path) -> None:
    from inversion_fbpic.lib.laser import GaussianLaserPulse
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    path = tmp_path / "laser.yaml"
    path.write_text(EXAMPLE_LASER_MINIMAL_YAML, encoding="utf-8")

    from_path = SerializableConfig.from_any(path)
    from_str = SerializableConfig.from_any(str(path))

    assert isinstance(from_path, GaussianLaserPulse)
    assert isinstance(from_str, GaussianLaserPulse)
    assert from_path.source_file == path.resolve()
    assert from_str.source_file == path.resolve()


def test_from_any_loads_from_dict() -> None:
    from inversion_fbpic.lib.laser import GaussianLaserPulse
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    pulse = GaussianLaserPulse(
        energy=5.0,
        z0=-30e-6,
        wavelength=800e-9,
        tau_fwhm=38e-15,
        cep=0.0,
        waist=28e-6,
        focal_position=3e-3,
        polarization=0.0,
    )
    loaded = SerializableConfig.from_any(pulse.to_dict())
    assert isinstance(loaded, GaussianLaserPulse)
    assert loaded.energy == pulse.energy


def test_from_any_resolves_yaml_string() -> None:
    from inversion_fbpic.lib.laser import GaussianLaserPulse
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    loaded = SerializableConfig.from_any(EXAMPLE_LASER_MINIMAL_YAML)
    assert isinstance(loaded, GaussianLaserPulse)
    assert loaded.energy == 5.0


def test_from_any_resolves_json_string() -> None:
    from inversion_fbpic.lib.laser import GaussianLaserPulse
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    pulse = GaussianLaserPulse(
        energy=5.0,
        z0=-30e-6,
        wavelength=800e-9,
        tau_fwhm=38e-15,
        cep=0.0,
        waist=28e-6,
        focal_position=3e-3,
        polarization=0.0,
    )
    loaded = SerializableConfig.from_any(pulse.to_json())
    assert isinstance(loaded, GaussianLaserPulse)
    assert loaded.energy == pulse.energy


def test_from_any_invalid_path_raises(tmp_path: Path) -> None:
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    missing = tmp_path / "missing.yaml"
    with pytest.raises(ValueError, match="Invalid path"):
        SerializableConfig.from_any(missing)


def test_from_any_invalid_string_raises() -> None:
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    with pytest.raises(ValueError, match="YAML or JSON object"):
        SerializableConfig.from_any("not a config")


def test_from_any_unsupported_type_raises() -> None:
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    with pytest.raises(TypeError, match="Cannot deserialize config"):
        SerializableConfig.from_any(42)


# ---------------------------------------------------------------------------
# LaserPulse XOR energy / a0
# ---------------------------------------------------------------------------

_LASER_BASE = dict(
    z0=-30e-6,
    wavelength=800e-9,
    tau_fwhm=38e-15,
    cep=0.0,
    waist=28e-6,
    focal_position=3e-3,
    polarization=0.0,
)


def test_laser_energy_only_resolves_a0() -> None:
    from inversion_fbpic.lib.laser import GaussianLaserPulse

    pulse = GaussianLaserPulse(energy=5.0, **_LASER_BASE)
    assert pulse.a0 > 0.0
    assert pulse._amplitude_source == "energy"
    d = pulse.to_dict()
    params = d["parameters"]
    assert "energy" in params
    assert "a0" not in params
    assert params["out_a0"] == pytest.approx(pulse.a0)
    assert "out_energy" not in params


def test_laser_a0_only_resolves_energy() -> None:
    from inversion_fbpic.lib.laser import GaussianLaserPulse

    pulse = GaussianLaserPulse(a0=2.0, **_LASER_BASE)
    assert pulse.energy > 0.0
    assert pulse._amplitude_source == "a0"
    d = pulse.to_dict()
    params = d["parameters"]
    assert "a0" in params
    assert "energy" not in params
    assert params["out_energy"] == pytest.approx(pulse.energy)
    assert "out_a0" not in params


def test_laser_both_energy_and_a0_raises() -> None:
    from inversion_fbpic.lib.laser import GaussianLaserPulse

    with pytest.raises(ValueError, match="exactly one"):
        GaussianLaserPulse(energy=5.0, a0=2.0, **_LASER_BASE)


def test_laser_neither_energy_nor_a0_raises() -> None:
    from inversion_fbpic.lib.laser import GaussianLaserPulse

    with pytest.raises(ValueError, match="exactly one"):
        GaussianLaserPulse(**_LASER_BASE)


def test_to_dict_include_nones_false_preserves_nested_null_in_lists() -> None:
    """include_nones=False omits top-level optional nulls, not null list elements."""
    from inversion_fbpic.lib.density_profiles import ExampleDensityProfile
    from inversion_fbpic.lib.serializable_config import PARAMETERS_STR

    profile = ExampleDensityProfile(
        nominal_density=1.0e24,
        length=1.0e-5,
        p_nz=1,
        p_nr=1,
        p_nt=1,
        elec_select={"uz": [10.0, None], "z": [-1e-6, 1e-6]},
    )
    params = profile.to_dict(include_nones=False)[PARAMETERS_STR]
    assert "ion_select" not in params
    assert params["elec_select"]["uz"] == [10.0, None]
    assert params["elec_select"]["z"] == [-1e-6, 1e-6]

    yaml_text = profile.to_yaml(comments=False, include_nones=False)
    assert "elec_select: {uz: [10.0, null], z: [-1.0e-06, 1.0e-06]}" in yaml_text
    reloaded = ExampleDensityProfile.from_yaml(yaml_text)
    assert reloaded.elec_select == {"uz": [10.0, None], "z": [-1e-6, 1e-6]}


def test_to_yaml_can_disable_nested_flow_style() -> None:
    """Nested attrs default to flow YAML but can use block style on request."""
    from inversion_fbpic.lib.density_profiles import ExampleDensityProfile

    profile = ExampleDensityProfile(
        nominal_density=1.0e24,
        length=1.0e-5,
        p_nz=1,
        p_nr=1,
        p_nt=1,
        elec_select={"uz": [10.0, None]},
    )

    flow_yaml = profile.to_yaml(comments=False)
    block_yaml = profile.to_yaml(comments=False, nested_flow_style=False)

    assert "elec_select: {uz: [10.0, null]}" in flow_yaml
    assert "elec_select:\n    uz:\n    - 10.0\n    - null" in block_yaml


def test_laser_yaml_with_stale_out_a0_loads_correctly(tmp_path: Path) -> None:
    """Editing waist in a YAML with energy + out_a0 should reload fine."""
    from inversion_fbpic.lib.laser import GaussianLaserPulse
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    pulse = GaussianLaserPulse(energy=5.0, **_LASER_BASE)
    yaml_path = tmp_path / "laser.yaml"
    pulse.to_yaml_file(yaml_path, include_nones=False)

    text = yaml_path.read_text()
    assert "out_a0" in text
    modified = text.replace("waist: 2.8e-05", "waist: 2.8e-04")
    yaml_path.write_text(modified)

    reloaded = SerializableConfig.from_file(yaml_path)
    assert isinstance(reloaded, GaussianLaserPulse)
    assert reloaded.energy == 5.0
    assert reloaded.a0 != pulse.a0  # different waist → different a0


def test_laser_yaml_with_both_energy_and_a0_raises(tmp_path: Path) -> None:
    """Legacy YAMLs with both energy and a0 as inputs must raise."""
    from inversion_fbpic.lib.laser import GaussianLaserPulse

    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(
        """\
config_type: laser_pulse
subclass: gaussian
parameters:
  energy: 5.0
  a0: 2.0
  z0: -3.0e-05
  wavelength: 8.0e-07
  tau_fwhm: 3.8e-14
  cep: 0.0
  waist: 2.8e-05
  focal_position: 3.0e-03
  polarization: 0.0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one"):
        GaussianLaserPulse.from_file(yaml_path)


# ===================================================================
# Registry completeness
# ===================================================================


_EXPECTED_REGISTRY_ENTRIES: list[tuple[str, str]] = [
    ("density_profile", "sine_squared_bump"),
    ("density_profile", "asymmetric_sine"),
    ("density_profile", "smooth_sine_flattop"),
    ("density_profile", "gaussian_plus_triangle"),
    ("density_profile", "generalized_gaussian_plus_triangle"),
    ("density_profile", "modified_density_profile"),
    ("density_profile", "interp_from_h5"),
    ("density_modifier", "matched_radial_modifier"),
    ("laser_pulse", "gaussian"),
    ("laser_pulse", "lasy"),
    ("grid_parameters", "grid_parameters"),
    ("simulation_hyperparameters", "simulation_hyperparameters"),
    ("simulation", "simulation"),
]


@pytest.mark.parametrize(
    "config_type,subclass",
    _EXPECTED_REGISTRY_ENTRIES,
    ids=[f"{ct}/{sub}" for ct, sub in _EXPECTED_REGISTRY_ENTRIES],
)
def test_registry_contains_subclass(config_type: str, subclass: str) -> None:
    """Every concrete subclass with a SUBCLASS ClassVar is reachable via the registry."""
    import inversion_fbpic.lib  # noqa: F401 — triggers registry population
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    domain_cls = SerializableConfig._DOMAIN_REGISTRY.get(config_type)
    assert domain_cls is not None, f"domain {config_type!r} not in _DOMAIN_REGISTRY"
    assert (
        subclass in domain_cls._CONCRETE_REGISTRY
    ), f"subclass {subclass!r} not in {config_type} concrete registry"


# ===================================================================
# Parametrized YAML round-trip for density subclasses
# ===================================================================


_DENSITY_YAML_FIXTURES = [
    pytest.param(
        """\
config_type: density_profile
subclass: asymmetric_sine
parameters:
  nominal_density: 1.0e+24
  p_nz: 2
  p_nr: 2
  p_nt: 2
  peak_z0: 0.0025
  upramp_length: 0.0025
  downramp_length: 0.0015
""",
        id="asymmetric_sine",
    ),
    pytest.param(
        """\
config_type: density_profile
subclass: smooth_sine_flattop
parameters:
  nominal_density: 1.0e+24
  p_nz: 2
  p_nr: 2
  p_nt: 2
  flattop_width: 0.001
  upramp_length: 0.0005
  downramp_length: 0.0005
""",
        id="smooth_sine_flattop",
    ),
    pytest.param(
        """\
config_type: density_profile
subclass: gaussian_plus_triangle
parameters:
  nominal_density: 1.0e+24
  p_nz: 2
  p_nr: 2
  p_nt: 2
  gauss_sigma: 3.0e-06
  gauss_z0: 1.0e-05
  tri_z0: 1.5e-05
  tri_left_width: 5.0e-06
  tri_right_width: 5.0e-06
""",
        id="gaussian_plus_triangle",
    ),
    pytest.param(
        """\
config_type: density_profile
subclass: generalized_gaussian_plus_triangle
parameters:
  nominal_density: 1.0e+24
  p_nz: 2
  p_nr: 2
  p_nt: 2
  gauss_peak: 1.0
  gauss_alpha: 3.0e-06
  gauss_beta: 2.0
  gauss_z0: 1.0e-05
  tri_z0: 1.5e-05
  tri_left_width: 5.0e-06
  tri_right_width: 5.0e-06
  tri_height: 1.0
""",
        id="generalized_gaussian_plus_triangle",
    ),
]


@pytest.mark.parametrize("yaml_str", _DENSITY_YAML_FIXTURES)
def test_density_subclass_yaml_round_trip(yaml_str: str) -> None:
    """Each density subclass can survive a YAML round-trip (load -> dump -> reload)."""
    from inversion_fbpic.lib.density_core import _DensityProfile

    profile = _DensityProfile.from_yaml(yaml_str)
    dumped = profile.to_yaml(comments=False)
    reloaded = _DensityProfile.from_yaml(dumped)
    assert type(reloaded) is type(profile)
    orig_params = profile.to_dict()["parameters"]
    reload_params = reloaded.to_dict()["parameters"]
    for key, val in orig_params.items():
        if val is not None:
            assert reload_params[key] == pytest.approx(
                val, rel=1e-6
            ), f"Mismatch on {key}"

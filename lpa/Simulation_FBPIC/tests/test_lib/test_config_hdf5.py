"""Native HDF5 configuration round trips, embedding, and schema validation."""

from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Any, ClassVar

import attrs
import h5py
import numpy as np
import pytest

from inversion_fbpic.lib.density_profiles import ExampleDensityProfile
from inversion_fbpic.lib.laser import GaussianLaserPulse
from inversion_fbpic.lib.serializable_config import SerializableConfig
from inversion_fbpic.lib.simulation import Simulation, SimulationHyperparameters


@attrs.define(kw_only=True, slots=False)
class Hdf5TestConfig(SerializableConfig):
    CONFIG_TYPE: ClassVar[str] = "hdf5_test_config"
    SUBCLASS: ClassVar[str] = "hdf5_test_config"
    _CONCRETE_REGISTRY: ClassVar[dict[str, type[SerializableConfig]]] = {}

    value: Any = attrs.field(default=None)
    filename: Path | None = attrs.field(default=None, metadata={"input_path": True})


@pytest.fixture
def density() -> ExampleDensityProfile:
    return ExampleDensityProfile(
        nominal_density=1e24, length=1e-6, p_nz=1, p_nr=1, p_nt=1
    )


@pytest.mark.parametrize("suffix", [".h5", ".hdf5", ".H5", ".HDF5"])
def test_file_dispatch(
    tmp_path: Path, density: ExampleDensityProfile, suffix: str
) -> None:
    path = tmp_path / "nested" / f"density{suffix}"
    assert density.to_hdf5_file(path) == path.resolve()
    for source in (path, str(path)):
        loaded = SerializableConfig.from_any(source)
        assert isinstance(loaded, ExampleDensityProfile)
        assert loaded.to_dict() == density.to_dict()
        assert loaded.source_file == path.resolve()
    with h5py.File(path, "r") as handle:
        assert handle["config/config_type"].asstr()[()] == "density_profile"
        assert handle["config/parameters/nominal_density"][()] == 1e24
    loaded = SerializableConfig.from_file(
        path, overrides={"parameters": {"length": 2e-6}}
    )
    assert loaded.length == 2e-6


@pytest.mark.parametrize("revision", ["a" * 40, None])
@pytest.mark.parametrize("include_nones", [True, False])
def test_git_hash_metadata_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    density: ExampleDensityProfile,
    revision: str | None,
    include_nones: bool,
) -> None:
    from inversion_fbpic.lib import serializable_config as module

    monkeypatch.setattr(module, "_git_hash", lambda: revision)
    path = density.to_hdf5_file(tmp_path / "config.h5", include_nones=include_nones)
    with h5py.File(path, "r") as handle:
        node = handle["config/git_hash"]
        assert isinstance(node, h5py.Dataset)
        assert "git_hash" not in handle["config/parameters"]
        if revision is None:
            assert node.shape is None
            assert node.attrs["_config_kind"] == "none"
        else:
            assert node.asstr()[()] == revision
    assert SerializableConfig.from_file(path).to_dict() == density.to_dict()
    density.to_hdf5_file(path, include_nones=include_nones, overwrite=True)
    assert SerializableConfig.from_file(path).to_dict()["git_hash"] == revision


@pytest.mark.parametrize("incoming_hash", ["b" * 40, None, "missing"])
def test_loaded_git_hash_is_informational(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, incoming_hash: str | None
) -> None:
    from inversion_fbpic.lib import serializable_config as module

    monkeypatch.setattr(module, "_git_hash", lambda: incoming_hash)
    path = Hdf5TestConfig(value="payload").to_hdf5_file(tmp_path / "config.h5")
    if incoming_hash == "missing":
        with h5py.File(path, "r+") as handle:
            del handle["config/git_hash"]
    monkeypatch.setattr(module, "_git_hash", lambda: "a" * 40)
    loaded = SerializableConfig.from_file(path)
    assert loaded.value == "payload"
    assert loaded.to_dict()["git_hash"] == "a" * 40
    loaded.to_hdf5_file(path, overwrite=True)
    with h5py.File(path, "r") as handle:
        assert handle["config/git_hash"].asstr()[()] == "a" * 40


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [],
        "",
        "日本語 é",
        "embedded\0null",
        True,
        False,
        1,
        -1,
        2**100,
        -(2**100),
        1.5,
        [True, False],
        [1, 2, 3],
        [1.0, 2.0],
        [2**100, 1],
        [1, 2.5, None, {"nested": []}],
        [[1, 2], [3]],
        {
            "": None,
            "/": [],
            ".": {},
            "..": "parent",
            "%": "percent",
            "a/b": "slash",
            "_config_kind": "user",
        },
        np.array([[1, 2], [3, 4]]),
        np.float64(2.5),
        (1, None),
        1 + 2j,
    ],
)
def test_canonical_values(tmp_path: Path, value: Any) -> None:
    config = Hdf5TestConfig(value=value)
    path = config.to_hdf5_file(tmp_path / "values.h5")
    assert SerializableConfig.from_file(path).to_dict() == config.to_dict()


def test_none_is_a_null_dataset(tmp_path: Path) -> None:
    config = SimulationHyperparameters(
        zmin=-1e-5,
        zmax=0,
        rmax=1e-5,
        nz=8,
        nr=8,
        nm=1,
        use_mpi=False,
        number_dumps=2,
    )
    path = config.to_hdf5_file(tmp_path / "optional.h5")
    with h5py.File(path, "r") as handle:
        for name in ("right_buffer", "beta_window"):
            node = handle[f"config/parameters/{name}"]
            assert isinstance(node, h5py.Dataset)
            assert node.shape is None
            assert node.dtype == np.dtype("f8")
            assert node.attrs["_config_kind"] == "none"
    loaded = SerializableConfig.from_file(path)
    assert loaded.to_dict() == config.to_dict()
    attrs.evolve(config, right_buffer=1e-6, beta_window=0.99).to_hdf5_file(
        path, overwrite=True
    )
    with h5py.File(path, "r") as handle:
        assert handle["config/parameters/right_buffer"].shape == ()
        assert handle["config/parameters/beta_window"][()] == 0.99


def test_none_omission(tmp_path: Path) -> None:
    config = Hdf5TestConfig(value={"preserved": None, "empty": []})
    path = config.to_hdf5_file(tmp_path / "none.h5", include_nones=False)
    with h5py.File(path, "r") as handle:
        assert "filename" not in handle["config/parameters"]
    loaded = SerializableConfig.from_file(path)
    assert loaded.value == {"preserved": None, "empty": []}
    assert loaded.filename is None


@pytest.mark.parametrize("amplitude", [{"energy": 5.0}, {"a0": 1.5}])
def test_laser_domain_override(tmp_path: Path, amplitude: dict[str, float]) -> None:
    pulse = GaussianLaserPulse(
        **amplitude,
        z0=0,
        wavelength=800e-9,
        tau_fwhm=38e-15,
        cep=0,
        waist=28e-6,
        focal_position=0,
        polarization=(0.2, 0.5),
    )
    path = pulse.to_hdf5_file(tmp_path / "laser.h5")
    assert SerializableConfig.from_file(path).to_dict() == pulse.to_dict()
    assert GaussianLaserPulse.from_hdf5_file(path).to_dict() == pulse.to_dict()


def test_nested_simulation(tmp_path: Path, density: ExampleDensityProfile) -> None:
    hyperparameters = SimulationHyperparameters(
        zmin=-1e-5, zmax=0, rmax=1e-5, nz=8, nr=8, nm=1, use_mpi=False, number_dumps=2
    )
    pulse = GaussianLaserPulse(
        energy=5.0,
        z0=0,
        wavelength=800e-9,
        tau_fwhm=38e-15,
        cep=0,
        waist=28e-6,
        focal_position=0,
        polarization=0,
    )
    simulation = Simulation(elements=[hyperparameters, density, pulse])
    path = simulation.to_hdf5_file(tmp_path / "simulation.h5")
    loaded = SerializableConfig.from_file(path)
    assert isinstance(loaded, Simulation)
    assert loaded.to_dict() == simulation.to_dict()
    assert len(loaded.densities) == 1
    with h5py.File(path, "r") as handle:
        assert isinstance(handle["config/parameters/elements/0/parameters"], h5py.Group)


def test_embedding_and_overwrite(
    tmp_path: Path, density: ExampleDensityProfile
) -> None:
    path = tmp_path / "simulation.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=[1, 2])
        handle.attrs["scientific_metadata"] = "keep"
        group = handle.create_group("metadata/config")
        assert density.to_hdf5(group) is group
        assert SerializableConfig.from_any(group).to_dict() == density.to_dict()
        assert handle.id.valid
        with pytest.raises(ValueError, match="occupied"):
            density.to_hdf5(group)
        density.to_hdf5(group, overwrite=True)
        with pytest.raises(ValueError, match="occupied"):
            density.to_hdf5(handle, overwrite=True)
        np.testing.assert_array_equal(handle["data"][()], [1, 2])
        assert handle.attrs["scientific_metadata"] == "keep"
    density.to_hdf5_file(path, group_path="/metadata/config", overwrite=True)
    assert (
        SerializableConfig.from_hdf5_file(path, group_path="/metadata/config").to_dict()
        == density.to_dict()
    )


def test_root_and_fileobj_handles(density: ExampleDensityProfile) -> None:
    with h5py.File(io.BytesIO(), "w") as handle:
        density.to_hdf5(handle)
        loaded = SerializableConfig.from_hdf5(handle)
        assert loaded.to_dict() == density.to_dict()
        assert loaded.source_file is None
        assert handle.id.valid
        density.to_hdf5(handle, overwrite=True)
        assert SerializableConfig.from_hdf5(handle).to_dict() == density.to_dict()


def test_portable_input_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = tmp_path / "original"
    original.mkdir()
    data = original / "input.dat"
    data.write_text("data")
    config = Hdf5TestConfig(filename=data)
    config.to_hdf5_file(original / "config.h5")
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    monkeypatch.chdir(tmp_path)
    loaded = SerializableConfig.from_file("config.h5", relative_to=relocated)
    assert loaded.filename == relocated / "input.dat"
    assert loaded.source_file == relocated / "config.h5"
    with h5py.File(relocated / "config.h5", "r") as handle:
        assert handle["config/parameters/filename"].asstr()[()] == "input.dat"


def test_failed_write_preserves_config_and_context(tmp_path: Path) -> None:
    from inversion_fbpic.lib.serializable_config import _config_serialize_relative_to

    path = Hdf5TestConfig(value="original").to_hdf5_file(tmp_path / "config.h5")
    with pytest.raises(TypeError, match="Unsupported HDF5"):
        Hdf5TestConfig(value=object()).to_hdf5_file(path, overwrite=True)
    assert _config_serialize_relative_to.get() is None
    assert SerializableConfig.from_file(path).value == "original"


@pytest.mark.parametrize(
    "corruption",
    ["version", "kind", "sequence", "none_group", "discriminator", "parameters"],
)
def test_malformed_config(tmp_path: Path, corruption: str) -> None:
    from inversion_fbpic.lib.serializable_config import _config_load_source

    path = Hdf5TestConfig(value=[None, 1]).to_hdf5_file(tmp_path / "config.h5")
    with h5py.File(path, "r+") as handle:
        group = handle["config"]
        if corruption == "version":
            group.attrs["_config_version"] = 999
        elif corruption == "kind":
            del group["parameters/value"].attrs["_config_kind"]
        elif corruption == "sequence":
            del group["parameters/value/0"]
        elif corruption == "none_group":
            del group["parameters/value/0"]
            group["parameters/value"].create_group("0").attrs["_config_kind"] = "none"
        elif corruption == "discriminator":
            group["subclass"][()] = "unknown"
        else:
            del group["parameters"]
    with pytest.raises(ValueError):
        SerializableConfig.from_file(path)
    assert _config_load_source.get() is None


def test_non_config_hdf5(tmp_path: Path) -> None:
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=[1, 2])
        with pytest.raises(ValueError, match="not a serialized config"):
            SerializableConfig.from_hdf5(handle)
    with pytest.raises(ValueError, match="Missing HDF5 config group"):
        SerializableConfig.from_file(path)


def test_copy_failure_preserves_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = Hdf5TestConfig(value="original").to_hdf5_file(tmp_path / "config.h5")

    def fail_copy(*args: Any, **kwargs: Any) -> None:
        raise OSError("copy failed")

    monkeypatch.setattr(h5py.Group, "copy", fail_copy)
    with pytest.raises(OSError, match="copy failed"):
        Hdf5TestConfig(value="replacement").to_hdf5_file(path, overwrite=True)
    assert SerializableConfig.from_file(path).value == "original"
    with h5py.File(path, "r") as handle:
        assert set(handle) == {"config"}


def test_unrelated_nested_data_is_not_deleted(tmp_path: Path) -> None:
    path = Hdf5TestConfig().to_hdf5_file(tmp_path / "config.h5")
    with h5py.File(path, "r+") as handle:
        handle["config/parameters"].create_dataset("scientific_data", data=[1, 2])
        with pytest.raises(ValueError, match="node kind"):
            Hdf5TestConfig().to_hdf5(handle["config"], overwrite=True)
        np.testing.assert_array_equal(
            handle["config/parameters/scientific_data"][()], [1, 2]
        )


def test_unrelated_root_data_is_not_deleted(tmp_path: Path) -> None:
    path = Hdf5TestConfig().to_hdf5_file(tmp_path / "config.h5")
    with h5py.File(path, "r+") as handle:
        group = handle["config"]
        # Even a valid codec node is not permitted as an arbitrary root member.
        node = group.create_dataset("scientific_data", data=1)
        node.attrs["_config_kind"] = "int"
        with pytest.raises(ValueError, match="payload"):
            SerializableConfig.from_hdf5(group)
        with pytest.raises(ValueError, match="occupied"):
            Hdf5TestConfig().to_hdf5(group, overwrite=True)
        assert node[()] == 1


@pytest.mark.parametrize("failed_move", [2, 5])
def test_link_move_failure_restores_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_move: int
) -> None:
    path = Hdf5TestConfig(value="original").to_hdf5_file(tmp_path / "config.h5")
    original_move = h5py.Group.move
    calls = 0

    def fail_once(group: h5py.Group, source: str, destination: str) -> None:
        nonlocal calls
        calls += 1
        if calls == failed_move:
            raise OSError("link move failed")
        original_move(group, source, destination)

    monkeypatch.setattr(h5py.Group, "move", fail_once)
    with pytest.raises(OSError, match="link move failed"):
        Hdf5TestConfig(value="replacement").to_hdf5_file(path, overwrite=True)
    assert SerializableConfig.from_file(path).value == "original"
    with h5py.File(path, "r") as handle:
        assert set(handle) == {"config"}


def test_cyclic_link_is_rejected(tmp_path: Path) -> None:
    path = Hdf5TestConfig(value={}).to_hdf5_file(tmp_path / "config.h5")
    with h5py.File(path, "r+") as handle:
        group = handle["config/parameters/value"]
        group["cycle"] = group
        with pytest.raises(ValueError, match="Cyclic"):
            SerializableConfig.from_hdf5(handle["config"])


@pytest.mark.parametrize("with_input_path", [False, True])
@pytest.mark.parametrize("change_cwd", [False, True])
@pytest.mark.parametrize("at_root", [False, True])
def test_relative_open_handle_can_write_without_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    density: ExampleDensityProfile,
    with_input_path: bool,
    change_cwd: bool,
    at_root: bool,
) -> None:
    from inversion_fbpic.lib.serializable_config import _config_serialize_relative_to

    input_path = tmp_path / "input.dat"
    input_path.write_text("data")
    config = Hdf5TestConfig(filename=input_path) if with_input_path else density
    monkeypatch.chdir(tmp_path)
    with h5py.File("run.h5", "w") as handle:
        group = handle if at_root else handle.create_group("config")
        if change_cwd:
            other = tmp_path / "elsewhere"
            other.mkdir()
            monkeypatch.chdir(other)
        assert config.to_hdf5(group) is group
        config.to_hdf5(group, overwrite=True)
        assert handle.id.valid
        if with_input_path:
            assert group["parameters/filename"].asstr()[()] == str(input_path)
        assert _config_serialize_relative_to.get() is None

    loaded = SerializableConfig.from_hdf5_file(
        tmp_path / "run.h5", group_path="/" if at_root else "/config"
    )
    assert loaded.to_dict() == config.to_dict()


def test_relative_open_handle_write_uses_explicit_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.dat"
    input_path.write_text("data")
    config = Hdf5TestConfig(filename=input_path)
    monkeypatch.chdir(tmp_path)
    with h5py.File("run.h5", "w") as handle:
        other = tmp_path / "elsewhere"
        other.mkdir()
        monkeypatch.chdir(other)
        with SerializableConfig.resolving_paths_relative_to(tmp_path):
            config.to_hdf5(handle.create_group("config"))
        assert handle["config/parameters/filename"].asstr()[()] == "input.dat"
    assert SerializableConfig.from_file(tmp_path / "run.h5").filename == input_path


def test_relative_open_handle_requires_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Hdf5TestConfig(value="anchored").to_hdf5_file(tmp_path / "config.h5")
    monkeypatch.chdir(tmp_path)
    with h5py.File("config.h5", "r") as handle:
        other = tmp_path / "elsewhere"
        other.mkdir()
        monkeypatch.chdir(other)
        with pytest.raises(ValueError, match="absolute path"):
            SerializableConfig.from_hdf5(handle["config"])
        with SerializableConfig.resolving_paths_relative_to(tmp_path):
            loaded = SerializableConfig.from_hdf5(handle["config"])
        assert loaded.source_file == tmp_path / "config.h5"


def test_nested_config_file_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, density: ExampleDensityProfile
) -> None:
    from inversion_fbpic.lib.density_modifiers import ModifiedDensityProfile

    directory = tmp_path / "bundle"
    directory.mkdir()
    density.to_hdf5_file(directory / "density.h5")
    # Keep the nested reference as a path to exercise source context on reload.
    payload = {
        "config_type": "density_profile",
        "subclass": "modified_density_profile",
        "parameters": {"base_density_profile": "density.h5", "modifiers": []},
    }
    with SerializableConfig.resolving_paths_relative_to(directory):
        profile = SerializableConfig.from_dict(payload)
    profile.to_hdf5_file(directory / "modified.h5")
    monkeypatch.chdir(tmp_path)
    loaded = SerializableConfig.from_file(directory / "modified.h5")
    assert isinstance(loaded, ModifiedDensityProfile)
    assert loaded.resolved_base_density_profile.to_dict() == density.to_dict()

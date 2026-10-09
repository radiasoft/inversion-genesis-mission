"""How an archive stores a simulator's config: native HDF5, and no paths.

Each config object is written with `SerializableConfig.to_hdf5()` into a group of the archive. A
config can hold two kinds of path, and neither goes into the archive as a path:

- Output paths (`save_directory`, `lasy_file`): where a run writes, decided when it runs. Only
  the basename of the class default is stored (`diags`, `lasy_laser`), and a loaded config has
  the class default; the simulator's working directory decides the rest.
- Input files (fields marked `input_path`, such as the table of `InterpolateFromH5Profile`): the
  archive keeps only the file's basename and its md5. The file itself is looked for in the
  `input_dirs` given to `archive()` and `from_archive()`. When it is not in `input_dirs` at
  `archive()` it is embedded in the archive, and a loaded archive then uses the embedded copy.

The configs are written and read through an in-memory HDF5 file: the library makes a path relative
to the file it writes into and resolves one against the file it reads from, and neither is wanted
here.
"""

from __future__ import annotations

import copy
import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
import typing

import attrs
import h5py
import numpy

from inversion_fbpic.lib.laser import LasyLaserPulse
from inversion_fbpic.lib.serializable_config import SerializableConfig
from inversion_fbpic.lib.simulation import SimulationHyperparameters
from lume_fbpic.pwfa_config import PWFAGrid

# The output path fields of the config classes: where a run writes, not data. Another registered
# class with a path field that is not here, and not an input file, is caught by a test.
OUTPUT_PATH_FIELDS: dict[type, tuple[str, ...]] = {
    LasyLaserPulse: ("lasy_file",),
    PWFAGrid: ("save_directory",),
    SimulationHyperparameters: ("save_directory",),
}

# An input file embedded in the archive that is larger than this gets a warning.
EMBED_WARNING_BYTES = 10 * 2**20

# Config fields whose value may name a config file; an archive cannot hold the path.
_PATH_VALUED_FIELDS = ("base_density_profile", "modifiers")

_CHUNK_BYTES = 8 * 2**20


@dataclass
class InputFile:
    """An input file of a config: its basename and md5, and where the bytes are.

    The bytes are on disk at `path`, or in memory as `data` for a file that was embedded in an
    archive and has not been written out.
    """

    basename: str
    md5: str
    size: int
    path: Path | None = None
    data: numpy.ndarray | None = None

    def copy_into(self, dataset: h5py.Dataset) -> None:
        """Fill `dataset` (uint8, `size` long) with the bytes of the file."""
        if self.data is not None:
            dataset[...] = self.data
            return
        with open(self.path, "rb") as stream:
            offset = 0
            while chunk := stream.read(_CHUNK_BYTES):
                dataset[offset : offset + len(chunk)] = numpy.frombuffer(
                    chunk, dtype=numpy.uint8
                )
                offset += len(chunk)

    def write_to(self, directory: Path) -> Path:
        """Write the embedded bytes as `directory/basename` and return that path."""
        target = Path(directory) / self.basename
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or checksum(target) != self.md5:
            target.write_bytes(self.data.tobytes())
        return target.resolve()


@dataclass
class InputRef:
    """An input-file field of a config object: which one, and the file it names."""

    config: str
    index: int | None
    field: str
    file: InputFile

    @property
    def key(self) -> tuple[str, int | None, str]:
        return (self.config, self.index, self.field)


def checksum(path: Path | str) -> str:
    """The md5 of a file, read in chunks."""
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as stream:
        while chunk := stream.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def collect_inputs(
    config: dict[str, typing.Any],
    embedded: dict[tuple[str, int | None, str], InputFile],
) -> list[InputRef]:
    """The input files `config` names, found the way the simulation finds them.

    A path is opened as written (a relative one from the current directory). An input file that
    was embedded in an archive and is not on disk is taken from `embedded`.

    Raises:
        FileNotFoundError: If a file is in neither place: the simulation could not run.
    """
    refs = []
    for name, index, item in items(config):
        _reject_path_values(name, item)
        for field in input_fields(item):
            value = Path(getattr(item, field))
            if value.is_file():
                refs.append(
                    InputRef(
                        name,
                        index,
                        field,
                        InputFile(
                            value.name,
                            checksum(value),
                            value.stat().st_size,
                            path=value,
                        ),
                    )
                )
                continue
            held = embedded.get((name, index, field))
            if held is None or held.basename != value.name:
                raise FileNotFoundError(
                    f"The input file {str(value)!r} of {name}.{field} does not exist, so the "
                    "simulation could not run and cannot be archived."
                )
            refs.append(InputRef(name, index, field, held))
    return refs


def directories(input_dirs: typing.Any) -> list[Path]:
    """`input_dirs` as a list of paths: None is the current directory."""
    if input_dirs is None:
        return [Path.cwd()]
    if isinstance(input_dirs, (str, Path)):
        return [Path(input_dirs)]
    return [Path(directory) for directory in input_dirs]


def input_fields(config: SerializableConfig) -> list[str]:
    """The names of the fields of `config` that are input files."""
    return [
        field.name
        for field in attrs.fields(type(config))
        if field.metadata.get("input_path")
    ]


def items(
    config: dict[str, typing.Any],
) -> typing.Iterator[tuple[str, int | None, SerializableConfig]]:
    """Each config object of `config` with its name and, in a list, its index."""
    for name, value in config.items():
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                yield name, index, item
        else:
            yield name, None, value


def load_inputs(
    group: h5py.Group, input_dirs: typing.Any, working_directory: str | Path
) -> tuple[dict, dict]:
    """Find the input files an archive's `inputs` group records.

    Returns `(paths, embedded)`. `paths` is the path to use for each (config, index, field). An
    embedded file is written to `working_directory/inputs/` first, because a config is built from
    the file. `embedded` holds the embedded files, for `collect_inputs()` when the loaded model
    is archived again.

    A referenced file is looked for in `input_dirs`; one with a different md5 than at archive
    time is used with a warning.

    Raises:
        FileNotFoundError: If a referenced file is not in any of `input_dirs`.
    """
    paths: dict[tuple[str, int | None, str], str] = {}
    embedded: dict[tuple[str, int | None, str], InputFile] = {}
    if group is None:
        return paths, embedded
    dirs = directories(input_dirs)
    for entry in sorted(group, key=int):
        record = group[entry]
        index = int(record.attrs["index"])
        key = (
            str(record.attrs["config"]),
            None if index < 0 else index,
            str(record.attrs["field"]),
        )
        basename, md5 = str(record.attrs["basename"]), str(record.attrs["md5"])
        if "data" in record:
            file = InputFile(
                basename, md5, int(record.attrs["size"]), data=record["data"][()]
            )
            embedded[key] = file
            paths[key] = str(file.write_to(Path(working_directory) / "inputs"))
            continue
        found = next((d / basename for d in dirs if (d / basename).is_file()), None)
        if found is None:
            raise FileNotFoundError(
                f"The input file {basename!r} of {key[0]}.{key[2]} is not in "
                f"{[str(d) for d in dirs]}; give its directory as input_dirs."
            )
        if checksum(found) != md5:
            warnings.warn(
                f"The input file {str(found)!r} is not the one that was archived (its md5 is "
                f"not {md5}); using it anyway.",
                UserWarning,
                stacklevel=3,
            )
        paths[key] = str(found.resolve())
    return paths, embedded


def normalized(config: SerializableConfig) -> SerializableConfig:
    """A copy of `config` that holds no path: output paths as the basename of their default, input
    files by basename. Not rebuilt (an input file is read when a config is built), so the original, with
    its paths, is untouched."""
    result = copy.copy(config)
    for cls, names in OUTPUT_PATH_FIELDS.items():
        if isinstance(config, cls):
            for name in names:
                default = _default(config, name)
                basename = Path(default).name
                object.__setattr__(
                    result,
                    name,
                    Path(basename) if isinstance(default, Path) else basename,
                )
    for name in input_fields(config):
        object.__setattr__(result, name, Path(Path(getattr(config, name)).name))
    return result


def read_config(
    group: h5py.Group, overrides: dict[str, typing.Any] | None = None
) -> SerializableConfig:
    """Read the config in `group`. `overrides` are the parameters to set; the paths of input
    files go in this way. The output paths are the class defaults."""
    with h5py.File("stage", "w", driver="core", backing_store=False) as stage:
        group.file.copy(group, stage, name="config")
        config = SerializableConfig.from_hdf5(
            stage["config"], overrides={"parameters": overrides} if overrides else None
        )
    for cls, names in OUTPUT_PATH_FIELDS.items():
        if isinstance(config, cls):
            for name in names:
                object.__setattr__(config, name, _default(config, name))
    return config


def store_inputs(
    group: h5py.Group, refs: list[InputRef], input_dirs: typing.Any
) -> None:
    """Record `refs` in `group`: a file found in `input_dirs` by its basename and md5, any other
    one embedded.

    A file in `input_dirs` with another md5 than the one the config names is a warning, and the
    archive refers to it by name as before (it is not embedded: files can be large). An embedded
    file over `EMBED_WARNING_BYTES` is a warning.
    """
    dirs = directories(input_dirs)
    for number, ref in enumerate(refs):
        file = ref.file
        record = group.create_group(str(number))
        record.attrs["config"] = ref.config
        record.attrs["index"] = -1 if ref.index is None else ref.index
        record.attrs["field"] = ref.field
        record.attrs["basename"] = file.basename
        record.attrs["md5"] = file.md5
        record.attrs["size"] = file.size
        found = next(
            (d / file.basename for d in dirs if (d / file.basename).is_file()), None
        )
        if found is not None:
            if checksum(found) != file.md5:
                warnings.warn(
                    f"The input file {str(found)!r} in input_dirs differs from the one "
                    f"{ref.config}.{ref.field} uses (md5 {file.md5}); the archive refers to it "
                    "by name and a loaded model will use it.",
                    UserWarning,
                    stacklevel=3,
                )
            continue
        if file.size > EMBED_WARNING_BYTES:
            warnings.warn(
                f"Embedding the input file {file.basename!r} ({file.size / 2**20:.1f} MB) in "
                "the archive; put its directory in input_dirs to refer to it instead.",
                UserWarning,
                stacklevel=3,
            )
        file.copy_into(
            record.create_dataset("data", shape=(file.size,), dtype=numpy.uint8)
        )


def write_config(
    parent: h5py.Group, name: str, config: SerializableConfig
) -> h5py.Group:
    """Write `config`, without paths, as the group `parent/name` and return it."""
    with h5py.File("stage", "w", driver="core", backing_store=False) as stage:
        normalized(config).to_hdf5(stage.create_group("config"))
        stage.copy("config", parent, name=name)
    return parent[name]


def _default(config: SerializableConfig, name: str) -> typing.Any:
    """The default of field `name` of `config`, converted as the class converts a value."""
    field = attrs.fields_dict(type(config))[name]
    if field.default is attrs.NOTHING:
        raise ValueError(f"{type(config).__name__}.{name} has no default")
    value = field.default
    if isinstance(value, attrs.Factory):
        value = value.factory()
    return field.converter(value) if field.converter is not None else value


def _reject_path_values(name: str, config: SerializableConfig) -> None:
    for field in _PATH_VALUED_FIELDS:
        if isinstance(getattr(config, field, None), (str, Path)):
            raise ValueError(
                f"{name}.{field} is a path to a config file, which an archive cannot hold; "
                "give the config object itself."
            )

"""Attrs-based serializable configuration and serialization.

This module defines a base class for serializable configurations used by
simulation setup code. Configuration objects can be serialized to and from JSON with
schema metadata and a type tag so they can be recorded and reloaded reliably.
"""

from __future__ import annotations

import contextvars
import copy
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
import types
import typing
from abc import ABC
from pathlib import Path
from typing import Any, ClassVar

import attrs

import numpy as np
import yaml

from inversion_fbpic.lib.commented_yaml import (
    YamlDescriptionMap,
    YamlDescriptions,
    YamlPath,
    dump_yaml_with_comments,
    extract_yaml_comments,
)
from inversion_fbpic.lib._doc_management.merge import (
    build_merged_docstring,
    class_description,
    format_args_block,
    merge_parameter_descriptions,
    parameter_descriptions_from_doc,
)

# Backward-compatible alias used by tests.
_parameter_descriptions_from_doc = parameter_descriptions_from_doc

CONFIG_TYPE_STR = "config_type"
SUBCLASS_STR = "subclass"
PARAMETERS_STR = "parameters"
NULL_CONCRETE_STR = "null"

_EXAMPLE_SIMPLE_DEFAULTS: dict[type, Any] = {
    float: 0.0,
    np.floating: 0.0,
    int: 0,
    str: "",
    bool: False,
    complex: [0.0, 0.0],
}
_EXAMPLE_ORIGIN_DEFAULTS: dict[type, Any] = {list: [], dict: {}}


def _default_for_example_type(tp: Any) -> Any:
    """Map a type hint to a placeholder value for example YAML/JSON payloads."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is types.UnionType or origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if not non_none or type(None) in args:
            return None
        return _default_for_example_type(non_none[0])

    if tp in _EXAMPLE_SIMPLE_DEFAULTS:
        return _EXAMPLE_SIMPLE_DEFAULTS[tp]

    if origin in _EXAMPLE_ORIGIN_DEFAULTS:
        return _EXAMPLE_ORIGIN_DEFAULTS[origin]

    if origin is tuple:
        if args:
            return [_default_for_example_type(a) for a in args if a is not Ellipsis]
        return []

    return None


def _optional_payload_class(
    payload: dict[str, Any],
) -> type["SerializableConfig"] | None:
    """Return the concrete config class for a payload dict, or None if not a config."""
    if not {CONFIG_TYPE_STR, SUBCLASS_STR, PARAMETERS_STR}.issubset(payload):
        return None
    try:
        return SerializableConfig._payload_class(payload)
    except ValueError:
        return None


def _contains_config_payload(value: Any) -> bool:
    """Return whether *value* contains a serialized configuration payload."""
    if isinstance(value, dict):
        return _optional_payload_class(value) is not None or any(
            _contains_config_payload(nested) for nested in value.values()
        )
    if isinstance(value, list):
        return any(_contains_config_payload(item) for item in value)
    return False


def _nested_flow_style_paths(data: dict[str, Any]) -> set[YamlPath]:
    """Return paths for nested attrs that should use YAML flow style."""
    paths: set[YamlPath] = set()

    def collect(value: Any, path: YamlPath) -> None:
        if not isinstance(value, dict):
            if isinstance(value, list):
                for index, item in enumerate(value):
                    collect(item, (*path, index))
            return

        config_cls = _optional_payload_class(value)
        if config_cls is not None:
            parameters = value.get(PARAMETERS_STR)
            if isinstance(parameters, dict):
                for name, parameter in parameters.items():
                    parameter_path = (*path, PARAMETERS_STR, name)
                    if isinstance(
                        parameter, (dict, list)
                    ) and not _contains_config_payload(parameter):
                        paths.add(parameter_path)
                    collect(parameter, parameter_path)
            return

        for key, nested in value.items():
            collect(nested, (*path, str(key)))

    collect(data, ())
    return paths


def _complete_init_parameters(
    concrete_cls: type["SerializableConfig"], parameters: dict[str, Any]
) -> dict[str, Any]:
    """Fill missing init parameters using type-hint fallbacks for optional fields."""
    completed = dict(parameters)
    type_hints = typing.get_type_hints(concrete_cls)
    for field in attrs.fields(concrete_cls):
        if not field.init or field.name in completed:
            continue
        if field.default is not attrs.NOTHING:
            continue
        has_fallback, fallback_value = _missing_parameter_default(
            type_hints.get(field.name)
        )
        if has_fallback:
            completed[field.name] = fallback_value
    return completed


def _resolve_input_paths(
    concrete_cls: type["SerializableConfig"], parameters: dict[str, Any]
) -> dict[str, Any]:
    """Resolve declared input-file fields against the active config source."""
    resolved = dict(parameters)
    for field in attrs.fields(concrete_cls):
        if not field.init or not field.metadata.get("input_path"):
            continue
        value = resolved.get(field.name)
        if value is not None:
            resolved[field.name] = SerializableConfig._resolve_config_path(value)
    return resolved


def _write_text_file(path: str | Path, content: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


_OMIT = object()

# While deserializing from a file, nested path references resolve relative to this file.
_config_load_source: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "_config_load_source", default=None
)

# Explicit anchor for resolving config path strings (e.g. programmatic construction).
_config_path_anchor: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "_config_path_anchor", default=None
)

# While serializing to a file, Path values are written relative to this directory.
_config_serialize_relative_to: contextvars.ContextVar[Path | None] = (
    contextvars.ContextVar("_config_serialize_relative_to", default=None)
)


def _serialize_path(path: Path, *, input_path: bool = False) -> str:
    """Serialize a path, preferring paths relative to the output config directory."""
    resolved = path.resolve()
    base = _config_serialize_relative_to.get(None)
    if base is not None:
        try:
            return resolved.relative_to(base.resolve()).as_posix()
        except ValueError:
            if input_path:
                return os.path.relpath(resolved, base.resolve())
    return path.as_posix()


def _set_instance_attr(instance: "SerializableConfig", name: str, value: Any) -> None:
    """Set an attribute on a config instance, including frozen attrs classes."""
    object.__setattr__(instance, name, value)


def _deep_merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overrides* into *base* (mutates and returns *base*)."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _apply_payload_overrides(
    payload: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    """Return a copy of *payload* with *overrides* merged in."""
    if not overrides:
        return payload
    merged = copy.deepcopy(payload)
    return _deep_merge_dict(merged, overrides)


def _missing_parameter_default(tp: Any) -> tuple[bool, Any]:
    """Return whether a safe auto-default exists for a missing parameter type."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is types.UnionType or origin is typing.Union:
        if type(None) in args:
            return True, None

    if origin is list:
        return True, []
    if origin is tuple:
        return True, ()
    if origin is dict:
        return True, {}

    return False, None


@attrs.define(kw_only=True, slots=False)
class SerializableConfig(ABC):
    """
    Base class for serializable configurations.
    Supports serialization to and from JSON and YAML.
    """

    # Two-tier discriminator system:
    #   CONFIG_TYPE: identifies the *domain* (e.g. "density_profile", "laser_pulse",
    #     "simulation"). Set on the domain base class (DensityProfile, etc.).
    #   SUBCLASS:    identifies the *concrete* class within a domain (e.g.
    #     "gaussian_laser_pulse"). Set on each instantiable subclass.
    CONFIG_TYPE: ClassVar[str]
    SUBCLASS: ClassVar[str]

    # Top-level registry: CONFIG_TYPE -> domain base class.
    # Lives on SerializableConfig and is shared across all domains.
    _DOMAIN_REGISTRY: ClassVar[dict[str, type["SerializableConfig"]]] = {}

    # Per-domain registry: SUBCLASS -> concrete class. Each domain base class
    # (DensityProfile, LaserConfig, ...) must redeclare this as an empty dict
    # in its own class body so subclasses register into the right registry.
    _CONCRETE_REGISTRY: ClassVar[dict[str, type["SerializableConfig"]]]

    # Round-trip YAML comments (from a loaded commented YAML file). Not serialized.
    yaml_class_description: str | None = attrs.field(
        default=None, init=False, repr=False
    )
    yaml_parameter_descriptions: dict[str, list[str]] | None = attrs.field(
        default=None, init=False, repr=False
    )
    yaml_description_map: dict[YamlPath, YamlDescriptions] | None = attrs.field(
        default=None, init=False, repr=False
    )
    source_file: Path | None = attrs.field(default=None, init=False, repr=False)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _register_config_subclass(cls)

    @classmethod
    def _parameter_names(cls) -> list[str]:
        """Get the names of the parameters for the class."""
        return [a.name for a in attrs.fields(cls) if a.init]

    @classmethod
    def _parameter_descriptions(cls) -> dict[str, list[str]]:
        """Extract parameter descriptions from Args: blocks in the MRO docstrings."""
        return merge_parameter_descriptions(
            getattr(klass, "__doc__", None) for klass in reversed(cls.__mro__)
        )

    @classmethod
    def _class_description(cls) -> str:
        """Extract the class description (summary), excluding Args blocks."""
        return class_description(cls.__doc__)

    @classmethod
    def _format_args_block(cls, descriptions: dict[str, list[str]]) -> str:
        """Render a Google-style ``Args:`` section from parameter descriptions."""
        return format_args_block(descriptions)

    @classmethod
    def _merged_docstring(cls) -> str:
        """Build a class docstring with ``Args:`` merged across the MRO."""
        return build_merged_docstring(
            cls.__doc__,
            (getattr(klass, "__doc__", None) for klass in reversed(cls.__mro__)),
            cls._parameter_names(),
        )

    @classmethod
    def _docstring_descriptions_for(cls, data: dict[str, Any]) -> YamlDescriptionMap:
        """Build docstring-derived YAML comments for every config payload in *data*."""
        descriptions: dict[YamlPath, YamlDescriptions] = {}

        def collect(value: Any, path: YamlPath) -> None:
            if isinstance(value, dict):
                config_cls = _optional_payload_class(value)
                if config_cls is not None:
                    descriptions[path] = (
                        config_cls._class_description(),
                        config_cls._parameter_descriptions(),
                    )
                for key, nested in value.items():
                    collect(nested, (*path, str(key)))
                return

            if isinstance(value, list):
                for idx, nested in enumerate(value):
                    collect(nested, (*path, idx))

        collect(data, ())
        descriptions.setdefault(
            (), (cls._class_description(), cls._parameter_descriptions())
        )
        return descriptions

    @staticmethod
    def _payload_class(payload: dict[str, Any]) -> type["SerializableConfig"]:
        """Resolve a serialized config payload to its concrete class."""
        domain_type = payload.get(CONFIG_TYPE_STR)
        if not isinstance(domain_type, str):
            raise ValueError(f"Missing or invalid '{CONFIG_TYPE_STR}' in payload.")
        if domain_type not in SerializableConfig._DOMAIN_REGISTRY:
            raise ValueError(f"Unknown '{CONFIG_TYPE_STR}' '{domain_type}'.")
        domain_cls = SerializableConfig._DOMAIN_REGISTRY[domain_type]

        concrete_type = payload.get(SUBCLASS_STR)
        if not isinstance(concrete_type, str):
            raise ValueError(f"Missing or invalid '{SUBCLASS_STR}' in payload.")
        if concrete_type not in domain_cls._CONCRETE_REGISTRY:
            raise ValueError(
                f"Unknown '{SUBCLASS_STR}' '{concrete_type}' for '{CONFIG_TYPE_STR}' '{domain_type}'."
            )
        return domain_cls._CONCRETE_REGISTRY[concrete_type]

    def _has_round_trip_yaml_comments(self) -> bool:
        """Return whether this instance has comments loaded from a YAML source."""
        if self.yaml_description_map:
            return True
        return (
            self.yaml_class_description is not None
            or self.yaml_parameter_descriptions is not None
        )

    def _ingest_yaml_comment_map(
        self, comment_map: dict[YamlPath, YamlDescriptions] | None
    ) -> None:
        """Store comments extracted from a YAML source on this instance."""
        if not comment_map:
            return
        _set_instance_attr(self, "yaml_description_map", comment_map)
        root = comment_map.get(())
        if root is not None:
            class_desc, param_descs = root
            _set_instance_attr(self, "yaml_class_description", class_desc or None)
            _set_instance_attr(self, "yaml_parameter_descriptions", param_descs or None)

    def _yaml_descriptions_for_dump(
        self,
        data: dict[str, Any],
        *,
        round_trip_comments: bool,
    ) -> YamlDescriptionMap:
        """Resolve YAML comments for dumping: round-trip store or docstrings."""
        if round_trip_comments:
            if self.yaml_description_map:
                return self.yaml_description_map
            if self._has_round_trip_yaml_comments():
                return {
                    (): (
                        self.yaml_class_description or "",
                        self.yaml_parameter_descriptions or {},
                    )
                }
        return type(self)._docstring_descriptions_for(data)

    @staticmethod
    def _to_serializable(value: Any, *, include_nones: bool = True) -> Any:
        """Convert values to serializable builtins recursively.

        When *include_nones* is False, only a top-level ``None`` is omitted (e.g.
        optional attrs fields in :meth:`to_dict`). Nested ``None`` values inside
        dicts, lists, and tuples are always preserved.
        """
        if value is None and not include_nones:
            return _OMIT
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, complex):
            return [value.real, value.imag]
        if isinstance(value, Path):
            return _serialize_path(value)
        if isinstance(value, Mapping):
            converted: dict[str, Any] = {}
            for k, v in value.items():
                serialized = SerializableConfig._to_serializable(v, include_nones=True)
                if serialized is not _OMIT:
                    converted[str(k)] = serialized
            if not converted and not include_nones:
                return _OMIT
            return converted
        if isinstance(value, (list, tuple)):
            converted = [
                item
                for v in value
                if (item := SerializableConfig._to_serializable(v, include_nones=True))
                is not _OMIT
            ]
            if not converted and not include_nones:
                return _OMIT
            return converted
        if isinstance(value, SerializableConfig):
            return value.to_dict(include_nones=include_nones)
        return value

    def to_dict(self, *, include_nones: bool = True) -> dict[str, Any]:
        """Serialize this configuration to a JSON-friendly dictionary."""
        parameters: dict[str, Any] = {}
        for field in attrs.fields(type(self)):
            if not field.init:
                continue
            value = getattr(self, field.name)
            if isinstance(value, Path) and field.metadata.get("input_path"):
                serialized = _serialize_path(value, input_path=True)
            else:
                serialized = self._to_serializable(value, include_nones=include_nones)
            if serialized is not _OMIT:
                parameters[field.name] = serialized
        return {
            CONFIG_TYPE_STR: self.CONFIG_TYPE,
            SUBCLASS_STR: self.SUBCLASS,
            PARAMETERS_STR: parameters,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        overrides: dict[str, Any] | None = None,
    ) -> "SerializableConfig":
        """
        Deserialize any registered configuration subclass from a dictionary.

        Args:
            payload: The dictionary to deserialize.
            overrides: Optional values merged into *payload* before deserialization.
                Nested dicts are merged recursively (e.g. ``{"parameters": {"energy": 1.0}}``).

        Returns:
            The deserialized configuration object.
        """
        concrete_cls = SerializableConfig._payload_class(payload)
        domain_type = payload[CONFIG_TYPE_STR]
        concrete_type = payload[SUBCLASS_STR]
        domain_cls = SerializableConfig._DOMAIN_REGISTRY[domain_type]

        # If the domain base class overrides from_dict (e.g. to strip
        # informational keys), delegate to it so preprocessing runs before
        # unknown-parameter validation.  The domain override calls
        # super().from_dict() which re-enters here with cls != SerializableConfig,
        # so this branch runs at most once.
        if cls is SerializableConfig and "from_dict" in domain_cls.__dict__:
            return domain_cls.from_dict(payload, overrides=overrides)

        payload = _apply_payload_overrides(payload, overrides)

        # If from_dict is called on a specific domain class, the payload's
        # config_type must resolve to that class (or one of its subclasses).
        if (
            cls is not SerializableConfig
            and not issubclass(domain_cls, cls)
            and not issubclass(concrete_cls, cls)
        ):
            raise ValueError(
                f"Payload '{CONFIG_TYPE_STR}' '{domain_type}', '{SUBCLASS_STR}' '{concrete_type}' resolves to "
                f"{concrete_cls.__name__}, which is not a subclass of {cls.__name__}."
            )

        parameters = payload.get(PARAMETERS_STR, {})
        if not isinstance(parameters, dict):
            raise ValueError(f"Missing or invalid '{PARAMETERS_STR}' in payload.")

        allowed_parameters = set(concrete_cls._parameter_names())
        unknown_parameters = sorted(set(parameters) - allowed_parameters)
        if unknown_parameters:
            raise ValueError(
                f"Unknown parameter(s) for '{domain_type}.{concrete_type}': "
                f"{unknown_parameters}."
            )

        completed_parameters = _resolve_input_paths(
            concrete_cls,
            _complete_init_parameters(concrete_cls, parameters),
        )

        try:
            return concrete_cls(**completed_parameters)
        except TypeError as exc:
            raise ValueError(
                f"Failed to construct '{domain_type}.{concrete_type}' "
                f"from '{PARAMETERS_STR}': {completed_parameters}."
            ) from exc

    @staticmethod
    def _field_example_value(field: attrs.Attribute, type_hints: dict[str, Any]) -> Any:
        """Resolve the example value for one init field."""
        if field.default is not attrs.NOTHING:
            if isinstance(field.default, attrs.Factory):
                return field.default.factory()
            return field.default

        tp = type_hints.get(field.name)
        return _default_for_example_type(tp) if tp is not None else None

    @classmethod
    def example_dict(cls, *, include_nones: bool = True) -> dict[str, Any]:
        """Build an example payload dict with default values for each init field.

        Intended for concrete subclasses. Fields with an explicit attrs default
        use that value (e.g. optional ``str | None`` with default ``"H"``).
        Otherwise each field type is mapped to a sensible zero-value (float ->
        0.0, str -> "", int -> 0, etc.). Optional (union with None) fields
        without an explicit default use null.

        Returns:
            The example payload dictionary.
        """
        type_hints = typing.get_type_hints(cls)
        init_fields = [a for a in attrs.fields(cls) if a.init]

        parameters = {}
        for a in init_fields:
            serialized = SerializableConfig._to_serializable(
                cls._field_example_value(a, type_hints),
                include_nones=include_nones,
            )
            if serialized is not _OMIT:
                parameters[a.name] = serialized

        return {
            CONFIG_TYPE_STR: getattr(cls, "CONFIG_TYPE", ""),
            SUBCLASS_STR: getattr(cls, "SUBCLASS", ""),
            PARAMETERS_STR: parameters,
        }

    @classmethod
    def example_json(
        cls,
        file_name: str | Path | None = None,
        *,
        indent: int = 2,
        include_nones: bool = True,
    ) -> str:
        """
        Return an example JSON serialization with default values for each field.

        Args:
            file_name: The file to write the example JSON to. If None, it is not written to a file. Defaults to None.
            indent: The indentation level for the JSON. Defaults to 2.

        Returns:
            The example JSON string.
        """
        out = json.dumps(
            cls.example_dict(include_nones=include_nones),
            indent=indent,
            sort_keys=False,
        )
        if file_name is not None:
            _write_text_file(file_name, out)
        return out

    @classmethod
    def from_json(
        cls,
        payload: str,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> "SerializableConfig":
        """
        Deserialize a configuration from a JSON string.

        Args:
            payload: The JSON string to deserialize.
            overrides: Optional values merged into the parsed payload before
                deserialization. See :meth:`from_dict`.

        Returns:
            The deserialized configuration object.
        """
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("Config JSON payload must be a JSON object.")
        return cls.from_dict(parsed, overrides=overrides)

    def to_json(self, *, indent: int = 2, include_nones: bool = True) -> str:
        """
        Serialize this configuration to a JSON string.

        Args:
            indent: The indentation level for the JSON. Defaults to 2.

        Returns:
            The JSON string.
        """
        return json.dumps(
            self.to_dict(include_nones=include_nones), indent=indent, sort_keys=False
        )

    def to_json_file(
        self, path: str | Path, *, indent: int = 2, include_nones: bool = True
    ) -> Path:
        """
        Write this configuration to a JSON file.

        Args:
            path: The path to the JSON file to write.
            indent: The indentation level for the JSON. Defaults to 2.

        Returns:
            The path to the written JSON file.
        """
        output_path = Path(path).resolve()
        token = _config_serialize_relative_to.set(output_path.parent)
        try:
            _write_text_file(
                path,
                self.to_json(indent=indent, include_nones=include_nones),
            )
        finally:
            _config_serialize_relative_to.reset(token)
        return output_path

    def _dump_yaml_with_comments(
        self,
        data: dict[str, Any],
        *,
        indent: int = 2,
        comments: bool = True,
        round_trip_comments: bool = True,
        nested_flow_style: bool = True,
    ) -> str:
        """Shared pipeline: dump *data* to YAML and optionally inject comments."""
        return dump_yaml_with_comments(
            data,
            self._yaml_descriptions_for_dump(
                data, round_trip_comments=round_trip_comments
            ),
            indent=indent,
            comments=comments,
            header_key=CONFIG_TYPE_STR,
            block_key=PARAMETERS_STR,
            flow_style_paths=(
                _nested_flow_style_paths(data) if nested_flow_style else ()
            ),
        )

    @classmethod
    def example_yaml(
        cls,
        file_name: str | Path | None = None,
        *,
        indent: int = 2,
        comments: bool = True,
        include_nones: bool = True,
        nested_flow_style: bool = True,
    ) -> str:
        """
        Return an example YAML serialization with default values for each field.

        Args:
            file_name: The file to write the example YAML to. If None, it is not written to a file. Defaults to None.
            indent: The indentation level for the YAML. Defaults to 2.
            comments: Whether to include comments in the YAML. Defaults to True.
            nested_flow_style: Whether nested dict and list attrs use YAML flow
                style. Defaults to True.

        Returns:
            The example YAML string.
        """
        example_data = cls.example_dict(include_nones=include_nones)
        out = dump_yaml_with_comments(
            example_data,
            cls._docstring_descriptions_for(example_data),
            indent=indent,
            comments=comments,
            header_key=CONFIG_TYPE_STR,
            block_key=PARAMETERS_STR,
            flow_style_paths=(
                _nested_flow_style_paths(example_data) if nested_flow_style else ()
            ),
        )
        if file_name is not None:
            _write_text_file(file_name, out)
        return out

    @classmethod
    def from_yaml(
        cls,
        payload: str,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> "SerializableConfig":
        """
        Deserialize a configuration from a YAML string.

        Inline and block ``#`` comments in the source are stored on the returned
        instance for round-trip serialization via :meth:`to_yaml`.

        Args:
            payload: The YAML string to deserialize.
            overrides: Optional values merged into the parsed payload before
                deserialization. See :meth:`from_dict`.

        Returns:
            The deserialized configuration object.
        """
        comment_map = extract_yaml_comments(
            payload, header_key=CONFIG_TYPE_STR, block_key=PARAMETERS_STR
        )
        parsed = yaml.safe_load(payload)
        if not isinstance(parsed, dict):
            raise ValueError("Config YAML payload must be a YAML object.")
        obj = cls.from_dict(parsed, overrides=overrides)
        obj._ingest_yaml_comment_map(comment_map)
        return obj

    def to_yaml(
        self,
        *,
        indent: int = 2,
        comments: bool = True,
        round_trip_comments: bool = True,
        include_nones: bool = True,
        nested_flow_style: bool = True,
    ) -> str:
        """
        Serialize this configuration to a YAML string.

        Args:
            indent: The indentation level for the YAML. Defaults to 2.
            comments: Whether to include comments in the YAML. Defaults to True.
            round_trip_comments: When ``True`` and this instance was loaded from
                commented YAML, emit those stored comments; otherwise use
                docstring-derived comments. Ignored when ``comments`` is ``False``.
            nested_flow_style: Whether nested dict and list attrs use YAML flow
                style. Defaults to True.

        Returns:
            The YAML string.
        """
        return self._dump_yaml_with_comments(
            self.to_dict(include_nones=include_nones),
            indent=indent,
            comments=comments,
            round_trip_comments=round_trip_comments,
            nested_flow_style=nested_flow_style,
        )

    def to_yaml_file(
        self,
        path: str | Path,
        *,
        indent: int = 2,
        comments: bool = True,
        round_trip_comments: bool = True,
        include_nones: bool = True,
        nested_flow_style: bool = True,
    ) -> Path:
        """
        Serialize this configuration to a YAML file.

        Args:
            path: The path to the YAML file to write.
            indent: The indentation level for the YAML. Defaults to 2.
            comments: Whether to include comments in the YAML. Defaults to True.
            round_trip_comments: When ``True`` and this instance was loaded from
                commented YAML, emit those stored comments; otherwise use
                docstring-derived comments.
            nested_flow_style: Whether nested dict and list attrs use YAML flow
                style. Defaults to True.

        Returns:
            The path to the written YAML file.
        """
        output_path = Path(path).resolve()
        token = _config_serialize_relative_to.set(output_path.parent)
        try:
            _write_text_file(
                path,
                self.to_yaml(
                    indent=indent,
                    comments=comments,
                    round_trip_comments=round_trip_comments,
                    include_nones=include_nones,
                    nested_flow_style=nested_flow_style,
                ),
            )
        finally:
            _config_serialize_relative_to.reset(token)
        return output_path

    @classmethod
    def _nested_path_relative_to(cls) -> Path | None:
        """Directory for resolving nested config path strings during deserialization."""
        anchor = _config_path_anchor.get(None)
        if anchor is not None:
            return anchor if anchor.is_dir() else anchor.parent
        source = _config_load_source.get(None)
        return source.parent if source is not None else None

    @classmethod
    @contextmanager
    def resolving_paths_relative_to(cls, directory: str | Path):
        """Resolve nested config path strings relative to *directory* within the block."""
        directory = Path(directory)
        base = directory if directory.is_dir() else directory.parent
        token = _config_path_anchor.set(base)
        try:
            yield
        finally:
            _config_path_anchor.reset(token)

    @classmethod
    def _resolve_config_path(
        cls,
        path: str | Path,
        *,
        relative_to: Path | None = None,
        accept_dir: bool = False,
    ) -> Path:
        """
        Resolve *path* to an existing file, or file or directory when *accept_dir* is True.

        Relative paths are checked against *relative_to* (if given) and then the
        process working directory. This avoids dependence on incidental ``chdir``
        calls when nested configs reference sibling or child files.
        """
        path = Path(path)
        if path.is_absolute():
            resolved = path.resolve()
            if resolved.exists() and (accept_dir or resolved.is_file()):
                return resolved

        if relative_to is None:
            relative_to = cls._nested_path_relative_to()

        candidates: list[Path] = []
        if relative_to is not None:
            base = relative_to if relative_to.is_dir() else relative_to.parent
            candidates.append(base / path)
        candidates.append(Path.cwd() / path)

        tried: list[Path] = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in tried:
                continue
            tried.append(resolved)
            if accept_dir:
                if resolved.exists():
                    return resolved
            elif resolved.is_file():
                return resolved

        requirement = "file or directory" if accept_dir else "file"
        tried_display = ", ".join(str(p) for p in tried) or str(path)
        raise ValueError(
            f"Invalid path: {path}. Must be a {requirement}. Tried: {tried_display}"
        )

    @classmethod
    def _load_from_resolved_path(
        cls,
        resolved: Path,
        *,
        format: str,
        overrides: dict[str, Any] | None = None,
    ) -> "SerializableConfig":
        token = _config_load_source.set(resolved)
        try:
            payload = resolved.read_text(encoding="utf-8")
            if format == "json":
                obj = cls.from_json(payload, overrides=overrides)
            elif format == "yaml":
                obj = cls.from_yaml(payload, overrides=overrides)
            else:
                raise ValueError(f"Unsupported config file format: {format!r}.")
            _set_instance_attr(obj, "source_file", resolved)
            return obj
        finally:
            _config_load_source.reset(token)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        relative_to: Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "SerializableConfig":
        """
        Deserialize a configuration from a file.

        Relative paths are resolved against *relative_to* when provided, otherwise
        against the directory of the config file currently being loaded (for nested
        references), and finally against the process working directory.

        Args:
            path: Path to a ``.json``, ``.jsn``, ``.yaml``, or ``.yml`` config file.
            relative_to: Optional base directory for resolving a relative *path*.
            overrides: Optional values merged into the file payload before
                deserialization. See :meth:`from_dict`.
        """
        if relative_to is None:
            relative_to = cls._nested_path_relative_to()
        resolved = cls._resolve_config_path(path, relative_to=relative_to)
        suffix = resolved.suffix.lower()
        if suffix in [".json", ".jsn"]:
            return cls._load_from_resolved_path(
                resolved, format="json", overrides=overrides
            )
        if suffix in [".yaml", ".yml"]:
            return cls._load_from_resolved_path(
                resolved, format="yaml", overrides=overrides
            )
        raise ValueError(
            f"Unsupported file type: {resolved.suffix}. "
            "Must be .json, .jsn, .yaml, or .yml (case-insensitive)."
        )

    @staticmethod
    def _resolve_string_as_dict(payload: str) -> dict[str, Any]:
        """Parse a YAML or JSON string into a config payload dictionary."""
        try:
            parsed = yaml.safe_load(payload)
        except yaml.YAMLError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("Config string must be a YAML or JSON object.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Config string must be a YAML or JSON object.")
        return parsed

    @classmethod
    def from_any(
        cls,
        value: "SerializableConfig | str | Path | dict[str, Any]",
        *,
        relative_to: Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "SerializableConfig":
        """
        Deserialize a configuration from any supported source.

        Resolution order:

        1. Return *value* unchanged if it is already a :class:`SerializableConfig`.
        2. Load from a file when *value* is a :class:`~pathlib.Path`.
        3. Deserialize from a mapping when *value* is a ``dict``.
        4. For a ``str``, try loading from a file path first; otherwise parse
           as YAML or JSON and deserialize the resulting mapping.
        5. Raise :class:`TypeError` for unsupported types.

        Args:
            value: An existing config, file path, mapping, or serialized string.
            relative_to: Optional base directory for resolving a relative path.
            overrides: Optional values merged into the payload before
                deserialization. See :meth:`from_dict`.
        """
        if isinstance(value, SerializableConfig):
            return value

        if isinstance(value, Path):
            return cls.from_file(value, relative_to=relative_to, overrides=overrides)

        if isinstance(value, dict):
            return cls.from_dict(value, overrides=overrides)

        if isinstance(value, str):
            try:
                return cls.from_file(
                    value, relative_to=relative_to, overrides=overrides
                )
            except ValueError:
                return cls.from_dict(
                    cls._resolve_string_as_dict(value), overrides=overrides
                )

        raise TypeError(
            f"Cannot deserialize config from {type(value).__name__!r}; "
            "expected SerializableConfig, path, dict, or str."
        )


def _register_config_subclass(cls: type["SerializableConfig"]) -> None:
    """Register domain and concrete config subclasses from class-body declarations."""
    if "CONFIG_TYPE" in cls.__dict__:
        config_type = cls.__dict__["CONFIG_TYPE"]
        if config_type == NULL_CONCRETE_STR:
            return
        existing = SerializableConfig._DOMAIN_REGISTRY.get(config_type)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Duplicate CONFIG_TYPE '{config_type}' for "
                f"{cls.__name__} and {existing.__name__}."
            )
        SerializableConfig._DOMAIN_REGISTRY[config_type] = cls

    if "SUBCLASS" in cls.__dict__ and not getattr(
        cls, "__abstractmethods__", frozenset()
    ):
        subclass_key = cls.__dict__["SUBCLASS"]
        if subclass_key == NULL_CONCRETE_STR:
            return
        registry = cls._CONCRETE_REGISTRY
        existing = registry.get(subclass_key)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Duplicate SUBCLASS '{subclass_key}' for "
                f"{cls.__name__} and {existing.__name__}."
            )
        registry[subclass_key] = cls

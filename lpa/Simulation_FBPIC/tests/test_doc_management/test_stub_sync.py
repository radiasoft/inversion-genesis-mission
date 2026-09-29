"""Tests for AST parsing, LibIndex, and .pyi stub generation."""

from __future__ import annotations

import ast
import shutil
import textwrap
from pathlib import Path

import pytest

from inversion_fbpic.lib._doc_management.index import (
    InitField,
    LibIndex,
    ParsedClass,
    _class_fields,
    _class_methods,
    _field_default,
    _init_fields,
    _non_init_field_names,
    _parse_module,
    default_lib_dir,
    lib_module_names,
    module_source_path,
)
from inversion_fbpic.lib._doc_management.render import (
    _quote_docstring,
    generate_module_pyi,
)
from inversion_fbpic.lib._doc_management.stub_sync import sync_config_stubs

LIB_DIR = default_lib_dir()


@pytest.fixture()
def lib_copy(tmp_path: Path) -> Path:
    """Copy source ``.py`` files to a temp directory for write-safe sync tests."""
    for module in lib_module_names(LIB_DIR):
        destination = module_source_path(tmp_path, module)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(module_source_path(LIB_DIR, module), destination)
    return tmp_path


def _index() -> LibIndex:
    return LibIndex.from_lib_dir(LIB_DIR)


def _parse_class(source: str, class_name: str = "Example") -> ast.ClassDef:
    module = ast.parse(textwrap.dedent(source))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"class {class_name!r} not found")


def _field_call_from_class(source: str, class_name: str = "Example") -> ast.Call:
    node = _parse_class(source, class_name)
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, ast.Call):
            return stmt.value
    raise AssertionError("no attrs.field assignment found")


# ---------------------------------------------------------------------------
# Integration tests (real lib modules)
# ---------------------------------------------------------------------------


def test_merged_docstring_includes_parent_args() -> None:
    index = _index()
    merged = index.merged_docstring(
        index.classes[("density_profiles", "AsymmetricSine")]
    )
    assert "nominal_density:" in merged
    assert "peak_z0:" in merged
    assert "Asymmetric sine-squared density profile." in merged


def test_subclass_summary_is_preserved() -> None:
    index = _index()
    merged = index.merged_docstring(index.classes[("laser", "GaussianLaserPulse")])
    assert "Temporal-gaussian laser pulse with a Gaussian transverse profile." in merged
    assert "energy:" in merged
    assert "wavelength:" in merged


def test_parameter_order_follows_init_fields() -> None:
    index = _index()
    merged = index.merged_docstring(
        index.classes[("density_profiles", "AsymmetricSine")]
    )
    assert merged.index("nominal_density:") < merged.index("peak_z0:")


def test_generate_module_pyi_contains_merged_docstrings() -> None:
    pyi = generate_module_pyi("density_profiles", lib_dir=LIB_DIR)
    assert "class AsymmetricSine(_DensityProfile):" in pyi
    assert "nominal_density:" in pyi
    assert "peak_z0:" in pyi
    assert "def __init__(" in pyi
    assert "peak_z0: float" in pyi


def test_generated_docstrings_indent_section_headers_and_entries() -> None:
    pyi = generate_module_pyi("density_profiles", lib_dir=LIB_DIR)
    class_block = pyi.split("class ExampleDensityProfile(_DensityProfile):")[1]
    method_block = class_block.split("    def build_density_function(")[1].split(
        "\n\nclass "
    )[0]

    assert "    Args:" in class_block
    assert "        length:" in class_block
    assert "        Returns:" in method_block
    assert "            A callable function" in method_block


def test_smooth_sine_flattop_constructor_has_normalized_docstring() -> None:
    pyi = generate_module_pyi("density_profiles", lib_dir=LIB_DIR)
    constructor_block = pyi.split("class SmoothSineFlattop(_DensityProfile):")[1].split(
        "    def get_z_extent("
    )[0]

    assert (
        "        The profile is nonzero, starting at offset_length; "
        "the flattop starts at offset_length + upramp_length." in constructor_block
    )
    assert "        Args:" in constructor_block
    assert "            flattop_width:" in constructor_block


def test_init_signature_includes_defaults() -> None:
    pyi = generate_module_pyi("density_profiles", lib_dir=LIB_DIR)
    assert "species: str | None = 'H'" in pyi
    assert "elec_name: str | None = None" in pyi
    assert "ionization: int | None = 0" in pyi
    assert "start_position: float = 0.0" in pyi


def test_init_signature_includes_simulation_defaults() -> None:
    pyi = generate_module_pyi("simulation", lib_dir=LIB_DIR)
    assert "save_checkpoints: bool = False" in pyi
    assert "r_boundary: Literal['open', 'reflective'] = 'reflective'" in pyi
    assert "import logging" in pyi
    assert "from pathlib import Path" in pyi
    assert pyi.count("from .serializable_config import SerializableConfig") == 1


def test_sync_config_stubs_writes_pyi_files(lib_copy: Path) -> None:
    sync_config_stubs(lib_dir=lib_copy, check=False)
    for module in lib_module_names(lib_copy):
        pyi = module_source_path(lib_copy, module).with_suffix(".pyi")
        assert pyi.is_file()
        assert pyi.read_text(encoding="utf-8").startswith(
            "# Auto-generated by tools/sync_config_docstrings.py"
        )


def test_generate_nested_package_pyi_contains_merged_docstrings() -> None:
    pyi = generate_module_pyi(
        "_density_implementations.interpolate_from_h5_profile", lib_dir=LIB_DIR
    )
    assert "from ..density_core import _DensityProfile" in pyi
    assert "class InterpolateFromH5Profile(_DensityProfile):" in pyi
    assert "nominal_density:" in pyi
    assert "lineout_axis:" in pyi


def test_density_profiles_pyi_explicitly_reexports_h5_profile() -> None:
    pyi = generate_module_pyi("density_profiles", lib_dir=LIB_DIR)
    assert (
        "from ._density_implementations.interpolate_from_h5_profile import "
        "InterpolateFromH5Profile as InterpolateFromH5Profile"
    ) in pyi
    assert "InterpolateFromH5Profile = _InterpolateFromH5Profile" not in pyi


def test_sync_config_stubs_check_detects_stale_file(lib_copy: Path) -> None:
    sync_config_stubs(lib_dir=lib_copy, check=False)
    pyi_path = lib_copy / "density_core.pyi"
    pyi_path.write_text(pyi_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    stale = sync_config_stubs(lib_dir=lib_copy, check=True)
    assert pyi_path in stale


# ---------------------------------------------------------------------------
# AST parsing edge cases
# ---------------------------------------------------------------------------


def test_parse_module_ignores_non_attrs_classes(tmp_path: Path) -> None:
    path = tmp_path / "example.py"
    path.write_text(
        textwrap.dedent(
            """
            class PlainClass:
                '''Not an attrs config.'''

            import attrs

            @attrs.define(kw_only=True)
            class Config(SerializableConfig):
                '''Config class.'''
                value: int = attrs.field()
            """
        ),
        encoding="utf-8",
    )
    classes, _ = _parse_module(path)
    assert set(classes) == {("example", "Config")}


def test_init_fields_excludes_field_level_init_false() -> None:
    node = _parse_class(
        """
        import attrs

        @attrs.define(kw_only=True)
        class Example:
            public: int = attrs.field()
            hidden: int = attrs.field(init=False, default=0)
        """
    )
    fields = _init_fields(node)
    assert fields == (InitField("public", "int", None),)
    assert _non_init_field_names(node) == ("hidden",)


def test_init_fields_excludes_all_fields_when_class_init_false() -> None:
    node = _parse_class(
        """
        import attrs

        @attrs.define(kw_only=True, init=False)
        class Example:
            value: int = attrs.field()
        """
    )
    assert _init_fields(node) == ()


def test_field_default_parses_literals_and_marks_factory_optional() -> None:
    none_node = _field_call_from_class(
        """
        import attrs

        @attrs.define(kw_only=True)
        class Example:
            value: str | None = attrs.field(default=None)
        """
    )
    list_node = _field_call_from_class(
        """
        import attrs

        @attrs.define(kw_only=True)
        class Example:
            items: list[str] = attrs.field(default=[])
        """
    )
    factory_node = _field_call_from_class(
        """
        import attrs

        @attrs.define(kw_only=True)
        class Example:
            items: list[str] = attrs.field(factory=list)
        """
    )
    attrs_factory_node = _field_call_from_class(
        """
        import attrs

        @attrs.define(kw_only=True)
        class Example:
            items: list[str] = attrs.field(default=attrs.Factory(list))
        """
    )

    assert _field_default(none_node) == "None"
    assert _field_default(list_node) == "[]"
    assert _field_default(factory_node) == "..."
    assert _field_default(attrs_factory_node) == "..."


def test_quote_docstring_escapes_embedded_triple_quotes() -> None:
    quoted = _quote_docstring('Summary with """embedded""" quotes.')
    assert '\\"\\"\\"' in quoted
    assert quoted.startswith('"""')
    assert quoted.endswith('"""')


# ---------------------------------------------------------------------------
# LibIndex / stub generation edge cases
# ---------------------------------------------------------------------------


def test_mro_merges_three_level_inheritance_fields() -> None:
    index = _index()
    fields = index.init_fields(index.classes[("laser", "GaussianLaserPulse")])
    names = [field.name for field in fields]
    assert names.index("energy") < names.index("wavelength")
    assert names.index("wavelength") < names.index("polarization")


def test_class_with_no_init_fields_has_no_generated_init_stub() -> None:
    pyi = generate_module_pyi("density_core", lib_dir=LIB_DIR)
    density_modifier_block = pyi.split("class _DensityModifier(SerializableConfig):")[1]
    density_profile_block = density_modifier_block.split("class _DensityProfile")[0]
    assert "def __init__(" not in density_profile_block


def test_args_without_blank_line_is_merged_for_real_asymmetric_sine() -> None:
    index = _index()
    merged = index.merged_docstring(
        index.classes[("density_profiles", "AsymmetricSine")]
    )
    assert "peak_z0:" in merged
    assert "nominal_density:" in merged


def test_modified_density_profile_includes_list_default() -> None:
    pyi = generate_module_pyi("density_modifiers", lib_dir=LIB_DIR)
    assert "modifiers:" in pyi
    assert (
        "= []"
        in pyi.split("class ModifiedDensityProfile(_DensityProfile):")[1].split(
            "class "
        )[0]
    )


def test_lasy_laser_pulse_inherits_parent_init_fields_despite_init_false() -> None:
    """Document current behavior: class-level init=False still inherits parent attrs fields."""
    index = _index()
    fields = index.init_fields(index.classes[("laser", "LasyLaserPulse")])
    assert {field.name for field in fields} == {
        "energy",
        "a0",
        "z0",
        "method",
        "z0_antenna",
        "v_antenna",
    }


def test_generated_h5_profile_stub_excludes_overridden_non_init_field() -> None:
    pyi = generate_module_pyi(
        "_density_implementations.interpolate_from_h5_profile", lib_dir=LIB_DIR
    )
    init_block = pyi.split("class InterpolateFromH5Profile(_DensityProfile):")[1].split(
        "    def build_density_function("
    )[0]
    assert "nominal_density: float," not in init_block
    assert "filename: str | Path," in init_block
    assert "interpolation_points: dict[str, float] = ...," in init_block


def test_init_fields_honors_closest_field_override() -> None:
    parent = ParsedClass(
        name="Parent",
        module="parent_mod",
        bases=(),
        docstring=None,
        init_fields=(InitField("value", "int", None),),
    )
    child = ParsedClass(
        name="Child",
        module="child_mod",
        bases=("Parent",),
        docstring=None,
        init_fields=(),
        non_init_field_names=("value",),
    )
    index = LibIndex(
        classes={
            ("parent_mod", "Parent"): parent,
            ("child_mod", "Child"): child,
        },
        imports={"child_mod": {"Parent": "parent_mod"}},
    )

    assert index.init_fields(child) == []


def test_generate_module_pyi_returns_empty_for_unknown_module() -> None:
    index = _index()
    assert generate_module_pyi("not_a_real_module", index=index) == ""


def test_generate_module_pyi_adds_cross_module_base_import() -> None:
    pyi = generate_module_pyi("density_core", lib_dir=LIB_DIR)
    assert "from .serializable_config import SerializableConfig" in pyi


def test_generated_stubs_define_annotation_dependencies() -> None:
    density_core_pyi = generate_module_pyi("density_core", lib_dir=LIB_DIR)
    laser_pyi = generate_module_pyi("laser", lib_dir=LIB_DIR)
    simulation_pyi = generate_module_pyi("simulation", lib_dir=LIB_DIR)

    assert (
        "DensityCallable = Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]"
        in density_core_pyi
    )
    assert "from fbpic.main import Simulation as FBPICSimulation" in density_core_pyi
    assert "from fbpic.particles.particles import Particles" in density_core_pyi
    assert "from fbpic.lpa_utils.laser.laser_profiles import LaserProfile" in laser_pyi
    assert "from .density_core import _DensityProfile" in simulation_pyi
    assert "from .laser import _LaserPulse" in simulation_pyi
    assert (
        "from fbpic.main import Simulation as FBPICSimulation, Particles"
        in simulation_pyi
    )


def test_sync_config_stubs_check_passes_when_up_to_date(lib_copy: Path) -> None:
    sync_config_stubs(lib_dir=lib_copy, check=False)
    assert sync_config_stubs(lib_dir=lib_copy, check=True) == []


def test_manual_index_resolves_relative_imported_base() -> None:
    child = ParsedClass(
        name="Child",
        module="child_mod",
        bases=("Parent",),
        docstring="Child.\n\nArgs:\n    child_only: Child field.",
        init_fields=(InitField("child_only", "float", None),),
    )
    parent = ParsedClass(
        name="Parent",
        module="parent_mod",
        bases=("SerializableConfig",),
        docstring="Parent.\n\nArgs:\n    parent_only: Parent field.",
        init_fields=(InitField("parent_only", "int", None),),
    )
    index = LibIndex(
        classes={
            ("child_mod", "Child"): child,
            ("parent_mod", "Parent"): parent,
        },
        imports={"child_mod": {"Parent": "parent_mod"}},
    )
    merged = index.merged_docstring(child)
    assert "parent_only:" in merged
    assert "child_only:" in merged
    assert merged.index("parent_only:") < merged.index("child_only:")


# ---------------------------------------------------------------------------
# Public method stub tests
# ---------------------------------------------------------------------------


def test_serializable_config_stub_includes_public_api() -> None:
    pyi = generate_module_pyi("serializable_config", lib_dir=LIB_DIR)
    assert "def to_yaml(" in pyi
    assert "def from_file(" in pyi
    assert "@classmethod" in pyi
    assert "def from_dict(" in pyi


def test_density_profile_stub_includes_abstract_methods() -> None:
    pyi = generate_module_pyi("density_core", lib_dir=LIB_DIR)
    block = pyi.split("class _DensityProfile")[1].split("\nclass ")[0]
    assert "def get_z_extent(" in block
    assert "def build_density_function(" in block
    assert "@abstractmethod" in block


def test_simulation_stub_includes_properties() -> None:
    pyi = generate_module_pyi("simulation", lib_dir=LIB_DIR)
    assert "@property" in pyi
    assert "def is_boosted(self) -> bool:" in pyi
    assert "def setup_simulation(" in pyi


def test_class_methods_skips_private_and_dunder() -> None:
    node = _parse_class(
        """
        import attrs

        @attrs.define(kw_only=True)
        class Example:
            value: int = attrs.field()

            def public(self) -> None: ...
            def _private(self) -> None: ...
            def __repr__(self) -> str: ...
            def __attrs_post_init__(self) -> None: ...
        """
    )
    methods = _class_methods(node)
    names = [m.name for m in methods]
    assert names == ["public"]


def test_class_methods_captures_decorators_and_return() -> None:
    node = _parse_class(
        """
        import attrs

        @attrs.define(kw_only=True)
        class Example:
            value: int = attrs.field()

            @property
            def total(self) -> int: ...

            @classmethod
            def create(cls, value: int) -> 'Example': ...
        """
    )
    methods = _class_methods(node)
    by_name = {m.name: m for m in methods}
    assert "property" in by_name["total"].decorators
    assert by_name["total"].return_annotation == "int"
    assert "classmethod" in by_name["create"].decorators
    assert by_name["create"].return_annotation == "'Example'"


# ---------------------------------------------------------------------------
# Field declaration tests
# ---------------------------------------------------------------------------


def test_simulation_stub_includes_field_declarations() -> None:
    pyi = generate_module_pyi("simulation", lib_dir=LIB_DIR)
    block = pyi.split("class Simulation(SerializableConfig):")[1].split("\nclass ")[0]
    assert "hyparams: SimulationHyperparameters | None" in block
    assert "is_setup: bool" in block
    assert "T_interact: float" in block


def test_serializable_config_stub_includes_classvar_fields() -> None:
    pyi = generate_module_pyi("serializable_config", lib_dir=LIB_DIR)
    assert "CONFIG_TYPE: ClassVar[str]" in pyi
    assert "SUBCLASS: ClassVar[str]" in pyi
    assert "ClassVar" in pyi


def test_class_fields_skips_private_names() -> None:
    node = _parse_class(
        """
        import attrs
        from typing import ClassVar

        @attrs.define(kw_only=True)
        class Example:
            public: int = attrs.field()
            _private: int = attrs.field(init=False, default=0)
            CONFIG_TYPE: ClassVar[str] = "example"
        """
    )
    fields = _class_fields(node)
    names = [f.name for f in fields]
    assert "public" in names
    assert "CONFIG_TYPE" in names
    assert "_private" not in names


def test_class_fields_detects_classvar() -> None:
    node = _parse_class(
        """
        import attrs
        from typing import ClassVar

        @attrs.define(kw_only=True)
        class Example:
            value: int = attrs.field()
            CONFIG_TYPE: ClassVar[str] = "example"
        """
    )
    fields = _class_fields(node)
    by_name = {f.name: f for f in fields}
    assert not by_name["value"].is_classvar
    assert by_name["CONFIG_TYPE"].is_classvar

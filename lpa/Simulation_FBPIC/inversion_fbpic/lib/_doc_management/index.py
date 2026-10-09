"""AST-based class index for ``@attrs.define`` config modules.

Parses Python source files (without importing them) to discover attrs config
classes, their inheritance hierarchies, docstrings, and init fields.  The
resulting :class:`LibIndex` drives both ``.pyi`` stub generation and runtime
docstring merging.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from . import merge

ClassKey = tuple[str, str]
"""Registry key of the form ``(module_stem, class_name)``."""

_LIB_MODULE_STEMS = (
    "serializable_config",
    "density_core",
    "density_profiles",
    "density_modifiers",
    "laser",
    "simulation",
)
_CONFIG_PACKAGE_DIRS = ("_density_implementations", "_laser_implementations")


def lib_module_names(lib_dir: Path) -> tuple[str, ...]:
    """Return config module names, including supported implementation packages."""
    package_modules = [
        path.relative_to(lib_dir).with_suffix("").as_posix().replace("/", ".")
        for package_dir in _CONFIG_PACKAGE_DIRS
        for path in sorted((lib_dir / package_dir).rglob("*.py"))
        if path.name != "__init__.py"
    ]
    return (*_LIB_MODULE_STEMS, *package_modules)


def module_source_path(lib_dir: Path, module: str) -> Path:
    """Return the source-file path corresponding to a dotted *module* name."""
    return lib_dir.joinpath(*module.split(".")).with_suffix(".py")


def default_lib_dir() -> Path:
    """Return the canonical ``inversion_fbpic/lib/`` directory path."""
    return Path(__file__).resolve().parents[1]


class InitField(NamedTuple):
    """A single ``attrs.field()`` that participates in ``__init__``.

    Attributes:
        name: Field identifier (e.g. ``"wavelength"``).
        annotation: Source-level type annotation text (e.g. ``"float"``).
        default: Unparsed default value text, or ``None`` when the field is
            required or uses ``factory=``.
    """

    name: str
    annotation: str
    default: str | None


class ParsedField(NamedTuple):
    """An ``attrs.field()`` declaration from a config class body.

    Captures every annotated field — both init and non-init — so stub
    files can declare class-level attributes visible to Pyright.

    Attributes:
        name: Field identifier (e.g. ``"hyparams"``).
        annotation: Source-level type annotation text.
        is_classvar: ``True`` when the annotation wraps ``ClassVar[...]``.
    """

    name: str
    annotation: str
    is_classvar: bool


class ParsedMethod(NamedTuple):
    """A public method extracted from a config class body.

    Attributes:
        name: Method name (e.g. ``"to_yaml"``).
        decorators: Decorator names in declaration order (e.g.
            ``("classmethod",)``, ``("property",)``).
        params: Unparsed parameter text *after* ``self``/``cls``
            (e.g. ``"*, indent: int = 2"``).
        return_annotation: Unparsed return-type text, or ``None``.
        docstring: Raw docstring text, or ``None`` if absent.
    """

    name: str
    decorators: tuple[str, ...]
    params: str
    return_annotation: str | None
    docstring: str | None


class ParsedClass(NamedTuple):
    """Attrs config class extracted from a single source module.

    Attributes:
        name: Unqualified class name.
        module: Stem of the source file the class was parsed from.
        bases: Unqualified base-class names in declaration order.
        docstring: Raw docstring text, or ``None`` if absent.
        init_fields: Fields that appear in the generated ``__init__``.
        non_init_field_names: Field names explicitly declared with
            ``attrs.field(init=False)``.
        fields: All public annotated fields (init and non-init, including
            ``ClassVar`` declarations).
        methods: Public methods defined directly on this class.
    """

    name: str
    module: str
    bases: tuple[str, ...]
    docstring: str | None
    init_fields: tuple[InitField, ...]
    non_init_field_names: tuple[str, ...] = ()
    fields: tuple[ParsedField, ...] = ()
    methods: tuple[ParsedMethod, ...] = ()


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _call_name(node: ast.AST) -> str | None:
    """Return the simple name of a call target (``Name`` or ``Attribute``)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _kwarg_is_false(node: ast.AST, name: str) -> bool:
    """Return ``True`` if *node* is a call whose *name* keyword is ``False``."""
    if not isinstance(node, ast.Call):
        return False
    for keyword in node.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value is False
    return False


def _is_attrs_define(node: ast.AST) -> bool:
    """Return ``True`` if *node* is an ``@attrs.define`` (or ``@define``) decorator."""
    return _call_name(node.func if isinstance(node, ast.Call) else node) == "define"


def _class_docstring(node: ast.ClassDef) -> str | None:
    """Extract the class-level docstring literal, or ``None`` if absent."""
    if not node.body:
        return None
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        if isinstance(first.value.value, str):
            return first.value.value
    return None


def _base_names(node: ast.ClassDef) -> tuple[str, ...]:
    """Return unqualified base-class names from the class header."""
    names: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return tuple(names)


def _annotation_text(node: ast.expr) -> str:
    """Unparse a type annotation node back to source text."""
    return ast.unparse(node)


def _field_default(call: ast.Call) -> str | None:
    """Extract the ``default=`` value from an ``attrs.field()`` call.

    Returns:
        Unparsed default text, ``"..."`` for factory defaults, or ``None``
        when no default is present.
    """
    for keyword in call.keywords:
        if keyword.arg == "factory":
            return "..."
        if keyword.arg != "default":
            continue
        if (
            isinstance(keyword.value, ast.Call)
            and _call_name(keyword.value.func) == "Factory"
        ):
            return "..."
        return ast.unparse(keyword.value)
    return None


def _init_fields(node: ast.ClassDef) -> tuple[InitField, ...]:
    """Collect init-participating ``attrs.field()`` assignments from *node*.

    Skips the entire class when ``@attrs.define(init=False)`` is used, and
    skips individual fields declared with ``field(init=False)``.
    """
    if any(
        _is_attrs_define(dec) and _kwarg_is_false(dec, "init")
        for dec in node.decorator_list
    ):
        return ()
    fields: list[InitField] = []
    for stmt in node.body:
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue
        if (
            not isinstance(stmt.value, ast.Call)
            or _call_name(stmt.value.func) != "field"
        ):
            continue
        if _kwarg_is_false(stmt.value, "init"):
            continue
        fields.append(
            InitField(
                stmt.target.id,
                _annotation_text(stmt.annotation),
                _field_default(stmt.value),
            )
        )
    return tuple(fields)


def _non_init_field_names(node: ast.ClassDef) -> tuple[str, ...]:
    """Collect names explicitly excluded from ``__init__`` by attrs fields."""
    return tuple(
        stmt.target.id
        for stmt in node.body
        if isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and isinstance(stmt.value, ast.Call)
        and _call_name(stmt.value.func) == "field"
        and _kwarg_is_false(stmt.value, "init")
    )


def _is_classvar(node: ast.expr) -> bool:
    """Return ``True`` if *node* is a ``ClassVar[...]`` annotation."""
    if isinstance(node, ast.Subscript):
        return _call_name(node.value) == "ClassVar"
    return _call_name(node) == "ClassVar"


def _class_fields(node: ast.ClassDef) -> tuple[ParsedField, ...]:
    """Collect all public annotated fields from a class body.

    Includes both ``attrs.field()`` assignments and bare ``ClassVar``
    declarations. Private names (starting with ``_``) are excluded.
    """
    fields: list[ParsedField] = []
    for stmt in node.body:
        if not isinstance(stmt, ast.AnnAssign):
            continue
        if not isinstance(stmt.target, ast.Name):
            continue
        name = stmt.target.id
        if name.startswith("_"):
            continue
        annotation = _annotation_text(stmt.annotation)
        fields.append(ParsedField(name, annotation, _is_classvar(stmt.annotation)))
    return tuple(fields)


_KNOWN_DECORATORS = frozenset(
    {
        "classmethod",
        "staticmethod",
        "property",
        "abstractmethod",
    }
)


def _func_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """Extract the docstring from a function/method body."""
    if not node.body:
        return None
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        if isinstance(first.value.value, str):
            return first.value.value
    return None


def _unparse_params(
    node: ast.FunctionDef | ast.AsyncFunctionDef, *, skip_first: bool
) -> str:
    """Unparse the parameter list of a function, optionally dropping self/cls."""
    args = node.args
    # Build a minimal copy of the arguments node with self/cls removed.
    if skip_first and args.args:
        trimmed = ast.arguments(
            posonlyargs=args.posonlyargs,
            args=args.args[1:],
            vararg=args.vararg,
            kwonlyargs=args.kwonlyargs,
            kw_defaults=args.kw_defaults,
            kwarg=args.kwarg,
            defaults=args.defaults[max(0, len(args.defaults) - len(args.args) + 1) :],
        )
    else:
        trimmed = args
    # ast.unparse on an arguments node gives a clean param string.
    return ast.unparse(trimmed)


def _class_methods(node: ast.ClassDef) -> tuple[ParsedMethod, ...]:
    """Extract public methods from a class body."""
    methods: list[ParsedMethod] = []
    for stmt in node.body:
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = stmt.name
        if name.startswith("_"):
            continue

        dec_names: list[str] = []
        for dec in stmt.decorator_list:
            dec_name = _call_name(dec.func if isinstance(dec, ast.Call) else dec)
            if dec_name in _KNOWN_DECORATORS:
                dec_names.append(dec_name)

        is_static = "staticmethod" in dec_names
        skip_first = not is_static
        params = _unparse_params(stmt, skip_first=skip_first)
        ret = ast.unparse(stmt.returns) if stmt.returns else None

        methods.append(
            ParsedMethod(
                name=name,
                decorators=tuple(dec_names),
                params=params,
                return_annotation=ret,
                docstring=_func_docstring(stmt),
            )
        )
    return tuple(methods)


def _import_nodes(nodes: list[ast.stmt]) -> list[ast.Import | ast.ImportFrom]:
    """Return imports nested only in module-level conditional blocks."""
    imports: list[ast.Import | ast.ImportFrom] = []
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.If):
            imports.extend(_import_nodes(node.body))
            imports.extend(_import_nodes(node.orelse))
        elif isinstance(node, ast.Try):
            imports.extend(_import_nodes(node.body))
            imports.extend(_import_nodes(node.orelse))
            imports.extend(
                imported
                for handler in node.handlers
                for imported in _import_nodes(handler.body)
            )
            imports.extend(_import_nodes(node.finalbody))
    return imports


def _parse_imports(tree: ast.Module) -> dict[str, str]:
    """Build a ``{imported_name: source_module_stem}`` map from *tree*.

    ``from inversion_fbpic.lib.`` prefixes are stripped so keys like
    ``"SerializableConfig"`` map to ``"serializable_config"``.
    """
    imports: dict[str, str] = {}
    for node in _import_nodes(tree.body):
        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").removeprefix("inversion_fbpic.lib.")
            for alias in node.names:
                imports[alias.asname or alias.name] = module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name] = alias.name
    return imports


def _parse_import_lines(tree: ast.Module) -> dict[str, str]:
    """Map imported names to import statements that define them."""
    lines: dict[str, str] = {}
    for node in _import_nodes(tree.body):
        line = ast.unparse(node)
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            lines.setdefault(name, line)
    return lines


def _parse_definitions(tree: ast.Module) -> dict[str, str]:
    """Map module-level simple assignments to their source expressions."""
    definitions: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = ast.unparse(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            definitions[node.target.id] = ast.unparse(node)
    return definitions


def _parse_module(
    path: Path, *, module: str | None = None
) -> tuple[dict[ClassKey, ParsedClass], dict[str, str]]:
    """Parse a single ``.py`` file and return its attrs classes and imports.

    Args:
        path: Filesystem path to the Python source file.

    Returns:
        A two-tuple of ``(classes, imports)`` where *classes* is keyed by
        :data:`ClassKey` and *imports* maps imported names to module stems.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module = module or path.stem
    classes: dict[ClassKey, ParsedClass] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(_is_attrs_define(decorator) for decorator in node.decorator_list):
            continue
        classes[(module, node.name)] = ParsedClass(
            name=node.name,
            module=module,
            bases=_base_names(node),
            docstring=_class_docstring(node),
            init_fields=_init_fields(node),
            non_init_field_names=_non_init_field_names(node),
            fields=_class_fields(node),
            methods=_class_methods(node),
        )
    return classes, _parse_imports(tree)


def _parse_module_metadata(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return import statements and simple module-level definitions from *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _parse_import_lines(tree), _parse_definitions(tree)


# ---------------------------------------------------------------------------
# LibIndex
# ---------------------------------------------------------------------------


@dataclass
class LibIndex:
    """Cross-module registry of attrs config classes.

    Holds every :class:`ParsedClass` discovered across the lib modules and
    their import maps, enabling MRO resolution, init-field collection, and
    docstring merging without importing any runtime code.

    Attributes:
        classes: All parsed config classes keyed by :data:`ClassKey`.
        imports: Per-module import maps (``{module_stem: {name: source_stem}}``).
        import_lines: Per-module maps of imported names to defining import
            statements.
        definitions: Per-module maps of simple names to source assignments.
    """

    classes: dict[ClassKey, ParsedClass]
    imports: dict[str, dict[str, str]]
    import_lines: dict[str, dict[str, str]] | None = None
    definitions: dict[str, dict[str, str]] | None = None

    @classmethod
    def from_lib_dir(cls, lib_dir: Path) -> LibIndex:
        """Build an index by parsing every module in *lib_dir*.

        Args:
            lib_dir: Directory containing the top-level config modules and
                supported implementation packages.

        Returns:
            A fully populated :class:`LibIndex`.
        """
        classes: dict[ClassKey, ParsedClass] = {}
        imports: dict[str, dict[str, str]] = {}
        import_lines: dict[str, dict[str, str]] = {}
        definitions: dict[str, dict[str, str]] = {}
        for module in lib_module_names(lib_dir):
            path = module_source_path(lib_dir, module)
            module_classes, module_imports = _parse_module(path, module=module)
            module_import_lines, module_definitions = _parse_module_metadata(path)
            classes.update(module_classes)
            imports[module] = module_imports
            import_lines[module] = module_import_lines
            definitions[module] = module_definitions
        return cls(
            classes=classes,
            imports=imports,
            import_lines=import_lines,
            definitions=definitions,
        )

    def _resolve_base(self, base_name: str, module: str) -> ClassKey | None:
        """Resolve an unqualified base-class name to a :data:`ClassKey`.

        Lookup order: same module, then the module's import map, then a
        fallback scan of all registered class names.
        """
        if (module, base_name) in self.classes:
            return (module, base_name)
        imported = self.imports.get(module, {}).get(base_name)
        if imported is not None:
            key = (imported or module, base_name)
            return key if key in self.classes else None
        for key in self.classes:
            if key[1] == base_name:
                return key
        return None

    def resolved_bases(self, parsed: ParsedClass) -> list[tuple[str, ClassKey]]:
        """Return ``(base_name, resolved_key)`` for each resolvable base.

        Args:
            parsed: The class whose declared bases should be resolved.

        Returns:
            List of ``(base_name, key)`` pairs in declaration order, omitting
            any bases that could not be resolved (e.g. ``object``).
        """
        result: list[tuple[str, ClassKey]] = []
        for base_name in parsed.bases:
            resolved = self._resolve_base(base_name, parsed.module)
            if resolved is not None:
                result.append((base_name, resolved))
        return result

    def mro(self, parsed: ParsedClass) -> list[ClassKey]:
        """Compute a linearised MRO for *parsed* across registered classes.

        The result is ordered subclass-first (self, then parents depth-first).
        Cycles and unresolvable bases are silently skipped.

        Args:
            parsed: The class to walk.

        Returns:
            List of :data:`ClassKey` entries from *parsed* through its
            ancestor chain, subclass-first.
        """
        key: ClassKey = (parsed.module, parsed.name)
        visiting: set[ClassKey] = set()

        def walk(current: ClassKey) -> list[ClassKey]:
            if current in visiting or current not in self.classes:
                return [current] if current in self.classes else []
            visiting.add(current)
            chain = [current]
            for base_name in self.classes[current].bases:
                resolved = self._resolve_base(base_name, current[0])
                if resolved is None:
                    continue
                for item in walk(resolved):
                    if item not in chain:
                        chain.append(item)
            return chain

        return walk(key)

    def init_fields(
        self, parsed: ParsedClass, *, mro: list[ClassKey] | None = None
    ) -> list[InitField]:
        """Collect all ``__init__`` fields for *parsed* across the MRO.

        Fields are gathered base-first so parent fields precede child fields,
        while the closest MRO declaration determines whether each field
        participates in ``__init__``.

        Args:
            parsed: The class to collect fields for.
            mro: Pre-computed MRO to avoid a redundant walk.  When ``None``,
                :meth:`mro` is called internally.

        Returns:
            Ordered list of :class:`InitField` instances.
        """
        chain = mro if mro is not None else self.mro(parsed)
        field_owners: dict[str, ClassKey | None] = {}
        for key in chain:
            if key not in self.classes:
                continue
            current = self.classes[key]
            for name in current.non_init_field_names:
                field_owners.setdefault(name, None)
            for init_field in current.init_fields:
                field_owners.setdefault(init_field.name, key)

        fields: list[InitField] = []
        for key in reversed(chain):
            if key not in self.classes:
                continue
            for init_field in self.classes[key].init_fields:
                if field_owners[init_field.name] == key:
                    fields.append(init_field)
        return fields

    def merged_docstring(self, parsed: ParsedClass) -> str:
        """Build a merged docstring combining MRO ``Args:`` sections.

        Args:
            parsed: The class whose docstring should be assembled.

        Returns:
            Complete docstring text with the class's own summary and a merged
            ``Args:`` block drawn from the full inheritance chain.
        """
        chain = self.mro(parsed)
        mro_docs = [
            self.classes[key].docstring
            for key in reversed(chain)
            if key in self.classes
        ]
        names = [f.name for f in self.init_fields(parsed, mro=chain)]
        return merge.build_merged_docstring(parsed.docstring, mro_docs, names)

    def classes_in_module(self, module: str) -> list[ParsedClass]:
        """Return parsed classes for *module*, sorted by MRO depth then name.

        Args:
            module: Module stem (e.g. ``"density"``).

        Returns:
            List of :class:`ParsedClass` instances ordered so that base
            classes appear before their subclasses.
        """
        classes = [parsed for (mod, _), parsed in self.classes.items() if mod == module]
        return sorted(classes, key=lambda parsed: (len(self.mro(parsed)), parsed.name))

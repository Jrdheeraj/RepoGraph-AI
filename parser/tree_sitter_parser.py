from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FunctionInfo:
    name: str
    qualified_name: str
    file_path: Path
    line_start: int
    line_end: int
    docstring: str | None = None
    parameters: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    parent_class: str | None = None
    is_async: bool = False


@dataclass(frozen=True)
class ClassInfo:
    name: str
    qualified_name: str
    file_path: Path
    line_start: int
    line_end: int
    docstring: str | None = None
    bases: list[str] = field(default_factory=list)
    methods: list[FunctionInfo] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ImportInfo:
    module: str | None
    name: str | None
    alias: str | None
    level: int
    statement_type: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class FileMetadata:
    path: Path
    module_name: str
    docstring: str | None = None
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    has_syntax_error: bool = False


@lru_cache(maxsize=1)
def _load_tree_sitter_parser() -> Any:
    try:
        from tree_sitter_language_pack import get_parser

        return get_parser("python")
    except ImportError as exc:
        raise RuntimeError(
            "tree-sitter-language-pack is required to parse repository files."
        ) from exc
    except Exception:
        try:
            from tree_sitter import Parser
            from tree_sitter_language_pack import get_language

            parser = Parser()
            language = get_language("python")
            if hasattr(parser, "set_language"):
                parser.set_language(language)
            else:
                parser.language = language
            return parser
        except ImportError as exc:
            raise RuntimeError("tree-sitter is required to parse repository files.") from exc


def _parse_with_tree_sitter(source: str) -> bool:
    parser = _load_tree_sitter_parser()
    try:
        tree = parser.parse(source)
        return bool(getattr(tree.root_node, "has_error", False))
    except Exception as exc:
        logger.warning("Tree-sitter parse crashed", exc_info=True, extra={"error": str(exc)})
        return True


def _module_name(file_path: Path, repository_root: Path | None) -> str:
    path = file_path.resolve()
    if repository_root is not None:
        try:
            path = path.relative_to(repository_root.resolve())
        except ValueError:
            pass
    module_parts = list(path.with_suffix("").parts)
    if module_parts and module_parts[-1] == "__init__":
        module_parts.pop()
    return ".".join(module_parts)


def _extract_docstring(body: cst.CSTNode) -> str | None:
    body_items = getattr(body, "body", ())
    if not body_items:
        return None

    first = body_items[0]
    if not isinstance(first, cst.SimpleStatementLine) or not first.body:
        return None

    expression = first.body[0]
    if not isinstance(expression, cst.Expr) or not isinstance(expression.value, cst.SimpleString):
        return None

    try:
        return expression.value.evaluated_value
    except Exception:
        return None


def _name_for_attr(node: cst.BaseExpression) -> str:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return f"{_name_for_attr(node.value)}.{node.attr.value}"
    if isinstance(node, cst.Subscript):
        return _name_for_attr(node.value)
    return ""


def _param_names(params: cst.Parameters) -> list[str]:
    names: list[str] = []
    for collection in (params.posonly_params, params.params, params.kwonly_params):
        names.extend(param.name.value for param in collection)
    if params.star_arg and isinstance(params.star_arg, cst.Param):
        names.append(f"*{params.star_arg.name.value}")
    if params.star_kwarg:
        names.append(f"**{params.star_kwarg.name.value}")
    return names


class _MetadataVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, module: cst.Module, file_path: Path, module_name: str) -> None:
        self.module = module
        self.file_path = file_path
        self.module_name = module_name
        self.imports: list[ImportInfo] = []
        self.functions: list[FunctionInfo] = []
        self._classes: list[dict[str, Any]] = []
        self._class_stack: list[dict[str, Any]] = []
        self._function_stack: list[bool] = []

    @property
    def classes(self) -> list[ClassInfo]:
        return [
            ClassInfo(
                name=item["name"],
                qualified_name=item["qualified_name"],
                file_path=self.file_path,
                line_start=item["line_start"],
                line_end=item["line_end"],
                docstring=item["docstring"],
                bases=item["bases"],
                methods=item["methods"],
                decorators=item["decorators"],
            )
            for item in self._classes
        ]

    def _position(self, node: cst.CSTNode) -> tuple[int, int]:
        position = self.get_metadata(PositionProvider, node)
        return position.start.line, position.end.line

    def _code_for_node(self, node: cst.CSTNode) -> str:
        return self.module.code_for_node(node).strip()

    def visit_Import(self, node: cst.Import) -> None:
        line_start, line_end = self._position(node)
        for alias in node.names:
            if isinstance(alias, cst.ImportAlias):
                self.imports.append(
                    ImportInfo(
                        module=_name_for_attr(alias.name),
                        name=None,
                        alias=alias.asname.name.value if alias.asname else None,
                        level=0,
                        statement_type="import",
                        line_start=line_start,
                        line_end=line_end,
                    )
                )

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        line_start, line_end = self._position(node)
        module = _name_for_attr(node.module) if node.module else None
        level = len(node.relative) if node.relative else 0

        if isinstance(node.names, cst.ImportStar):
            self.imports.append(
                ImportInfo(module=module, name="*", alias=None, level=level, statement_type="from", line_start=line_start, line_end=line_end)
            )
            return

        for alias in node.names:
            self.imports.append(
                ImportInfo(
                    module=module,
                    name=_name_for_attr(alias.name),
                    alias=alias.asname.name.value if alias.asname else None,
                    level=level,
                    statement_type="from",
                    line_start=line_start,
                    line_end=line_end,
                )
            )

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        line_start, line_end = self._position(node)
        prefix = f"{self.module_name}."
        if self._class_stack:
            prefix = f"{self._class_stack[-1]['qualified_name']}."

        class_data: dict[str, Any] = {
            "name": node.name.value,
            "qualified_name": f"{prefix}{node.name.value}",
            "line_start": line_start,
            "line_end": line_end,
            "docstring": _extract_docstring(node.body),
            "bases": [self._code_for_node(arg.value) for arg in node.bases],
            "methods": [],
            "decorators": [self._code_for_node(decorator.decorator) for decorator in node.decorators],
        }
        self._class_stack.append(class_data)

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self._classes.append(self._class_stack.pop())

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if self._function_stack:
            self._function_stack.append(False)
            return False

        line_start, line_end = self._position(node)
        parent_class = self._class_stack[-1]["qualified_name"] if self._class_stack else None
        prefix = f"{parent_class}." if parent_class else f"{self.module_name}."
        function = FunctionInfo(
            name=node.name.value,
            qualified_name=f"{prefix}{node.name.value}",
            file_path=self.file_path,
            line_start=line_start,
            line_end=line_end,
            docstring=_extract_docstring(node.body),
            parameters=_param_names(node.params),
            decorators=[self._code_for_node(decorator.decorator) for decorator in node.decorators],
            parent_class=parent_class,
            is_async=node.asynchronous is not None,
        )

        if self._class_stack:
            self._class_stack[-1]["methods"].append(function)
        else:
            self.functions.append(function)

        self._function_stack.append(True)
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self._function_stack.pop()


def parse_file(file_path: str | Path, repository_root: str | Path | None = None) -> FileMetadata:
    path = Path(file_path).resolve()
    root = Path(repository_root).resolve() if repository_root is not None else None
    
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.warning("Failed to read file", extra={"file_path": str(path), "error": str(exc)})
        return FileMetadata(
            path=path,
            module_name=_module_name(path, root),
            has_syntax_error=True,
        )

    logger.debug("Parsing Python file", extra={"file_path": str(path)})
    
    try:
        has_syntax_error = _parse_with_tree_sitter(source)
    except Exception as exc:
        logger.warning("Tree-sitter crashed", extra={"file_path": str(path), "error": str(exc)})
        has_syntax_error = True

    try:
        module = cst.parse_module(source)
    except Exception as exc:
        logger.warning("LibCST failed to parse file", extra={"file_path": str(path), "error": str(exc)})
        return FileMetadata(
            path=path,
            module_name=_module_name(path, root),
            has_syntax_error=True,
        )

    module_name = _module_name(path, root)
    wrapper = MetadataWrapper(module)
    visitor = _MetadataVisitor(module=module, file_path=path, module_name=module_name)
    try:
        wrapper.visit(visitor)
    except Exception as exc:
        logger.warning("LibCST visitor crashed", extra={"file_path": str(path), "error": str(exc)})
        return FileMetadata(
            path=path,
            module_name=module_name,
            has_syntax_error=True,
        )

    metadata = FileMetadata(
        path=path,
        module_name=module_name,
        docstring=_extract_docstring(module),
        classes=visitor.classes,
        functions=visitor.functions,
        imports=visitor.imports,
        has_syntax_error=has_syntax_error,
    )
    logger.debug(
        "Parsed Python file metadata",
        extra={
            "file_path": str(path),
            "classes": len(metadata.classes),
            "functions": len(metadata.functions),
            "imports": len(metadata.imports),
            "has_syntax_error": metadata.has_syntax_error,
        },
    )
    return metadata


def parse_files(file_paths: list[Path], repository_root: str | Path | None = None) -> list[FileMetadata]:
    return [parse_file(file_path, repository_root=repository_root) for file_path in file_paths]


__all__ = [
    "ClassInfo",
    "FileMetadata",
    "FunctionInfo",
    "ImportInfo",
    "parse_file",
    "parse_files",
]

"""Static metrics collection stage for the RepoGraph AI dataset pipeline.

Analyzes cloned repositories under ``data/repositories`` and writes repository
level software engineering metrics to CSV.
"""

import ast
import io
import logging
import math
import os
import time
import tokenize
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

try:
    from scripts.dataset_pipeline.collection_state import CollectionStateManager, STATE_FILE_PATH
    from scripts.dataset_pipeline.utils import setup_pipeline_logging
except ImportError:
    from collection_state import CollectionStateManager, STATE_FILE_PATH
    from utils import setup_pipeline_logging

logger = setup_pipeline_logging(__name__)

REPOSITORIES_DIR = Path("data/repositories")
CANDIDATES_CSV_PATH = Path("data/datasets/repository_candidates.csv")
METRICS_CSV_PATH = Path("data/datasets/repositories_metrics.csv")
METRICS_FAILURES_CSV_PATH = Path("data/logs/metrics_failures.csv")
MAX_ANALYSIS_RETRIES = 2
DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 2) - 1, 4))

METRIC_COLUMNS = [
    "repository",
    "repository_path",
    "total_files",
    "total_directories",
    "total_python_files",
    "total_modules",
    "total_packages",
    "loc",
    "blank_lines",
    "comment_lines",
    "average_file_size",
    "largest_file_size",
    "function_count",
    "class_count",
    "method_count",
    "average_function_length",
    "avg_function_length",
    "maximum_function_length",
    "cyclomatic_complexity",
    "maintainability_index",
    "halstead_volume",
    "halstead_difficulty",
    "halstead_effort",
    "dependency_count",
    "external_imports",
    "internal_imports",
    "fan_in",
    "fan_out",
    "readme_exists",
    "docstring_coverage",
    "comment_density",
    "has_tests",
    "test_file_count",
    "has_examples",
    "has_ci",
    "has_dockerfile",
    "has_requirements",
    "has_pyproject",
    "module_count",
]

EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "site-packages",
    "venv",
}

PYTHON_KEYWORDS_FOR_HALSTEAD = {
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
}


@dataclass
class FileMetrics:
    """Metrics collected for a single Python source file."""

    path: Path
    module_name: str
    loc: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    size_bytes: int = 0
    function_count: int = 0
    class_count: int = 0
    method_count: int = 0
    function_lengths: List[int] = field(default_factory=list)
    cyclomatic_complexity: int = 0
    halstead_operators: Counter[str] = field(default_factory=Counter)
    halstead_operands: Counter[str] = field(default_factory=Counter)
    external_imports: Set[str] = field(default_factory=set)
    internal_imports: Set[str] = field(default_factory=set)
    docstring_nodes: int = 0
    documented_nodes: int = 0


class PythonFileAnalyzer(ast.NodeVisitor):
    """AST visitor that extracts function, class, complexity, and import metrics."""

    def __init__(self, internal_modules: Set[str]):
        """Initializes the analyzer with known internal module names."""
        self.internal_modules = internal_modules
        self.function_count = 0
        self.class_count = 0
        self.method_count = 0
        self.function_lengths: List[int] = []
        self.cyclomatic_complexity = 1
        self.external_imports: Set[str] = set()
        self.internal_imports: Set[str] = set()
        self.docstring_nodes = 0
        self.documented_nodes = 0
        self.class_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Visits a synchronous function definition."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        """Visits an asynchronous function definition."""
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        """Visits a class definition."""
        self.class_count += 1
        self._track_docstring(node)
        self.class_depth += 1
        self.generic_visit(node)
        self.class_depth -= 1

    def visit_Import(self, node: ast.Import) -> Any:
        """Visits import statements."""
        for alias in node.names:
            self._track_import(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        """Visits from-import statements."""
        if node.module:
            self._track_import(node.module)

    def visit_If(self, node: ast.If) -> Any:
        self.cyclomatic_complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        self.cyclomatic_complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        self.cyclomatic_complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> Any:
        self.cyclomatic_complexity += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> Any:
        self.cyclomatic_complexity += max(1, len(node.handlers))
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> Any:
        self.cyclomatic_complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        self.cyclomatic_complexity += max(len(node.values) - 1, 0)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        self.cyclomatic_complexity += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> Any:
        self.cyclomatic_complexity += 1 + len(node.ifs)
        self.generic_visit(node)

    def _visit_function(self, node: ast.AST) -> None:
        self.function_count += 1
        if self.class_depth > 0:
            self.method_count += 1
        self._track_docstring(node)
        start_line = getattr(node, "lineno", 0)
        end_line = getattr(node, "end_lineno", start_line)
        self.function_lengths.append(max(end_line - start_line + 1, 1))
        self.generic_visit(node)

    def _track_docstring(self, node: ast.AST) -> None:
        self.docstring_nodes += 1
        if ast.get_docstring(node):
            self.documented_nodes += 1

    def _track_import(self, module_name: str) -> None:
        top_level_module = module_name.split(".", 1)[0]
        if top_level_module in self.internal_modules:
            self.internal_imports.add(top_level_module)
        else:
            self.external_imports.add(top_level_module)


def should_skip_path(path: Path) -> bool:
    """Returns True when a path belongs to an ignored directory."""
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)


def repository_name_from_path(repo_path: Path) -> str:
    """Normalizes a local repository folder name into owner/repository format when possible."""
    return repo_path.name.replace("__", "/")


def iter_repository_files(repo_path: Path) -> Iterable[Path]:
    """Yields non-excluded files inside a repository."""
    for path in repo_path.rglob("*"):
        if should_skip_path(path.relative_to(repo_path)):
            continue
        if path.is_file():
            yield path


def discover_python_files(repo_path: Path) -> List[Path]:
    """Finds analyzable Python files inside a repository."""
    return [path for path in iter_repository_files(repo_path) if path.suffix == ".py"]


def module_name_for_file(repo_path: Path, file_path: Path) -> str:
    """Converts a Python file path to a dotted module name."""
    relative_path = file_path.relative_to(repo_path).with_suffix("")
    parts = [part for part in relative_path.parts if part != "__init__"]
    return ".".join(parts)


def get_internal_modules(repo_path: Path, python_files: List[Path]) -> Set[str]:
    """Builds a set of importable top-level internal module names."""
    modules: Set[str] = set()
    for file_path in python_files:
        relative = file_path.relative_to(repo_path)
        if len(relative.parts) == 1:
            modules.add(relative.stem)
        else:
            modules.add(relative.parts[0])
    return {module for module in modules if module and module != "__init__"}


def count_lines(source: str) -> Tuple[int, int, int]:
    """Counts code, blank, and comment lines using tokenize."""
    physical_lines = source.splitlines()
    blank_lines = sum(1 for line in physical_lines if not line.strip())
    comment_lines: Set[int] = set()

    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                comment_lines.add(token.start[0])
    except tokenize.TokenError:
        pass

    loc = max(len(physical_lines) - blank_lines - len(comment_lines), 0)
    return loc, blank_lines, len(comment_lines)


def collect_halstead_tokens(source: str) -> Tuple[Counter[str], Counter[str]]:
    """Collects approximate Halstead operators and operands from Python tokens."""
    operators: Counter[str] = Counter()
    operands: Counter[str] = Counter()

    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            token_type = token.type
            token_text = token.string
            if token_type == tokenize.OP or token_text in PYTHON_KEYWORDS_FOR_HALSTEAD:
                operators[token_text] += 1
            elif token_type in {tokenize.NAME, tokenize.NUMBER, tokenize.STRING}:
                if token_text not in PYTHON_KEYWORDS_FOR_HALSTEAD:
                    operands[token_text] += 1
    except tokenize.TokenError:
        pass

    return operators, operands


def analyze_python_file(repo_path: Path, file_path: Path, internal_modules: Set[str]) -> FileMetrics:
    """Analyzes a Python file using source lines, tokens, and AST."""
    source = file_path.read_text(encoding="utf-8", errors="ignore")
    loc, blank_lines, comment_lines = count_lines(source)
    operators, operands = collect_halstead_tokens(source)
    metrics = FileMetrics(
        path=file_path,
        module_name=module_name_for_file(repo_path, file_path),
        loc=loc,
        blank_lines=blank_lines,
        comment_lines=comment_lines,
        size_bytes=file_path.stat().st_size,
        halstead_operators=operators,
        halstead_operands=operands,
    )

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return metrics

    analyzer = PythonFileAnalyzer(internal_modules)
    analyzer._track_docstring(tree)
    analyzer.visit(tree)
    metrics.function_count = analyzer.function_count
    metrics.class_count = analyzer.class_count
    metrics.method_count = analyzer.method_count
    metrics.function_lengths = analyzer.function_lengths
    metrics.cyclomatic_complexity = analyzer.cyclomatic_complexity
    metrics.external_imports = analyzer.external_imports
    metrics.internal_imports = analyzer.internal_imports
    metrics.docstring_nodes = analyzer.docstring_nodes
    metrics.documented_nodes = analyzer.documented_nodes
    return metrics


def calculate_halstead_metrics(operators: Counter[str], operands: Counter[str]) -> Tuple[float, float, float]:
    """Calculates aggregate Halstead volume, difficulty, and effort."""
    distinct_operators = len(operators)
    distinct_operands = len(operands)
    total_operators = sum(operators.values())
    total_operands = sum(operands.values())
    vocabulary = distinct_operators + distinct_operands
    length = total_operators + total_operands

    if vocabulary <= 1 or length == 0:
        return 0.0, 0.0, 0.0

    volume = length * math.log2(vocabulary)
    difficulty = (distinct_operators / 2) * (total_operands / max(distinct_operands, 1))
    effort = volume * difficulty
    return round(volume, 2), round(difficulty, 2), round(effort, 2)


def calculate_maintainability_index(
    halstead_volume: float,
    cyclomatic_complexity: int,
    loc: int,
    comment_density: float,
) -> float:
    """Calculates an approximate maintainability index normalized to 0-100."""
    if loc <= 0:
        return 100.0

    raw_index = (
        171
        - 5.2 * math.log(max(halstead_volume, 1))
        - 0.23 * cyclomatic_complexity
        - 16.2 * math.log(loc)
        + 50 * math.sin(math.sqrt(2.4 * max(comment_density, 0)))
    )
    return round(max(0.0, min(raw_index * 100 / 171, 100.0)), 2)


def has_any_file(repo_path: Path, names: Set[str]) -> bool:
    """Checks whether a repository contains any file with one of the provided lowercase names."""
    return any(path.name.lower() in names for path in iter_repository_files(repo_path))


def count_test_files(python_files: List[Path]) -> int:
    """Counts Python test files using common test naming conventions."""
    count = 0
    for file_path in python_files:
        lower_parts = {part.lower() for part in file_path.parts}
        lower_name = file_path.name.lower()
        if "tests" in lower_parts or lower_name.startswith("test_") or lower_name.endswith("_test.py"):
            count += 1
    return count


def build_internal_graph(file_metrics: List[FileMetrics]) -> Tuple[int, int]:
    """Builds a repository-level internal import graph and returns fan-in/fan-out edge counts."""
    edges: Set[Tuple[str, str]] = set()
    incoming: Dict[str, Set[str]] = defaultdict(set)

    for metrics in file_metrics:
        source_module = metrics.module_name.split(".", 1)[0]
        for imported_module in metrics.internal_imports:
            if imported_module and imported_module != source_module:
                edge = (source_module, imported_module)
                edges.add(edge)
                incoming[imported_module].add(source_module)

    fan_in = sum(len(sources) for sources in incoming.values())
    fan_out = len(edges)
    return fan_in, fan_out


def analyze_repository(repo_path: Path) -> Dict[str, Any]:
    """Extracts static software engineering metrics for one cloned repository."""
    all_files = list(iter_repository_files(repo_path))
    all_directories = [
        path
        for path in repo_path.rglob("*")
        if path.is_dir() and not should_skip_path(path.relative_to(repo_path))
    ]
    python_files = [path for path in all_files if path.suffix == ".py"]
    internal_modules = get_internal_modules(repo_path, python_files)
    file_metrics = [analyze_python_file(repo_path, path, internal_modules) for path in python_files]

    total_size = sum(path.stat().st_size for path in all_files)
    largest_file_size = max((path.stat().st_size for path in all_files), default=0)
    function_lengths = [length for metrics in file_metrics for length in metrics.function_lengths]
    total_loc = sum(metrics.loc for metrics in file_metrics)
    blank_lines = sum(metrics.blank_lines for metrics in file_metrics)
    comment_lines = sum(metrics.comment_lines for metrics in file_metrics)
    total_lines = total_loc + blank_lines + comment_lines
    operators: Counter[str] = Counter()
    operands: Counter[str] = Counter()
    external_imports: Set[str] = set()
    internal_imports: Set[str] = set()

    for metrics in file_metrics:
        operators.update(metrics.halstead_operators)
        operands.update(metrics.halstead_operands)
        external_imports.update(metrics.external_imports)
        internal_imports.update(metrics.internal_imports)

    halstead_volume, halstead_difficulty, halstead_effort = calculate_halstead_metrics(operators, operands)
    cyclomatic_complexity = sum(metrics.cyclomatic_complexity for metrics in file_metrics)
    comment_density = round(comment_lines / max(total_lines, 1), 4)
    maintainability_index = calculate_maintainability_index(
        halstead_volume,
        cyclomatic_complexity,
        total_loc,
        comment_density,
    )
    fan_in, fan_out = build_internal_graph(file_metrics)
    docstring_nodes = sum(metrics.docstring_nodes for metrics in file_metrics)
    documented_nodes = sum(metrics.documented_nodes for metrics in file_metrics)
    test_file_count = count_test_files(python_files)
    total_packages = sum(1 for path in all_directories if (path / "__init__.py").exists())

    return {
        "repository": repository_name_from_path(repo_path),
        "repository_path": str(repo_path),
        "total_files": len(all_files),
        "total_directories": len(all_directories),
        "total_python_files": len(python_files),
        "total_modules": len(file_metrics),
        "total_packages": total_packages,
        "loc": total_loc,
        "blank_lines": blank_lines,
        "comment_lines": comment_lines,
        "average_file_size": round(total_size / max(len(all_files), 1), 2),
        "largest_file_size": largest_file_size,
        "function_count": sum(metrics.function_count for metrics in file_metrics),
        "class_count": sum(metrics.class_count for metrics in file_metrics),
        "method_count": sum(metrics.method_count for metrics in file_metrics),
        "average_function_length": round(sum(function_lengths) / max(len(function_lengths), 1), 2),
        "avg_function_length": round(sum(function_lengths) / max(len(function_lengths), 1), 2),
        "maximum_function_length": max(function_lengths, default=0),
        "cyclomatic_complexity": cyclomatic_complexity,
        "maintainability_index": maintainability_index,
        "halstead_volume": halstead_volume,
        "halstead_difficulty": halstead_difficulty,
        "halstead_effort": halstead_effort,
        "dependency_count": len(external_imports),
        "external_imports": ",".join(sorted(external_imports)),
        "internal_imports": ",".join(sorted(internal_imports)),
        "fan_in": fan_in,
        "fan_out": fan_out,
        "readme_exists": has_any_file(repo_path, {"readme", "readme.md", "readme.rst", "readme.txt"}),
        "docstring_coverage": round(documented_nodes / max(docstring_nodes, 1), 4),
        "comment_density": comment_density,
        "has_tests": test_file_count > 0,
        "test_file_count": test_file_count,
        "has_examples": any(path.name.lower() in {"examples", "example"} for path in all_directories),
        "has_ci": any(
            path.name.lower() in {".github", ".gitlab-ci.yml", "circle.yml", ".travis.yml", "azure-pipelines.yml"}
            for path in list(all_directories) + all_files
        ),
        "has_dockerfile": has_any_file(repo_path, {"dockerfile", "docker-compose.yml", "docker-compose.yaml"}),
        "has_requirements": has_any_file(repo_path, {"requirements.txt", "requirements-dev.txt", "requirements.in"}),
        "has_pyproject": has_any_file(repo_path, {"pyproject.toml"}),
        "module_count": len(file_metrics),
    }


def analyze_repository_with_retries(repo_path: Path, max_retries: int = MAX_ANALYSIS_RETRIES) -> Dict[str, Any]:
    """Analyzes one repository with retries for transient filesystem errors."""
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return analyze_repository(repo_path)
        except (OSError, UnicodeError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(0.5 * attempt)
    raise RuntimeError(str(last_error) if last_error else "Unknown metrics analysis failure")


class MetricsCollector:
    """Coordinates static metrics collection for cloned repositories."""

    def __init__(
        self,
        repositories_root: Path = REPOSITORIES_DIR,
        output_csv_path: Path = METRICS_CSV_PATH,
        failures_csv_path: Path = METRICS_FAILURES_CSV_PATH,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ):
        """Initializes repository, output, and failure paths."""
        self.repositories_root = Path(repositories_root)
        self.output_csv_path = Path(output_csv_path)
        self.failures_csv_path = Path(failures_csv_path)
        self.max_workers = max_workers
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.failures_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_manager = self.initialize_state_manager()

    def initialize_state_manager(self) -> CollectionStateManager:
        """Loads collection state and initializes it from candidates on first run."""
        state_exists = STATE_FILE_PATH.exists()
        manager = CollectionStateManager()
        if not state_exists and CANDIDATES_CSV_PATH.exists():
            manager.initialize_state(CANDIDATES_CSV_PATH)
            logger.info("Initialized collection state from %s", CANDIDATES_CSV_PATH)
        logger.info("Loaded collection state: %s", manager.export_summary())
        return manager

    def discover_repositories(self) -> List[Path]:
        """Discovers cloned repository directories under the configured root."""
        if not self.repositories_root.exists():
            logger.warning("Repository root does not exist: %s", self.repositories_root)
            return []
        repositories = sorted(path for path in self.repositories_root.iterdir() if path.is_dir())
        logger.info("Discovered %s cloned repositories in %s", len(repositories), self.repositories_root)
        return repositories

    def load_existing_metrics(self) -> Tuple[List[Dict[str, Any]], Set[str]]:
        """Loads existing metrics output to support resumable execution."""
        if not self.output_csv_path.exists():
            return [], set()

        df = pd.read_csv(self.output_csv_path).fillna("")
        records = df.to_dict(orient="records")
        processed = {str(record.get("repository", "")).lower() for record in records if record.get("repository")}
        logger.info("Loaded %s existing metrics records from %s", len(records), self.output_csv_path)
        return records, processed

    def save_metrics(self, records: List[Dict[str, Any]]) -> None:
        """Writes repository metrics to CSV."""
        pd.DataFrame(records, columns=METRIC_COLUMNS).to_csv(
            self.output_csv_path,
            index=False,
            encoding="utf-8",
        )

    def save_failures(self, failures: List[Dict[str, str]]) -> None:
        """Writes repository analysis failures to CSV."""
        failure_df = pd.DataFrame(failures, columns=["repository", "path", "reason"])
        failure_df.to_csv(self.failures_csv_path, index=False, encoding="utf-8")
        logger.info("Wrote %s metrics failures to %s", len(failures), self.failures_csv_path)

    def run(self) -> Dict[str, int]:
        """Runs static analysis for all cloned repositories."""
        repositories = self.discover_repositories()
        records, processed = self.load_existing_metrics()
        completed = len(self.state_manager.get_completed_repositories("metrics_extracted"))
        logger.info("Resume position for metrics: %s repositories already marked extracted.", completed)
        pending_repositories: List[Path] = []
        already_processed = 0
        for path in repositories:
            repository = repository_name_from_path(path)
            state_record = self.state_manager.get_repository(repository)
            if state_record and state_record.get("metrics_extracted", False):
                logger.info("Skipping completed metrics extraction from state: %s", repository)
                already_processed += 1
                continue

            if repository.lower() in processed:
                self.state_manager.update_step(repository, "metrics_extracted", True)
                logger.info("Updated collection state from existing metrics CSV for %s", repository)
                already_processed += 1
                continue

            pending_repositories.append(path)
        failures: List[Dict[str, str]] = []
        successful = 0

        if pending_repositories:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_repo = {
                    executor.submit(analyze_repository_with_retries, repo_path): repo_path
                    for repo_path in pending_repositories
                }
                for future in tqdm(as_completed(future_to_repo), total=len(future_to_repo), desc="Analyzing repositories"):
                    repo_path = future_to_repo[future]
                    repository = repository_name_from_path(repo_path)
                    try:
                        metrics = future.result()
                        records.append(metrics)
                        processed.add(repository.lower())
                        successful += 1
                        self.save_metrics(records)
                        self.state_manager.update_step(repository, "metrics_extracted", True)
                        logger.info("Updated collection state for metrics extraction: %s", repository)
                    except Exception as exc:
                        logger.error("Failed to analyze %s: %s", repository, exc)
                        self.state_manager.mark_failed(repository, str(exc))
                        logger.warning("Failed repository state updated for %s", repository)
                        failures.append({
                            "repository": repository,
                            "path": str(repo_path),
                            "reason": str(exc),
                        })

        self.save_metrics(records)
        self.save_failures(failures)

        summary = {
            "repositories_discovered": len(repositories),
            "already_processed": already_processed,
            "successful": successful,
            "failed": len(failures),
            "total_records": len(records),
        }
        logger.info("Metrics collection summary:")
        logger.info("Repositories discovered: %s", summary["repositories_discovered"])
        logger.info("Already processed: %s", summary["already_processed"])
        logger.info("Successful: %s", summary["successful"])
        logger.info("Failed: %s", summary["failed"])
        logger.info("Total metric records: %s", summary["total_records"])

        print("Metrics collection summary")
        print(f"Repositories discovered: {summary['repositories_discovered']}")
        print(f"Already processed: {summary['already_processed']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Total metric records: {summary['total_records']}")
        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = MetricsCollector()
    collector.run()

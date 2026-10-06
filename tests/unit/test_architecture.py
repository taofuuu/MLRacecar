"""Architecture rule that import-linter cannot express: the core's import allow-list (ADR-0003).

import-linter (configured in pyproject.toml) checks that layers only import downward. This test
checks the stricter rule for the bottom layer: `mlracecar.core` may import only the standard
library, NumPy, and other core modules. That keeps the core fast to test, deterministic, and
portable to the browser. import-linter's contracts are deny-lists, so an allow-list is enforced
here instead.
"""

import ast
import sys
from pathlib import Path

import pytest

import mlracecar.core

CORE_DIR = Path(mlracecar.core.__file__).parent
ALLOWED_THIRD_PARTY = frozenset({"numpy"})


def is_allowed_in_core(module: str) -> bool:
    top_level = module.partition(".")[0]
    return (
        top_level in sys.stdlib_module_names
        or top_level in ALLOWED_THIRD_PARTY
        or module == "mlracecar.core"
        or module.startswith("mlracecar.core.")
    )


def absolute_imports(path: Path) -> list[tuple[int, str]]:
    """Return (line, module) for every absolute import in a source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
    return found


@pytest.mark.parametrize(
    ("module", "allowed"),
    [
        ("math", True),
        ("collections.abc", True),
        ("__future__", True),
        ("numpy", True),
        ("numpy.typing", True),
        ("mlracecar.core.track", True),
        ("mlracecar", False),
        ("mlracecar.render", False),
        ("mlracecar.config", False),
        ("pygame", False),
        ("torch", False),
        ("pydantic", False),
    ],
)
def test_allow_list(module: str, allowed: bool) -> None:
    assert is_allowed_in_core(module) is allowed


def test_core_imports_only_stdlib_numpy_and_core() -> None:
    violations = [
        f"{path.relative_to(CORE_DIR.parent)}:{line} imports {module}"
        for path in sorted(CORE_DIR.rglob("*.py"))
        for line, module in absolute_imports(path)
        if not is_allowed_in_core(module)
    ]
    assert not violations, (
        "mlracecar.core may import only the standard library, NumPy, and mlracecar.core "
        "(ADR-0003):\n  " + "\n  ".join(violations)
    )

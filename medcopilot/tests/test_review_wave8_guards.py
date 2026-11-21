"""Wave 8 guard tests: source-level invariants + ops-script bootstrap + -O discipline.

本波次把一批"只能靠纪律维持"的约束变成可执行守卫：
A. 源码 AST 守卫（扫描 medcopilot 产品源码）：
   1. 禁止导入 langchain / langchain_core（静态 import 与动态
      importlib.import_module / __import__ 的常量字符串首参一并禁止；
      my_langchain_utils 等形近名不误伤）。
   2. 产品源码禁止 ``assert``：python -O 会剥离 assert，任何运行期校验必须
      显式 ``raise``。
   3. 禁止跨顶层包相对导入：level>0 且向上走出所在顶层包即违规；包内
      ``__init__.py`` 的 re-export 按契约放行。
   4. medcopilot 永不导入其他项目包（保持独立仓库自包含，静态与动态导入一并禁止）。
B. 运维脚本 sys.path 自举静态守卫：所有导入 medcopilot 的运维脚本必须存在
   ``sys.path.insert(0, str(Path(__file__).resolve().parents[2]))`` 且其行号早于
   第一个 medcopilot 导入行；其余自足脚本保持模块级零项目包导入，并由目录全集测试钉死。
C. python -O 冒烟守护：assert 被剥离后运行期安全护栏（急救分诊、剂量门构造校验）
   必须仍然显式 raise。
"""

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOP_PACKAGES = ("medcopilot",)
PRODUCT_SOURCE_FLOOR = 50


# ------------------------------------------------------- 扫描面


def _product_sources() -> list[Path]:
    """产品源码全集：medcopilot *.py（排除 tests/ 与 __pycache__）。"""
    return [
        path
        for project in TOP_PACKAGES
        for path in (REPO_ROOT / project).rglob("*.py")
        if "tests" not in path.parts and "__pycache__" not in path.parts
    ]


PRODUCT_SOURCES = _product_sources()


def test_product_source_scan_surface_is_healthy():
    """扫描面意外缩小（目录改名/移动）必须让守卫显式失败而不是静默放行。"""
    assert len(PRODUCT_SOURCES) >= PRODUCT_SOURCE_FLOOR


# ------------------------------------------------------- 判定函数（供真实扫描与合成用例共用）


def _is_langchain_root(module: str) -> bool:
    """根名精确判定：langchain.* 与 langchain_core.* 命中；my_langchain_utils 不命中。"""
    return module.split(".")[0] in {"langchain", "langchain_core"}


def _is_forbidden_external_root(module: str) -> bool:
    """根名精确判定：禁止导入任何其他 project* 顶层兄弟项目包。"""
    root = module.split(".")[0]
    return root.startswith("project") and root not in TOP_PACKAGES and not root.startswith("medcopilot_")


def _langchain_offenders(path: Path, source: str) -> list[str]:
    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_langchain_root(alias.name):
                    found.append(f"{path.as_posix()}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and _is_langchain_root(node.module):
                found.append(f"{path.as_posix()}:{node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Call):
            func = node.func
            dynamic = (isinstance(func, ast.Attribute) and func.attr == "import_module") or (
                isinstance(func, ast.Name) and func.id == "__import__"
            )
            if (
                dynamic
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and _is_langchain_root(node.args[0].value)
            ):
                found.append(f"{path.as_posix()}:{node.lineno}: dynamic import {node.args[0].value!r}")
    return found


def _assert_offenders(path: Path, source: str) -> list[str]:
    tree = ast.parse(source)
    return [
        f"{path.as_posix()}:{node.lineno}: assert statement"
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
    ]


def _cross_package_relative_import_offenders(path: Path, source: str) -> list[str]:
    """level>0 的相对导入向上走 level-1 层，跳出所在顶层包即违规；__init__.py re-export 放行。"""
    parts = path.relative_to(REPO_ROOT).parts
    pkg_dir = REPO_ROOT / parts[0]
    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        if path.name == "__init__.py":
            continue
        base = path.parent
        for _ in range(node.level - 1):
            base = base.parent
        if base != pkg_dir and pkg_dir not in base.parents:
            found.append(f"{path.as_posix()}:{node.lineno}: relative import level={node.level} escapes {parts[0]}/")
    return found


def _external_project_import_offenders(path: Path, source: str) -> list[str]:
    if path.relative_to(REPO_ROOT).parts[0] != "medcopilot":
        return []
    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_external_root(alias.name):
                    found.append(f"{path.as_posix()}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if _is_forbidden_external_root(node.module):
                found.append(f"{path.as_posix()}:{node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Call):
            func = node.func
            dynamic = (isinstance(func, ast.Attribute) and func.attr == "import_module") or (
                isinstance(func, ast.Name) and func.id == "__import__"
            )
            if (
                dynamic
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and _is_forbidden_external_root(node.args[0].value)
            ):
                found.append(f"{path.as_posix()}:{node.lineno}: dynamic import {node.args[0].value!r}")
    return found


# ------------------------------------------------------- A: 真实扫描


def test_no_langchain_imports_in_source():
    offenders = []
    for path in PRODUCT_SOURCES:
        offenders.extend(_langchain_offenders(path, path.read_text(encoding="utf-8")))
    assert not offenders, f"langchain/langchain_core imports are banned: {offenders}"


def test_no_assert_in_product_source():
    """python -O 剥离 assert：运行期校验必须显式 raise。"""
    offenders = []
    for path in PRODUCT_SOURCES:
        offenders.extend(_assert_offenders(path, path.read_text(encoding="utf-8")))
    assert not offenders, f"assert must not guard runtime behavior under python -O: {offenders}"


def test_no_cross_package_relative_imports():
    offenders = []
    for path in PRODUCT_SOURCES:
        offenders.extend(_cross_package_relative_import_offenders(path, path.read_text(encoding="utf-8")))
    assert not offenders, f"relative imports must not escape their top-level package: {offenders}"


def test_medcopilot_stays_self_contained_without_external_project_imports():
    """扫描全量 medcopilot/*.py（含 tests/），确保无任何静态或动态跨项目导入。"""
    offenders = []
    for path in (REPO_ROOT / "medcopilot").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        offenders.extend(_external_project_import_offenders(path, path.read_text(encoding="utf-8")))
    assert not offenders, f"medcopilot must be self-contained: {offenders}"


# ------------------------------------------------------- A5: 判定函数自身回归钉子


class TestLangchainJudgement:
    def test_static_imports_flagged(self):
        code = "import langchain\nimport langchain.agents\nfrom langchain_core.messages import BaseMessage\n"
        found = _langchain_offenders(Path("synthetic.py"), code)
        assert len(found) == 3
        assert all("langchain" in item for item in found)

    def test_dynamic_constant_imports_flagged(self):
        code = (
            "import importlib\n"
            'mod = importlib.import_module("langchain.embeddings")\n'
            'legacy = __import__("langchain_core")\n'
        )
        found = _langchain_offenders(Path("synthetic.py"), code)
        assert len(found) == 2

    def test_lookalike_names_and_strings_not_flagged(self):
        code = (
            "import my_langchain_utils\n"
            "from langchain_compat import shim\n"
            'note = "langchain is banned"\n'
        )
        assert _langchain_offenders(Path("synthetic.py"), code) == []


class TestAssertJudgement:
    def test_assert_flagged_with_lineno(self):
        code = "def check(value):\n    assert value > 0\n    return value\n"
        found = _assert_offenders(Path("synthetic.py"), code)
        assert len(found) == 1
        assert ":2:" in found[0]

    def test_explicit_raise_allowed(self):
        code = "def check(value):\n    if value <= 0:\n        raise ValueError('bad')\n"
        assert _assert_offenders(Path("synthetic.py"), code) == []


class TestRelativeImportJudgement:
    MOD = REPO_ROOT / "medcopilot" / "safety" / "synthetic_mod.py"
    ROOT_MOD = REPO_ROOT / "medcopilot" / "synthetic_root.py"

    def test_escaping_relative_import_flagged(self):
        found = _cross_package_relative_import_offenders(self.MOD, "from ...common.config import config\n")
        assert len(found) == 1
        found_root = _cross_package_relative_import_offenders(self.ROOT_MOD, "from ..other import x\n")
        assert len(found_root) == 1

    def test_intra_package_relative_import_allowed(self):
        code = "from . import helper\nfrom .policy import load_policy\nfrom ..common.config import config\n"
        assert _cross_package_relative_import_offenders(self.MOD, code) == []
        assert _cross_package_relative_import_offenders(self.ROOT_MOD, "from .common.config import config\n") == []

    def test_init_py_reexport_allowed(self):
        init = REPO_ROOT / "medcopilot" / "workflow" / "__init__.py"
        code = "from .agent_graph import AgentGraph\n"
        assert _cross_package_relative_import_offenders(init, code) == []


class TestCrossPackageImportJudgement:
    P2_MOD = REPO_ROOT / "medcopilot" / "common" / "synthetic_mod.py"
    EXTERNAL = "project_sibling"

    def test_direct_imports_flagged(self):
        code = f"import {self.EXTERNAL}.common\nfrom {self.EXTERNAL}.common.config import config\n"
        found = _external_project_import_offenders(self.P2_MOD, code)
        assert len(found) == 2

    def test_dynamic_constant_imports_flagged(self):
        code = (
            "import importlib\n"
            f'mod = importlib.import_module("{self.EXTERNAL}.common.config")\n'
            f'legacy = __import__("{self.EXTERNAL}")\n'
        )
        found = _external_project_import_offenders(self.P2_MOD, code)
        assert len(found) == 2

    def test_comment_and_string_mentions_not_flagged(self):
        code = (
            f"# historical note: decoupled from {self.EXTERNAL}\n"
            f'NOTE = "{self.EXTERNAL} is not imported"\n'
            "import medcopilot_compat\n"
            "import medcopilot.common.config\n"
        )
        assert _external_project_import_offenders(self.P2_MOD, code) == []


# ------------------------------------------------------- B: 运维脚本自举静态守卫

BOOTSTRAP_SCRIPTS = (
    "medcopilot/scripts/backup_snapshot.py",
    "medcopilot/scripts/build_embedding_eval.py",
    "medcopilot/scripts/emergency_eval.py",
    "medcopilot/scripts/eval_embedding.py",
    "medcopilot/scripts/eval_golden.py",
    "medcopilot/scripts/eval_rag.py",
    "medcopilot/scripts/llm_matrix_probe.py",
    "medcopilot/scripts/llm_probe.py",
    "medcopilot/scripts/ocr_probe.py",
    "medcopilot/scripts/policy_check.py",
    "medcopilot/scripts/policy_sign.py",
    "medcopilot/scripts/prepare_weights.py",
    "medcopilot/scripts/recovery_drill.py",
    "medcopilot/scripts/release_pack.py",
)

SELF_CONTAINED_SCRIPTS = (
    "medcopilot/scripts/export_lock.py",
    "medcopilot/scripts/load_check.py",
    "medcopilot/scripts/redact_scan.py",
)


def _references_repo_root(node: ast.AST) -> bool:
    """实参链须含 str(Path(__file__).resolve().parents[2]) 的要素。"""
    walked = list(ast.walk(node))
    has_parents_two = any(
        isinstance(item, ast.Subscript)
        and isinstance(item.value, ast.Attribute)
        and item.value.attr == "parents"
        and isinstance(item.slice, ast.Constant)
        and item.slice.value == 2
        for item in walked
    )
    has_file = any(isinstance(item, ast.Name) and item.id == "__file__" for item in walked)
    has_resolve = any(
        isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr == "resolve"
        for item in walked
    )
    has_path = any(
        isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == "Path"
        for item in walked
    )
    return has_parents_two and has_file and has_resolve and has_path


def _repo_root_bootstrap_lines(tree: ast.Module) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "insert"):
            continue
        receiver = func.value
        if not (isinstance(receiver, ast.Attribute) and receiver.attr == "path"):
            continue
        if not (isinstance(receiver.value, ast.Name) and receiver.value.id == "sys"):
            continue
        if len(node.args) != 2 or not (isinstance(node.args[0], ast.Constant) and node.args[0].value == 0):
            continue
        if not _references_repo_root(node.args[1]):
            continue
        lines.append(node.lineno)
    return lines


def _first_project_import_line(tree: ast.Module) -> int | None:
    """第一个 medcopilot 导入的行号（含函数内惰性导入）。"""
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            lines.extend(
                node.lineno for alias in node.names if alias.name.split(".")[0] in TOP_PACKAGES
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.split(".")[0] in TOP_PACKAGES:
                lines.append(node.lineno)
    return min(lines) if lines else None


@pytest.mark.parametrize("script", BOOTSTRAP_SCRIPTS)
def test_ops_script_bootstraps_repo_root_before_project_imports(script):
    path = REPO_ROOT / script
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bootstrap = _repo_root_bootstrap_lines(tree)
    assert bootstrap, f"{script} lacks sys.path.insert(0, str(Path(__file__).resolve().parents[2]))"
    first_project = _first_project_import_line(tree)
    assert first_project is not None, f"{script} no longer imports project packages?"
    assert min(bootstrap) < first_project, (
        f"{script}: bootstrap at line {min(bootstrap)} must precede first project import at line {first_project}"
    )


@pytest.mark.parametrize("script", SELF_CONTAINED_SCRIPTS)
def test_self_contained_scripts_have_no_project_imports(script):
    """自足脚本（给子进程自设 PYTHONPATH 或纯标准库/CLI）不得出现项目包导入。"""
    path = REPO_ROOT / script
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert _first_project_import_line(tree) is None, (
        f"{script} now imports {TOP_PACKAGES}; move it to BOOTSTRAP_SCRIPTS and add sys.path bootstrap"
    )


def test_all_scripts_in_directory_are_classified_and_guarded():
    """scripts/ 下每个 .py 文件都必须归入 BOOTSTRAP_SCRIPTS 或 SELF_CONTAINED_SCRIPTS。"""
    scripts_dir = REPO_ROOT / "medcopilot" / "scripts"
    actual = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in scripts_dir.glob("*.py")
        if path.name != "__init__.py"
    }
    declared = set(BOOTSTRAP_SCRIPTS) | set(SELF_CONTAINED_SCRIPTS)
    assert actual == declared, f"scripts classification drift: diff={actual ^ declared}"


# ------------------------------------------------------- C: python -O 冒烟守护


def _run_python_o(code: str) -> subprocess.CompletedProcess:
    """python -O 子进程驱动器：剥 PYTHONOPTIMIZE 保证 -O 语义由 argv 独占决定。"""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONOPTIMIZE"}
    env["PYTHONPATH"] = "."
    env["MOCK_LLM"] = "1"
    return subprocess.run(
        [sys.executable, "-O", "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )


class TestPythonOptimizeSafetyDiscipline:
    def test_emergency_source_guard_survives_assert_stripping(self):
        result = _run_python_o(
            """
            from medcopilot.workflow.emergency_detector import EmergencyDetector

            try:
                EmergencyDetector.evaluate("胸痛", source="ocr")
            except ValueError:
                print("EMERGENCY_O_GUARD_RAISED")
            else:
                print("EMERGENCY_O_NO_GUARD")
            """
        )
        assert "EMERGENCY_O_GUARD_RAISED" in result.stdout, result.stdout + result.stderr
        assert "EMERGENCY_O_NO_GUARD" not in result.stdout

    def test_dosage_gate_mode_guard_survives_assert_stripping(self):
        result = _run_python_o(
            """
            from medcopilot.workflow.agent_graph import AgentGraph

            try:
                AgentGraph(dosage_gate_mode="invalid_mode")
            except ValueError:
                print("DOSAGE_O_GUARD_RAISED")
            else:
                print("DOSAGE_O_NO_GUARD")
            """
        )
        assert "DOSAGE_O_GUARD_RAISED" in result.stdout, result.stdout + result.stderr
        assert "DOSAGE_O_NO_GUARD" not in result.stdout

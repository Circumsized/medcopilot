"""W5.5/W5.7 deploy contract tests: static + behavioral.

Pins the contract of the deployment artifacts so a silent weakening (sourcing
the secret file, a shell-executable value, interpreter hooks leaking in,
multi-worker units, hardcoded secrets, jitter in the backup calendar) fails CI:

* static: deploy files exist and contain the required directives;
* static: the container launcher imports nothing app-side before secrets are
  parsed, and its allowlist is exactly the AppConfig aliases;
* behavioral: a hostile dotenv secret file cannot run shell expansions or
  inject variables — only allowlisted AppConfig keys reach the environment,
  PROJECT_ENV must classify as production, and diagnostics never print secret
  values;
* behavioral: the shell wrapper execs the Python launcher untouched.
"""

from __future__ import annotations

import configparser
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from medcopilot.deploy import entrypoint as entrypoint_module

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = REPO_ROOT / "medcopilot" / "deploy"
DOCS_MANUAL = REPO_ROOT / "medcopilot" / "docs" / "PRODUCTION_DEPLOY.md"
DOCKERFILE = REPO_ROOT / "Dockerfile"
SHELL_WRAPPER = DEPLOY_DIR / "docker-entrypoint.sh"
API_SERVICE = DEPLOY_DIR / "med-rag.service"
BACKUP_SERVICE = DEPLOY_DIR / "med-rag-backup.service"
BACKUP_TIMER = DEPLOY_DIR / "med-rag-backup.timer"
WRAPPER_PATH_IN_IMAGE = "/app/medcopilot/deploy/docker-entrypoint.sh"

SYNTHETIC_SECRET = "synthetic-not-a-real-key-0001"


def _read_unit(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str  # keep systemd directive casing
    parser.read(path, encoding="utf-8")
    return parser


def _pre_lines(path: Path) -> list[str]:
    """systemd allows repeated directives; configparser keeps only the last."""
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("ExecStartPre=")]


def _exec_lines(path: Path) -> list[str]:
    parser = _read_unit(path)
    values: list[str] = []
    for section in ("Service",):
        if not parser.has_section(section):
            continue
        for key, value in parser.items(section):
            if key in {"ExecStart", "ExecStartPre", "ExecStartPost", "ExecStop"}:
                values.append(value)
    return values


def _hostile_secret_file(tmp_path: Path, sentinel: Path) -> Path:
    """A dotenv file whose values try to behave like shell code."""
    path = tmp_path / "med_env"
    path.write_text(
        "\n".join(
            [
                "PROJECT_ENV=production",
                f"ADMIN_API_KEY={SYNTHETIC_SECRET}",
                "LOG_FILE=/var/log/${HOME}-subst",
                f"AUDIT_NTP_SERVER=$(touch {sentinel})",
                "LLM_API_BASE=$HOME-plain",
                f"LLM_MODEL=`touch {sentinel}`",
                "PYTHONPATH=/tmp/evil-pythonpath",
                "LD_PRELOAD=/tmp/evil-preload.so",
                "MED_RAG_SECRET_FILE=/tmp/evil-secret-path",
                "SHLVL=99",
                "# comment lines and blanks are data-only",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


# -------------------------------------------------------------- static: files


class TestDeployArtifactsExist:
    def test_owned_files_exist(self):
        for path in (
            DEPLOY_DIR / "__init__.py",
            DEPLOY_DIR / "entrypoint.py",
            SHELL_WRAPPER,
            API_SERVICE,
            BACKUP_SERVICE,
            BACKUP_TIMER,
            DOCS_MANUAL,
        ):
            assert path.is_file(), f"missing deploy artifact: {path}"


class TestShellWrapper:
    @staticmethod
    def _code_only(text: str) -> str:
        """Strip comments so 'no eval/source' checks inspect code, not prose."""
        return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))

    def test_wrapper_only_execs_python_launcher(self):
        text = SHELL_WRAPPER.read_text(encoding="utf-8")
        code = self._code_only(text)
        # No shell source/dot-eval of the secret file — sourcing executes it.
        assert "source" not in code
        assert "eval" not in code
        assert ". /run/secrets" not in text and ".${" not in text and ". ${" not in text
        assert "exec python3 -m medcopilot.deploy.entrypoint" in code
        # The secret path is a single defaulted argument, not inlined config.
        assert "${MED_RAG_SECRET_FILE:-/run/secrets/med_env}" in text

    def test_wrapper_scrubs_inherited_interpreter_hooks(self, tmp_path):
        """Execute the wrapper the way Docker does: the hooks it INHERITS must
        not reach the exec'd launcher.

        An empty LD_PRELOAD is loader-safe (glibc/MSYS treat it as unset), so
        this probe can hand the wrapper a real inherited value and let the
        assertions inspect the child environment directly.
        """
        if shutil.which("sh") is None:
            pytest.skip("no POSIX shell available")
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        env_dump = tmp_path / "child_env.txt"
        shim = bin_dir / "python3"
        shim.write_text("#!/bin/sh\nenv > \"$MED_RAG_TEST_ENV\"\nexit 0\n", encoding="utf-8")
        shim.chmod(0o755)
        environ = dict(os.environ)
        environ["PATH"] = f"{bin_dir}{os.pathsep}{environ.get('PATH', '')}"
        environ["MED_RAG_TEST_ENV"] = str(env_dump)
        environ["MED_RAG_SECRET_FILE"] = str(tmp_path / "absent-secret")
        environ["PYTHONPATH"] = "/ambient/evil-pythonpath"
        environ["LD_PRELOAD"] = ""
        proc = subprocess.run(
            [shutil.which("sh"), str(SHELL_WRAPPER)],
            env=environ,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        dumped = env_dump.read_text(encoding="utf-8")
        assert dumped.strip(), "the exec'd child must have run and reported its environment"
        child_env = dict(line.split("=", 1) for line in dumped.splitlines() if "=" in line)
        # unset removes the key entirely — an empty value is not enough.
        assert "PYTHONPATH" not in child_env
        assert "LD_PRELOAD" not in child_env

    def test_wrapper_scrubs_interpreter_hooks_before_python_starts(self, tmp_path):
        """PYTHONPATH/LD_PRELOAD must be gone BEFORE the launcher's python3 starts.

        The launcher's own scrub only protects the process it execs, but an
        ambient PYTHONPATH is already in effect while the launcher imports
        dotenv and parses the secret file, and LD_PRELOAD cannot be removed
        from a running interpreter at all.

        The hooks are exported by an ALREADY-RUNNING shell (the way a
        supervisor hands them to a process mid-life) which then SOURCES the
        wrapper — passing a loader-killing LD_PRELOAD in the test's own
        environment would kill the shell before any script could run, which
        is precisely why the scrub has to live in the wrapper ahead of the
        exec instead of being deferred to the launcher. The wrapper path
        rides in an env var so sourcing is immune to backslashes/$0
        semantics.
        """
        if shutil.which("sh") is None:
            pytest.skip("no POSIX shell available")
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        env_dump = tmp_path / "child_env.txt"
        shim = bin_dir / "python3"
        shim.write_text("#!/bin/sh\nenv > \"$MED_RAG_TEST_ENV\"\nexit 0\n", encoding="utf-8")
        shim.chmod(0o755)
        driver = tmp_path / "driver.sh"
        driver.write_text(
            "#!/bin/sh\n"
            "export PYTHONPATH=/ambient/evil-pythonpath\n"
            "export LD_PRELOAD=/ambient/evil-preload.so\n"
            '. "$MED_RAG_TEST_WRAPPER"\n',
            encoding="utf-8",
        )
        environ = dict(os.environ)
        environ["PATH"] = f"{bin_dir}{os.pathsep}{environ.get('PATH', '')}"
        environ["MED_RAG_TEST_ENV"] = str(env_dump)
        environ["MED_RAG_TEST_WRAPPER"] = str(SHELL_WRAPPER)
        environ["MED_RAG_SECRET_FILE"] = str(tmp_path / "absent-secret")
        proc = subprocess.run(
            [shutil.which("sh"), str(driver)],
            env=environ,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        dumped = env_dump.read_text(encoding="utf-8")
        assert dumped.strip(), "the exec'd child must have run and reported its environment"
        child_env = dict(line.split("=", 1) for line in dumped.splitlines() if "=" in line)
        assert "PYTHONPATH" not in child_env
        assert "LD_PRELOAD" not in child_env
        # Ordering is load-bearing: the scrub must precede the exec, otherwise
        # the launcher's interpreter starts with the hooks still in place.
        code = self._code_only(SHELL_WRAPPER.read_text(encoding="utf-8"))
        assert code.index("unset PYTHONPATH LD_PRELOAD") < code.index("exec python3")

    def test_wrapper_runs_and_forwards_secret_path_argument(self, tmp_path):
        """Actually execute the wrapper: it must exec the launcher untouched."""
        if shutil.which("sh") is None:
            pytest.skip("no POSIX shell available")
        sentinel = tmp_path / "sentinel"
        secret = _hostile_secret_file(tmp_path, sentinel)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        argv_dump = tmp_path / "child_argv.txt"
        env_dump = tmp_path / "child_env.txt"
        # Fake `python3` on PATH: records argv + env, then exits. The wrapper
        # must reach it through exec (no shell parsing of the secret file).
        shim = bin_dir / "python3"
        shim.write_text(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$MED_RAG_TEST_ARGV\"\n"
            "env > \"$MED_RAG_TEST_ENV\"\nexit 0\n",
            encoding="utf-8",
        )
        shim.chmod(0o755)
        environ = dict(os.environ)
        environ["PATH"] = f"{bin_dir}{os.pathsep}{environ.get('PATH', '')}"
        environ["MED_RAG_SECRET_FILE"] = str(secret)
        environ["MED_RAG_TEST_ARGV"] = str(argv_dump)
        environ["MED_RAG_TEST_ENV"] = str(env_dump)
        proc = subprocess.run(
            [shutil.which("sh"), str(SHELL_WRAPPER)],
            env=environ,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        assert argv_dump.read_text(encoding="utf-8").splitlines() == [
            "-m",
            "medcopilot.deploy.entrypoint",
            str(secret),
        ]
        # The wrapper passes the ambient environment through unchanged: it does
        # not read the secret file itself, so nothing in it can be executed.
        child_env = dict(
            line.split("=", 1) for line in env_dump.read_text(encoding="utf-8").splitlines() if "=" in line
        )
        assert child_env.get("MED_RAG_SECRET_FILE") == str(secret)
        assert not sentinel.exists()

    def test_wrapper_ignores_cmd_args_so_the_default_command_works(self, tmp_path):
        """Docker hands the ENTRYPOINT the CMD tokens as arguments.

        The launcher accepts exactly one argument (the secret path), so the
        wrapper must NOT forward the CMD's uvicorn tokens to it — the default
        `docker run` (and any user-passed args) must still reach the exec'd
        single-worker command instead of dying on a usage error.
        """
        if shutil.which("sh") is None:
            pytest.skip("no POSIX shell available")
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        argv_dump = tmp_path / "child_argv.txt"
        shim = bin_dir / "python3"
        shim.write_text(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$MED_RAG_TEST_ARGV\"\nexit 0\n",
            encoding="utf-8",
        )
        shim.chmod(0o755)
        environ = dict(os.environ)
        environ["PATH"] = f"{bin_dir}{os.pathsep}{environ.get('PATH', '')}"
        environ["MED_RAG_SECRET_FILE"] = str(tmp_path / "absent-secret")
        environ["MED_RAG_TEST_ARGV"] = str(argv_dump)
        proc = subprocess.run(
            [
                shutil.which("sh"),
                str(SHELL_WRAPPER),
                # exactly what Docker appends from the image CMD, plus a
                # hypothetical user override:
                "python", "-m", "uvicorn", "medcopilot.api.server:app",
                "--host", "0.0.0.0", "--port", "8000", "--workers", "4",
            ],
            env=environ,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        assert argv_dump.read_text(encoding="utf-8").splitlines() == [
            "-m",
            "medcopilot.deploy.entrypoint",
            str(tmp_path / "absent-secret"),
        ], "the launcher must receive only the secret path; CMD/user args are inert"


class TestLauncherSourceContract:
    def test_no_app_side_import_before_secret_parsing(self):
        """Startup must import the app env only AFTER secrets are parsed."""
        source = (DEPLOY_DIR / "entrypoint.py").read_text(encoding="utf-8")
        assert "import uvicorn" not in source
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import project", "from project")):
                # project imports are allowed only lazily (inside functions),
                # i.e. after the secret-file parsing step in main().
                assert line[:1].isspace(), f"eager project import: {line}"
        assert 'DEFAULT_SECRET_FILE = "/run/secrets/med_env"' in source
        assert "WORKERS = 1" in source

    def test_allowlist_is_exactly_the_appconfig_aliases(self):
        from medcopilot.common.config import AppConfig

        aliases = {field.alias for field in AppConfig.model_fields.values() if field.alias}
        assert set(entrypoint_module.ALLOWLIST_ENV_KEYS) == aliases
        assert "PYTHONPATH" not in entrypoint_module.ALLOWLIST_ENV_KEYS
        assert "LD_PRELOAD" not in entrypoint_module.ALLOWLIST_ENV_KEYS


# --------------------------------------------------------- behavioral: launcher


class TestLauncherSecretsAreData:
    @pytest.fixture(autouse=True)
    def _restore_environ_after_launcher_runs(self):
        """Undo the launcher's environment writes once the test finishes.

        ``entrypoint.main`` exports the secret allowlist into ``os.environ`` and
        then execs. With ``execv`` faked the process survives with that
        environment, so a value like ``LOG_FILE=/var/log/${HOME}-subst`` would be
        inherited by every later test that spawns a subprocess — where loguru's
        path formatting turns it into ``KeyError: 'HOME'``. The launcher's real
        behaviour is unchanged; the test just stops leaking into its neighbours.
        """
        before = dict(os.environ)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(before)

    def _capture_exec(self, monkeypatch) -> tuple[list[str], dict[str, str]]:
        calls: dict[str, object] = {}

        def fake_execv(_path: str, argv: list[str]) -> None:
            calls["argv"] = argv
            calls["environ"] = dict(os.environ)

        monkeypatch.setattr(os, "execv", fake_execv)
        return lambda: calls["argv"], lambda: calls["environ"]

    def test_hostile_values_stay_literal_and_allowlist_filters(self, monkeypatch, tmp_path, capsys):
        get_argv, get_environ = self._capture_exec(monkeypatch)
        sentinel = tmp_path / "sentinel"
        secret = _hostile_secret_file(tmp_path, sentinel)
        monkeypatch.setenv("PYTHONPATH", "/tmp/ambient-evil")  # inherited hooks must be scrubbed
        monkeypatch.setenv("ADMIN_API_KEY", "ambient-placeholder")

        # Snapshot the module table first: the claim is that THIS call imports no
        # app code, not that nothing else in the session ever did (a purely global
        # check would silently depend on test-file ordering).
        modules_before = set(sys.modules)
        entrypoint_module.main([str(secret)])

        argv = get_argv()
        child_env = get_environ()
        # 1) uvicorn, single worker, right app module
        assert argv[1:3] == ["-m", "uvicorn"]
        assert argv[3] == "medcopilot.api.server:app"
        assert "--workers" in argv and argv[argv.index("--workers") + 1] == "1"
        # 2) shell syntax in values is inert data — nothing executed
        assert not sentinel.exists()
        assert child_env["LOG_FILE"] == "/var/log/${HOME}-subst"
        assert child_env["AUDIT_NTP_SERVER"] == f"$(touch {sentinel})"
        assert child_env["LLM_API_BASE"] == "$HOME-plain"
        assert child_env["LLM_MODEL"] == f"`touch {sentinel}`"
        # 3) only allowlisted keys were exported; hooks never enter or survive
        assert child_env["ADMIN_API_KEY"] == SYNTHETIC_SECRET  # secret applied over placeholder
        assert "PYTHONPATH" not in child_env
        assert "LD_PRELOAD" not in child_env
        assert "MED_RAG_SECRET_FILE" not in child_env
        # 4) no app import happened in-process (exec boundary intact)
        newly_imported = set(sys.modules) - modules_before
        assert "uvicorn" not in newly_imported
        assert "medcopilot.api.server" not in newly_imported
        # 5) diagnostics never print secret values
        err = capsys.readouterr().err
        assert SYNTHETIC_SECRET not in err
        assert "applied 6 allowlisted key(s), ignored 4 non-AppConfig key(s)" in err
        assert "ignored 4" in err  # PYTHONPATH/LD_PRELOAD/MED_RAG_SECRET_FILE/SHLVL

    def test_ambient_environment_fallback_without_secret_file(self, monkeypatch, capsys):
        get_argv, get_environ = self._capture_exec(monkeypatch)
        monkeypatch.setenv("PROJECT_ENV", "production")
        monkeypatch.setenv("ADMIN_API_KEY", "ambient-only-key")
        monkeypatch.delenv("PYTHONPATH", raising=False)

        entrypoint_module.main(["/nonexistent/secret/file"])

        assert get_argv()[1:3] == ["-m", "uvicorn"]
        assert get_environ()["ADMIN_API_KEY"] == "ambient-only-key"
        assert "using ambient environment" in capsys.readouterr().err

    @pytest.mark.parametrize("project_env", ["", "development", "prodction"])
    def test_project_env_must_classify_as_production(self, monkeypatch, tmp_path, project_env):
        if not project_env:
            monkeypatch.delenv("PROJECT_ENV", raising=False)
        else:
            monkeypatch.setenv("PROJECT_ENV", project_env)
        secret = tmp_path / "med_env"
        secret.write_text("LLM_MODEL=x\n", encoding="utf-8")

        with pytest.raises(entrypoint_module.EntrypointError, match="PROJECT_ENV"):
            entrypoint_module.main([str(secret)])

    def test_prod_literal_is_accepted_as_production(self, monkeypatch):
        get_argv, get_environ = self._capture_exec(monkeypatch)
        monkeypatch.setenv("PROJECT_ENV", "prod")

        entrypoint_module.main(["/nonexistent/secret/file"])

        assert get_argv()[1:3] == ["-m", "uvicorn"]
        assert get_environ()["PROJECT_ENV"] == "prod"

    def test_oversized_secret_file_is_refused(self, monkeypatch, tmp_path):
        self._capture_exec(monkeypatch)
        monkeypatch.setenv("PROJECT_ENV", "production")
        secret = tmp_path / "med_env"
        secret.write_bytes(b"A" * (entrypoint_module.MAX_SECRET_BYTES + 1))

        with pytest.raises(entrypoint_module.EntrypointError, match="too large"):
            entrypoint_module.main([str(secret)])

    def test_secret_path_pointing_at_directory_is_refused(self, monkeypatch, tmp_path):
        self._capture_exec(monkeypatch)
        monkeypatch.setenv("PROJECT_ENV", "production")

        with pytest.raises(entrypoint_module.EntrypointError, match="not a regular file"):
            entrypoint_module.main([str(tmp_path)])

    def test_broken_symlink_secret_path_fails_closed(self, monkeypatch, tmp_path):
        """A dangling secret mount must fail loudly, not fall back silently.

        ``Path.exists()`` follows symlinks and answers False for a broken one,
        so the old ``if secret_path.exists()`` gate sent a dangling mount down
        the ambient fallback — the misconfiguration an operator most needs to
        see. The symlink must reach _parse_secret_file, which refuses it.
        """
        link = tmp_path / "med_env"
        try:
            link.symlink_to(tmp_path / "missing-secret-target")
        except OSError:  # pragma: no cover - symlink privilege missing
            pytest.skip("symlinks unavailable on this platform/config")
        self._capture_exec(monkeypatch)
        monkeypatch.setenv("PROJECT_ENV", "production")
        monkeypatch.setenv("ADMIN_API_KEY", "ambient-only-key")

        with pytest.raises(entrypoint_module.EntrypointError, match="broken symlink"):
            entrypoint_module.main([str(link)])

    def test_too_many_arguments_are_refused(self, monkeypatch):
        self._capture_exec(monkeypatch)

        with pytest.raises(entrypoint_module.EntrypointError, match="usage"):
            entrypoint_module.main(["a", "b"])


# ------------------------------------------------------------ container image


class TestContainerEntrypoint:
    """C6: the documented ``docker run`` must start through the deploy wrapper.

    A bare CMD left ``/run/secrets/med_env`` unparsed (service 503 on the
    missing key) and invited operators to patch ``-e ADMIN_API_KEY=...`` —
    which writes the secret into image inspect/history. ENTRYPOINT also has to
    survive build hosts that lose POSIX file modes (Windows contexts,
    ``core.fileMode=false`` checkouts): the exec bit is chmod'ed inside the
    image, never trusted from the git index or the context tar.
    """

    @staticmethod
    def _dockerfile() -> str:
        return DOCKERFILE.read_text(encoding="utf-8")

    def test_entrypoint_is_exec_form_and_points_at_the_wrapper(self):
        text = self._dockerfile()
        entry_lines = [line.strip() for line in text.splitlines() if line.startswith("ENTRYPOINT")]
        assert entry_lines == [f'ENTRYPOINT ["{WRAPPER_PATH_IN_IMAGE}"]'], entry_lines
        # Exec form (JSON array): a shell-string ENTRYPOINT would wrap the
        # wrapper in /bin/sh -c and break signal delivery to the launcher.

    def test_wrapper_is_chmodded_inside_the_image_as_root(self):
        text = self._dockerfile()
        assert f"RUN chmod 0755 {WRAPPER_PATH_IN_IMAGE}" in text
        # Must run before the USER switch: only root can chmod a file it may
        # not own, and the exec bit must exist for the appuser runtime user.
        assert text.index(f"chmod 0755 {WRAPPER_PATH_IN_IMAGE}") < text.index("\nUSER appuser")

    def test_single_worker_command_is_pinned(self):
        text = self._dockerfile()
        assert '"--workers", "1"' in text
        assert "medcopilot.api.server:app" in text

    def test_container_timezone_is_pinned_utc(self):
        # PRODUCTION_DEPLOY.md §0 promises the container runs TZ=UTC so audit
        # and access-log timestamps share the audit trail's timezone.
        assert "TZ=UTC" in self._dockerfile()


# ------------------------------------------------------------------ systemd


class TestSystemdUnits:
    def test_api_unit_is_nonroot_single_worker_hardened(self):
        parser = _read_unit(API_SERVICE)
        service = dict(parser.items("Service"))
        assert service["User"] == "medrag" and service["Group"] == "medrag"
        assert service["EnvironmentFile"] == "/etc/med-rag/med-rag.env"
        assert service["ExecStart"] == (
            "/usr/bin/python3 -m uvicorn medcopilot.api.server:app "
            "--host 127.0.0.1 --port 8000 --workers 1"
        )
        assert service["Restart"] == "on-failure"
        for key in ("NoNewPrivileges", "PrivateTmp", "ProtectHome", "ProtectKernelModules"):
            assert service.get(key) == "yes", key
        assert service["ProtectSystem"] == "strict"
        assert "/var/lib/med-rag" in service["ReadWritePaths"]
        unit = dict(parser.items("Unit"))
        assert "time-sync.target" in unit["After"]  # NTP is an audit prerequisite

    def test_backup_unit_takes_paths_from_the_env_file(self):
        parser = _read_unit(BACKUP_SERVICE)
        service = dict(parser.items("Service"))
        assert service["Type"] == "oneshot"
        assert service["User"] == "medrag"
        assert service["EnvironmentFile"] == "/etc/med-rag/med-rag.env"
        # Double-quoted so systemd does not word-split a path that contains
        # spaces (unquoted ${VAR} expansion splits on whitespace).
        assert service["ExecStart"] == (
            "/usr/bin/python3 -m medcopilot.scripts.backup_snapshot create "
            '"${MED_CORPUS_DIR}" "${MED_BACKUP_DIR}"'
        )
        # Unset variables must fail the unit before an empty path is used.
        assert _pre_lines(BACKUP_SERVICE) == [
            'ExecStartPre=/usr/bin/test -n "${MED_CORPUS_DIR}"',
            'ExecStartPre=/usr/bin/test -n "${MED_BACKUP_DIR}"',
        ]
        assert service["ReadWritePaths"] == "/var/lib/med-rag"

    def test_backup_timer_is_exact_daily_without_jitter(self):
        parser = _read_unit(BACKUP_TIMER)
        timer = dict(parser.items("Timer"))
        assert timer["OnCalendar"] == "daily"
        assert timer["Persistent"] == "true"
        assert timer["Unit"] == "med-rag-backup.service"
        assert "WantedBy" in dict(parser.items("Install"))
        for path in (API_SERVICE, BACKUP_SERVICE, BACKUP_TIMER):
            # comments may mention the option; the directive itself is banned
            code = "\n".join(
                line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")
            )
            assert "RandomizedDelaySec" not in code, "backup calendar must stay deterministic"

    @pytest.mark.parametrize("unit", ["med-rag.service", "med-rag-backup.service"])
    def test_exec_commands_are_absolute_and_shell_free(self, unit):
        for line in _exec_lines(DEPLOY_DIR / unit):
            assert line.startswith("/usr/bin/"), line
            # systemd-level ``${VAR}`` expansion and quoting are fine; shell
            # control operators are not (systemd never runs them through sh).
            for metachar in (";", "&&", "|", "$(", "`"):
                assert metachar not in line, f"{metachar!r} in {line!r}"

    @pytest.mark.parametrize("unit", ["med-rag.service", "med-rag-backup.service", "med-rag-backup.timer"])
    def test_units_carry_no_secrets(self, unit):
        text = (DEPLOY_DIR / unit).read_text(encoding="utf-8")
        assert "ADMIN_API_KEY=" not in text
        assert "Bearer " not in text
        assert "your-" not in text and "<key" not in text.lower()


# --------------------------------------------------------------------- manual


class TestManualContract:
    @pytest.fixture(scope="module")
    def manual(self) -> str:
        return DOCS_MANUAL.read_text(encoding="utf-8")

    def test_manual_documents_ops_surface(self, manual):
        for needle in (
            "timedatectl",
            "时区",
            "/health/live",
            "/health",
            "/metrics",
            "MED_CORPUS_DIR",
            "MED_BACKUP_DIR",
            "OnCalendar=daily",
            "PROJECT_ENV=production",
            "--workers 1",
            "journalctl",
            "audit.jsonl",
        ):
            assert needle in manual, needle

    def test_manual_states_objectives_honestly(self, manual):
        assert "RPO ≤ 24h" in manual and "RTO ≤ 30min" in manual
        assert "目标（objectives）" in manual
        assert "未经演练的备份视为不存在" in manual
        # Clinical readiness is explicitly NOT proven by engineering evidence.
        assert "不由本手册或任何工程证据证明" in manual

    def test_manual_explains_docker_contract(self, manual):
        for needle in (
            "docker-entrypoint.sh",
            "entrypoint.py",
            "python-dotenv",
            "interpolate=False",
            "PYTHONPATH",
            "uv export --require-hashes",
            "runtime",
            "BAKE_WEIGHTS",
            "ENABLE_CHROMA_DENSE",
            "DISCLAIMER_TEXT",
        ):
            assert needle in manual, needle

    def test_manual_documents_the_container_entrypoint(self, manual):
        # C6: the manual must state that the image starts through the wrapper
        # (ENTRYPOINT) so the documented `docker run` cannot bypass the secret
        # channel, and that the exec bit is chmod'ed inside the image.
        assert "ENTRYPOINT" in manual
        # The documented run command must not need (or carry) an override.
        assert "--entrypoint" not in manual
        assert "chmod" in manual

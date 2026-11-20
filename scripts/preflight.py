"""
Phase 8 pre-flight checklist. Run before demo.
Prints PASS/FAIL/WARN for each check.

Usage:
  .venv\\Scripts\\python.exe scripts/preflight.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Project root on path so app.* imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Colour helpers (works on Windows 10+ and most terminals)
# ---------------------------------------------------------------------------
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"

# Enable ANSI on Windows
if sys.platform == "win32":
    os.system("")  # triggers VT100 mode in conhost


def _pass(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  {_GREEN}✓ PASS{_RESET}  {label}{suffix}")


def _fail(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  {_RED}✗ FAIL{_RESET}  {label}{suffix}")


def _warn(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  {_YELLOW}⚠ WARN{_RESET}  {label}{suffix}")


def check_config() -> bool:
    """config.yaml loads without error."""
    try:
        from app.config import load_config

        load_config("config.yaml")
        _pass("config.yaml loads without error")
        return True
    except Exception as exc:
        _fail("config.yaml loads without error", str(exc))
        return False


def check_db_init() -> bool:
    """DB initializes via init_db on a temp file."""
    try:
        from app.storage.db import init_db

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            init_db(tmp_path)
            _pass("DB initializes (schema.sql applied)")
            return True
        finally:
            tmp_path.unlink(missing_ok=True)
    except Exception as exc:
        _fail("DB initializes", str(exc))
        return False


def check_insightface_model_files() -> bool:
    """InsightFace buffalo_s model files exist in repo or ~/.insightface/models/buffalo_s/."""
    candidates = [
        Path("models/models/buffalo_s"),
        Path("models/buffalo_s"),
        Path.home() / ".insightface" / "models" / "buffalo_s",
    ]
    for c in candidates:
        try:
            if c.exists() and any(c.iterdir()):
                files = [f.name for f in c.iterdir()]
                _pass("InsightFace buffalo_s model files found", f"{len(files)} files in {c}")
                return True
        except Exception:
            continue

    _fail(
        "InsightFace buffalo_s model files",
        "Not found in models/ or ~/.insightface. Run: .venv\\Scripts\\python.exe scripts/download_models.py",
    )
    return False


def check_recognition_model_loads() -> bool:
    """FaceAnalysis can be instantiated and prepared."""
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_s",
            root="models",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        _pass("Recognition model (InsightFace buffalo_s) loads")
        return True
    except Exception as exc:
        _fail("Recognition model loads", str(exc)[:120])
        return False


def check_liveness_model() -> bool:
    """Liveness model path exists (WARN if missing, not FAIL)."""
    try:
        from app.config import load_config

        cfg = load_config("config.yaml")
        liveness_path = Path(cfg.liveness.model_path)
    except Exception:
        liveness_path = Path("models/minifasnet.onnx")

    if liveness_path.exists():
        _pass("Liveness model file exists", str(liveness_path))
        return True
    else:
        _warn(
            "Liveness model not found",
            f"Expected at {liveness_path}. App will run in disabled mode (no spoof detection).",
        )
        return True  # WARN, not FAIL


def check_reports_writable() -> bool:
    """reports/ directory is writable (created if absent)."""
    reports_dir = Path("reports")
    try:
        reports_dir.mkdir(parents=True, exist_ok=True)
        probe = reports_dir / "_preflight_probe.tmp"
        probe.write_text("ok")
        probe.unlink()
        _pass("reports/ directory is writable")
        return True
    except Exception as exc:
        _fail("reports/ directory is writable", str(exc))
        return False


def check_data_writable() -> bool:
    """data/ directory is writable (created if absent)."""
    data_dir = Path("data")
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / "_preflight_probe.tmp"
        probe.write_text("ok")
        probe.unlink()
        _pass("data/ directory is writable")
        return True
    except Exception as exc:
        _fail("data/ directory is writable", str(exc))
        return False


def check_python_version() -> bool:
    """Python >= 3.11."""
    major, minor = sys.version_info.major, sys.version_info.minor
    version_str = f"{major}.{minor}.{sys.version_info.micro}"
    if (major, minor) >= (3, 11):
        _pass("Python version >= 3.11", f"Found {version_str}")
        return True
    else:
        _fail("Python version >= 3.11", f"Found {version_str}")
        return False


def check_no_conflicting_processes() -> None:
    """No conflicting processes (optional — skip if psutil not installed)."""
    try:
        import psutil  # type: ignore

        suspicious = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = " ".join(proc.info.get("cmdline") or [])
                if "run_live.py" in cmdline or "streamlit_app" in cmdline:
                    suspicious.append(f"pid={proc.pid} {proc.info['name']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if suspicious:
            _warn("Conflicting processes", "; ".join(suspicious))
        else:
            _pass("No conflicting processes detected")
    except ImportError:
        _warn("Process check skipped", "psutil not installed (optional)")


def main() -> None:
    print("\n=== Pre-flight Checklist ===\n")

    failures = 0

    # Critical checks (FAILs count)
    checks = [
        check_config,
        check_db_init,
        check_insightface_model_files,
        check_recognition_model_loads,
        check_liveness_model,  # WARN only
        check_reports_writable,
        check_data_writable,
        check_python_version,
    ]

    for fn in checks:
        result = fn()
        if result is False:
            failures += 1

    # Optional check (never increments failures)
    check_no_conflicting_processes()

    print()
    if failures == 0:
        print(f"{_GREEN}All critical checks passed.{_RESET} System is ready for demo.\n")
        sys.exit(0)
    else:
        print(f"{_RED}{failures} critical check(s) FAILED.{_RESET} Fix issues above before running the demo.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

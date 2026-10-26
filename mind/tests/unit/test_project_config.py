"""project_config resolves credentials and cache from MIND_HOME or the user config dir."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
REPO_ROOT = SOURCE_ROOT.parents[1]
KEYS = "OPENROUTER_API_KEY: 'k'\nPERPLEXITY_API_KEY: 'k'\n"


def _import_in_subprocess(env_overrides, remove=()):
    """Import project_config in a fresh interpreter so import-time effects are real."""
    env = os.environ.copy()
    for name in remove:
        env.pop(name, None)
    env.update(env_overrides)
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import mind.project_config as c; print(c.CREDENTIALS_DIR); print(c.CACHE_DIR)",
        ],
        capture_output=True,
        text=True,
        env=env,
    )


def _package_tree():
    return {p for p in (SOURCE_ROOT / "mind").rglob("*") if "__pycache__" not in p.parts}


def _resolve(*args):
    from mind.mind_home import resolve_mind_home

    return resolve_mind_home(*args)


def test_mind_home_wins_over_every_platform_default():
    env = {"MIND_HOME": "/srv/mind", "XDG_CONFIG_HOME": "/xdg", "APPDATA": "/appdata"}
    for platform in ("linux", "win32", "darwin"):
        assert Path(_resolve(env, platform, "/home/u")) == Path("/srv/mind").resolve()


def test_linux_default_honours_xdg_then_dot_config():
    assert _resolve({"XDG_CONFIG_HOME": "/xdg"}, "linux", "/home/u") == os.path.join(
        "/xdg", "npc-mind"
    )
    assert _resolve({}, "linux", "/home/u") == os.path.join("/home/u", ".config", "npc-mind")


def test_windows_default_uses_appdata():
    assert _resolve({"APPDATA": "/appdata"}, "win32", "/home/u") == os.path.join(
        "/appdata", "npc-mind"
    )


def test_macos_default_uses_application_support():
    expected = os.path.join("/home/u", "Library", "Application Support", "npc-mind")
    assert _resolve({}, "darwin", "/home/u") == expected


def test_empty_mind_home_is_treated_as_unset():
    assert _resolve({"MIND_HOME": ""}, "linux", "/home/u") == os.path.join(
        "/home/u", ".config", "npc-mind"
    )


def test_import_places_credentials_and_cache_under_mind_home_not_the_checkout(tmp_path):
    home = tmp_path / "mindhome"
    (home / "credentials").mkdir(parents=True)
    (home / "credentials" / "api_keys.cfg").write_text(KEYS)
    tree_before = _package_tree()
    cache_existed = (REPO_ROOT / "cache").exists()

    result = _import_in_subprocess({"MIND_HOME": str(home)})

    assert result.returncode == 0, result.stderr
    creds, cache = result.stdout.split("\n")[:2]
    assert Path(creds) == home / "credentials"
    assert Path(cache) == home / "cache"
    assert (home / "cache").is_dir()
    assert _package_tree() == tree_before
    assert (REPO_ROOT / "cache").exists() == cache_existed


def test_missing_keys_fail_at_import_naming_the_path_and_the_override(tmp_path):
    home = tmp_path / "empty"

    result = _import_in_subprocess({"MIND_HOME": str(home)})

    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
    assert str(home / "credentials" / "api_keys.cfg") in result.stderr
    assert "MIND_HOME" in result.stderr


def test_partial_keys_file_fails_at_import_naming_the_path(tmp_path):
    home = tmp_path / "partial"
    (home / "credentials").mkdir(parents=True)
    (home / "credentials" / "api_keys.cfg").write_text("OPENROUTER_API_KEY: 'k'\n")

    result = _import_in_subprocess({"MIND_HOME": str(home)})

    assert result.returncode != 0
    assert "KeyNotFoundError" in result.stderr
    assert "PERPLEXITY_API_KEY" in result.stderr
    assert str(home / "credentials" / "api_keys.cfg") in result.stderr
    assert "MIND_HOME" in result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="XDG_CONFIG_HOME is the Linux/WSL default")
def test_unset_mind_home_defaults_to_xdg_config_home(tmp_path):
    xdg = tmp_path / "xdg"
    (xdg / "npc-mind" / "credentials").mkdir(parents=True)
    (xdg / "npc-mind" / "credentials" / "api_keys.cfg").write_text(KEYS)

    result = _import_in_subprocess({"XDG_CONFIG_HOME": str(xdg)}, remove=("MIND_HOME",))

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.split("\n")[0]) == xdg / "npc-mind" / "credentials"

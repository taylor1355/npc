"""Where the mind keeps credentials and cache, independent of the install location.

Everything lives under one base directory: ``$MIND_HOME`` when set, otherwise the
platform's user config directory. The base is never derived from ``__file__``: an
installed package sits in site-packages, where credentials would be lost on every
upgrade and directories could not be created at all.

Kept free of import-time effects so every platform's default can be checked from
any host.
"""

import os
import sys
from collections.abc import Mapping

MIND_HOME_ENV = "MIND_HOME"
_APP_DIR = "npc-mind"


def resolve_mind_home(
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: str | None = None,
) -> str:
    """Return the base directory: ``$MIND_HOME``, else the user config directory.

    Arguments default to the live process.
    """
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    home = os.path.expanduser("~") if home is None else home

    override = environ.get(MIND_HOME_ENV)
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if platform == "win32":
        base = environ.get("APPDATA") or os.path.join(home, "AppData", "Roaming")
    elif platform == "darwin":
        base = os.path.join(home, "Library", "Application Support")
    else:
        base = environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")
    return os.path.join(base, _APP_DIR)

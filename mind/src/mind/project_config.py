"""Credentials and cache locations; see mind.mind_home for how the base is chosen."""

import os

import config

from mind.mind_home import MIND_HOME_ENV, resolve_mind_home

MIND_HOME = resolve_mind_home()
CACHE_DIR = os.path.join(MIND_HOME, "cache")
CREDENTIALS_DIR = os.path.join(MIND_HOME, "credentials")
for dir in [CACHE_DIR, CREDENTIALS_DIR]:
    os.makedirs(dir, exist_ok=True)

API_KEYS_PATH = os.path.join(CREDENTIALS_DIR, "api_keys.cfg")


def _load_api_keys(path):
    """Load the keys file, failing with the resolved path and the override that moves it."""
    try:
        keys = config.Config(path)
        return keys, keys["PERPLEXITY_API_KEY"], keys["OPENROUTER_API_KEY"]
    except (FileNotFoundError, config.KeyNotFoundError) as error:
        raise type(error)(
            f"Mind API keys are missing or incomplete at {path} "
            f"(needs PERPLEXITY_API_KEY and OPENROUTER_API_KEY). "
            f"Set {MIND_HOME_ENV} to use a different base directory "
            f"(currently {MIND_HOME}): {error}"
        ) from error


API_KEYS, PERPLEXITY_API_KEY, OPENROUTER_API_KEY = _load_api_keys(API_KEYS_PATH)

GMAIL_CREDENTIALS_PATH = os.path.join(CREDENTIALS_DIR, "gmail_credentials.json")
GMAIL_TOKEN_PATH = os.path.join(CREDENTIALS_DIR, "gmail_token.pickle")
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.labels",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.settings.basic",
]

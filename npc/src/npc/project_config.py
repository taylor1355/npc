import config
import os

# TODO: fix the directory structure. npcs/npc/src/npc is too redundant
# TODO: move authetication files (creds, api keys, tokens, etc) to a subdirectory instead of the root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
CREDENTIALS_DIR = PROJECT_ROOT
os.makedirs(CREDENTIALS_DIR, exist_ok=True)

API_KEYS_PATH = os.path.join(CREDENTIALS_DIR, "api_keys.cfg")
API_KEYS = config.Config(API_KEYS_PATH)
ANTHROPIC_API_KEY = API_KEYS["ANTHROPIC_API_KEY"]
PERPLEXITY_API_KEY = API_KEYS["PERPLEXITY_API_KEY"]

GMAIL_CREDENTIALS_PATH = os.path.join(CREDENTIALS_DIR, "gmail_credentials.json")
GMAIL_TOKEN_PATH = os.path.join(CREDENTIALS_DIR, "gmail_token.pickle")
GMAIL_SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.labels',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.settings.basic',
]

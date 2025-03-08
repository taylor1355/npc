import config
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CREDENTIALS_DIR = os.path.join(PROJECT_ROOT, "credentials")
for dir in [CACHE_DIR, CREDENTIALS_DIR]:
    os.makedirs(dir, exist_ok=True)

API_KEYS_PATH = os.path.join(CREDENTIALS_DIR, "api_keys.cfg")
API_KEYS = config.Config(API_KEYS_PATH)
PERPLEXITY_API_KEY = API_KEYS["PERPLEXITY_API_KEY"]
OPENROUTER_API_KEY = API_KEYS["OPENROUTER_API_KEY"]

GMAIL_CREDENTIALS_PATH = os.path.join(CREDENTIALS_DIR, "gmail_credentials.json")
GMAIL_TOKEN_PATH = os.path.join(CREDENTIALS_DIR, "gmail_token.pickle")
GMAIL_SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.labels',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.settings.basic',
]

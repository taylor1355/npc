# Setup
## API setup
The following files should be found in `npc.project_config.CREDENTIALS_DIR` directory of the repo:

- *api_keys.cfg*
```
ANTHROPIC_API_KEY:"{anthropic api key}"
OPENAI_API_KEY:"{openai api key}"
PERPLEXITY_API_KEY:"{perplexity api key}"
```

- *gmail_credentials.json* (follow [these](https://developers.google.com/workspace/guides/create-credentials#oauth-client-id) instructions)

## Installation
```
poetry install
poetry run nbstripout --install
```

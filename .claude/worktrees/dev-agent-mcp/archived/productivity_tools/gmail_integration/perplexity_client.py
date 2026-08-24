import requests
from perplexipy import PerplexityClient as ExternalPerplexityClient

from npc.project_config import PERPLEXITY_API_KEY

DEFAULT_MODEL = "llama-3.1-sonar-small-128k-online"


def extract_final_url(url):
    try:
        response = requests.get(
            url,
            headers={'User-Agent': 'Mozilla/5.0'},
            allow_redirects=True,
            timeout=10
        )
        return response.url
    except requests.RequestException:
        return None


# Refer to https://cime-software.github.io/perplexipy/perplexipy.html#PerplexityClient
class PerplexityClient(ExternalPerplexityClient):
    def __init__(self, model=DEFAULT_MODEL):
        super().__init__(key=PERPLEXITY_API_KEY)
        self.model = model

    def summarize_url(self, url: str) -> str:
        final_url = extract_final_url(url)
        return self.query(f"Please summarize the content found at the URL: {final_url}")
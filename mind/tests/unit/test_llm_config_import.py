"""Import-only helpers must not require credentials; LLM construction still does."""

import os
import subprocess
import sys
from pathlib import Path


def test_cache_policy_import_is_credential_free_but_llm_construction_is_not():
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from unittest.mock import patch
with patch("config.Config", side_effect=FileNotFoundError("missing test config")) as loader:
    from mind.apis.langchain_llm import get_llm, supports_cache_control
    assert isinstance(supports_cache_control("unrecognized-model"), bool)
    loader.assert_not_called()
    with patch("mind.apis.langchain_llm.ChatOpenAI") as client:
        try:
            get_llm("unrecognized-model")
        except FileNotFoundError as error:
            assert str(error) == "missing test config"
        else:
            raise AssertionError("real LLM construction bypassed missing configuration")
        loader.assert_called_once()
        client.assert_not_called()
""",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_http_app_factory_is_credential_free_but_server_startup_is_not():
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from unittest.mock import MagicMock, patch
with patch("config.Config", side_effect=FileNotFoundError("missing test config")) as loader:
    from mind.interfaces.mcp.main import create_starlette_app, main
    loader.assert_not_called()
    app = create_starlette_app(MagicMock())
    assert app is not None
    loader.assert_not_called()
    with patch("sys.argv", ["mind-server"]), \
         patch("mind.interfaces.mcp.main.uvicorn.run") as serve, \
         patch("mind.interfaces.mcp.main.MCPServer") as server:
        try:
            main()
        except FileNotFoundError as error:
            assert str(error) == "missing test config"
        else:
            raise AssertionError("real server startup bypassed missing configuration")
        loader.assert_called_once()
        server.assert_not_called()
        serve.assert_not_called()
""",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_local_llm_construction_loads_no_credential_and_targets_the_local_server():
    """The experiment tier's local backend (NPC-1916) must run on a machine with no
    OpenRouter key, and must point the same client class at the local endpoint."""
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from unittest.mock import patch
with patch("config.Config", side_effect=FileNotFoundError("missing test config")) as loader:
    from mind.apis.langchain_llm import (
        OLLAMA_DEFAULT_BASE_URL, get_local_llm, supports_cache_control,
    )
    with patch("mind.apis.langchain_llm.ChatOpenAI") as client:
        get_local_llm("qwen3:8b", reasoning_effort="none")
        get_local_llm("gemma3:4b")
    loader.assert_not_called()
    first, second = client.call_args_list
    assert first.kwargs["openai_api_base"] == OLLAMA_DEFAULT_BASE_URL
    assert first.kwargs["reasoning_effort"] == "none"
    assert "reasoning_effort" not in second.kwargs
    assert first.kwargs["max_retries"] == 0
    assert not supports_cache_control("qwen3:8b")
""",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

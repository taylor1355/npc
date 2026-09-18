"""The server's two outward-facing bounds: debug exposure, and call duration.

Both defaults were previously implicit, and both failed open - debug mode was
hard-coded on, and the provider call had no timeout at all. These tests pin the
safe value so a future edit has to change an assertion to undo it.
"""

import logging
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from mind.apis.langchain_llm import (
    LLM_CLIENT_MAX_RETRIES,
    LLM_REQUEST_TIMEOUT_SECONDS,
    get_llm,
)
from mind.interfaces.mcp.main import LOG_HANDLER, create_starlette_app, main


@contextmanager
def _isolated_logging():
    """Undo the process-global logging state `main()` installs.

    main() attaches LOG_HANDLER to the ROOT logger and pins the root and "mind"
    levels - deliberately, since it is a program entry point. Calling it
    in-process therefore leaks into every later test in the session, and the
    leak is silent: it surfaces as unrelated /logs assertions finding no
    messages, because the root level it left behind filters them out. Observed
    exactly that, on tests/integration/test_http_endpoints.py::TestLogsEndpoint.

    (tests/unit/test_llm_config_import.py sidesteps this by running main() in a
    subprocess. That is not available here - these assertions need the mock's
    call_args back.)
    """
    root = logging.getLogger()
    mind_logger = logging.getLogger("mind")
    saved = (list(root.handlers), root.level, mind_logger.level, list(LOG_HANDLER.logs))
    try:
        yield
    finally:
        handlers, root_level, mind_level, buffered = saved
        root.handlers[:] = handlers
        root.setLevel(root_level)
        mind_logger.setLevel(mind_level)
        LOG_HANDLER.logs.clear()
        LOG_HANDLER.logs.extend(buffered)


class TestStarletteDebugIsOptIn:
    """Debug mode returns rendered tracebacks to whoever made the request."""

    def test_app_factory_defaults_to_debug_off(self):
        assert create_starlette_app(MagicMock()).debug is False

    def test_server_launched_with_no_flags_runs_with_debug_off(self):
        """The shipped launch path.

        Both launchers in the simulation repo invoke this module as
        `-m mind.interfaces.mcp.main --port N` and pass no --debug
        (`tools/setup_mind.sh::_mind_run_script` and
        `mcp_server_manager.gd`), so no-flags is the configuration that
        actually runs.
        """
        assert self._debug_flag_for(["mind-server", "--port", "8000"]) is False

    def test_debug_is_reachable_when_explicitly_asked_for(self):
        """Guards the guard: an app that could never enable debug would pass
        the assertion above for the wrong reason."""
        assert self._debug_flag_for(["mind-server", "--debug"]) is True

    @staticmethod
    def _debug_flag_for(argv: list[str]) -> bool:
        with (
            _isolated_logging(),
            patch("sys.argv", argv),
            patch("mind.interfaces.mcp.main.uvicorn.run"),
            patch("mind.interfaces.mcp.main.MCPServer"),
            patch("mind.project_config.OPENROUTER_API_KEY", "test-key"),
            patch("mind.interfaces.mcp.main.create_starlette_app") as factory,
        ):
            main()
            factory.assert_called_once()
            return factory.call_args.kwargs["debug"]


class TestProviderCallsAreBounded:
    """A call with no timeout wedges the decision forever. The game side applies
    no deadline of its own today, and [NPC-1682] — which adds a 60s per-attempt
    client deadline — is in flight rather than merged, so this bound must hold
    on its own AND stay inside that 60s once it lands."""

    @pytest.fixture
    def llm(self):
        with patch("mind.project_config.OPENROUTER_API_KEY", "test-key"):
            return get_llm("google/gemini-2.5-flash-lite")

    def test_the_timeout_reaches_the_http_layer(self, llm):
        """Asserted at httpx, not on the ChatOpenAI field.

        The field is where the bug hid: langchain-openai passes its
        `request_timeout` down unconditionally, and its None default overrides
        the OpenAI SDK's own 600s default, arriving as Timeout(timeout=None).
        Only the transport's view distinguishes bounded from unbounded.
        """
        timeout = llm.root_client._client.timeout
        assert timeout.read == LLM_REQUEST_TIMEOUT_SECONDS
        assert timeout.connect is not None, "connect phase left unbounded"
        assert timeout.write is not None, "write phase left unbounded"
        assert timeout.pool is not None, "pool phase left unbounded"

    def test_client_retry_is_disabled_so_it_cannot_multiply_node_retry(self, llm):
        """LLMNode already retries. The OpenAI SDK defaults to 2 more, and the
        two multiply rather than compose."""
        assert llm.root_client.max_retries == LLM_CLIENT_MAX_RETRIES == 0

    def test_the_bound_is_a_real_number(self):
        """A None or 0 timeout is httpx's spelling of 'no limit', so a
        regression to either would restore the unbounded behavior while still
        looking like a configured value."""
        assert isinstance(LLM_REQUEST_TIMEOUT_SECONDS, int | float)
        assert LLM_REQUEST_TIMEOUT_SECONDS > 0

    def test_the_bound_stays_inside_the_client_deadline(self):
        """Nested deadlines: the inner one must fire first, or the client
        cancels a call this server still believes is live and is still paying a
        provider for.

        60s is `mcp_mind_client.gd::DEADLINE_SECONDS_DECIDE_ACTION` from
        [NPC-1682]. It is duplicated here rather than imported because it lives
        in the other repo; this assertion exists so that raising the timeout
        past it fails loudly instead of silently re-opening the race.
        """
        client_deadline_seconds = 60.0
        assert LLM_REQUEST_TIMEOUT_SECONDS < client_deadline_seconds

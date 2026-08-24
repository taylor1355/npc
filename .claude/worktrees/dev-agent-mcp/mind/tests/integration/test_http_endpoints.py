"""Integration tests for MCP server HTTP endpoints"""

import logging
import time

import pytest
from starlette.testclient import TestClient

from mind.interfaces.mcp.main import LOG_HANDLER, create_starlette_app
from mind.interfaces.mcp.server import MCPServer


@pytest.fixture
def test_client():
    """Create a test client for the Starlette app"""
    # Clear logs before creating new app to avoid accumulation
    LOG_HANDLER.logs.clear()

    server = MCPServer("Test Server")
    mcp_server = server.mcp._mcp_server
    app = create_starlette_app(mcp_server, debug=True)
    return TestClient(app)


@pytest.fixture
def logger_with_handler():
    """Create a test logger with the in-memory handler attached"""
    test_logger = logging.getLogger("test_logs")
    test_logger.addHandler(LOG_HANDLER)
    test_logger.setLevel(logging.DEBUG)
    yield test_logger
    # Cleanup: clear logs after test
    LOG_HANDLER.logs.clear()


class TestHealthEndpoint:
    """Tests for /health endpoint"""

    def test_health_returns_200(self, test_client):
        """Should return 200 OK"""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, test_client):
        """Should return JSON response"""
        response = test_client.get("/health")
        data = response.json()
        assert isinstance(data, dict)

    def test_health_contains_status(self, test_client):
        """Should include status field"""
        response = test_client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_contains_version(self, test_client):
        """Should include version field"""
        response = test_client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "1.0"

    def test_health_contains_uptime(self, test_client):
        """Should include uptime_seconds field"""
        response = test_client.get("/health")
        data = response.json()
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    def test_health_rejects_post(self, test_client):
        """Should reject POST requests"""
        response = test_client.post("/health")
        assert response.status_code == 405  # Method Not Allowed


class TestShutdownEndpoint:
    """Tests for /shutdown endpoint"""

    def test_shutdown_returns_200(self, test_client):
        """Should return 200 OK"""
        response = test_client.post("/shutdown")
        assert response.status_code == 200

    def test_shutdown_returns_json(self, test_client):
        """Should return JSON response"""
        response = test_client.post("/shutdown")
        data = response.json()
        assert isinstance(data, dict)

    def test_shutdown_contains_status(self, test_client):
        """Should include status field with shutdown message"""
        response = test_client.post("/shutdown")
        data = response.json()
        assert "status" in data
        assert data["status"] == "shutting down"

    def test_shutdown_rejects_get(self, test_client):
        """Should reject GET requests"""
        response = test_client.get("/shutdown")
        assert response.status_code == 405  # Method Not Allowed


class TestLogsEndpoint:
    """Tests for /logs endpoint"""

    def test_logs_returns_200(self, test_client):
        """Should return 200 OK"""
        response = test_client.get("/logs")
        assert response.status_code == 200

    def test_logs_returns_json(self, test_client):
        """Should return JSON response"""
        response = test_client.get("/logs")
        data = response.json()
        assert isinstance(data, dict)
        assert "logs" in data
        assert isinstance(data["logs"], list)

    def test_logs_empty_by_default(self, test_client):
        """Should return empty list when no logs exist"""
        LOG_HANDLER.logs.clear()
        response = test_client.get("/logs")
        data = response.json()
        assert data["logs"] == []

    def test_logs_contains_logged_messages(self, test_client, logger_with_handler):
        """Should return logged messages"""
        LOG_HANDLER.logs.clear()

        # Log some test messages
        logger_with_handler.info("Test message 1")
        logger_with_handler.warning("Test message 2")
        logger_with_handler.error("Test message 3")

        response = test_client.get("/logs")
        data = response.json()

        assert len(data["logs"]) == 3
        # Should be newest-first
        assert "Test message 3" in data["logs"][0]["message"]
        assert "Test message 2" in data["logs"][1]["message"]
        assert "Test message 1" in data["logs"][2]["message"]

    def test_logs_include_level(self, test_client, logger_with_handler):
        """Should include log level in response"""
        LOG_HANDLER.logs.clear()

        logger_with_handler.debug("Debug message")
        logger_with_handler.info("Info message")
        logger_with_handler.warning("Warning message")
        logger_with_handler.error("Error message")

        response = test_client.get("/logs")
        data = response.json()

        levels = [log["level"] for log in data["logs"]]
        assert "DEBUG" in levels
        assert "INFO" in levels
        assert "WARNING" in levels
        assert "ERROR" in levels

    def test_logs_include_timestamp(self, test_client, logger_with_handler):
        """Should include timestamp in each log entry"""
        LOG_HANDLER.logs.clear()

        before = time.time()
        logger_with_handler.info("Test message")
        after = time.time()

        response = test_client.get("/logs")
        data = response.json()

        assert len(data["logs"]) == 1
        timestamp = data["logs"][0]["timestamp"]
        assert isinstance(timestamp, (int, float))
        assert before <= timestamp <= after

    def test_logs_since_parameter(self, test_client, logger_with_handler):
        """Should filter logs by timestamp using since parameter"""
        LOG_HANDLER.logs.clear()

        logger_with_handler.info("Message 1")
        time.sleep(0.01)  # Small delay
        checkpoint = time.time()
        time.sleep(0.01)
        logger_with_handler.info("Message 2")

        response = test_client.get(f"/logs?since={checkpoint}")
        data = response.json()

        assert len(data["logs"]) == 1
        assert "Message 2" in data["logs"][0]["message"]

    def test_logs_limit_parameter(self, test_client, logger_with_handler):
        """Should limit number of returned logs"""
        LOG_HANDLER.logs.clear()

        for i in range(10):
            logger_with_handler.info(f"Message {i}")

        response = test_client.get("/logs?limit=5")
        data = response.json()

        assert len(data["logs"]) == 5
        # Should get the 5 newest (9, 8, 7, 6, 5)
        assert "Message 9" in data["logs"][0]["message"]
        assert "Message 5" in data["logs"][4]["message"]

    def test_logs_default_limit(self, test_client, logger_with_handler):
        """Should use default limit of 100"""
        LOG_HANDLER.logs.clear()

        for i in range(150):
            logger_with_handler.info(f"Message {i}")

        response = test_client.get("/logs")
        data = response.json()

        assert len(data["logs"]) == 100

    def test_logs_invalid_since_parameter(self, test_client):
        """Should return 400 for invalid since parameter"""
        response = test_client.get("/logs?since=invalid")
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_logs_invalid_limit_parameter(self, test_client):
        """Should return 400 for invalid limit parameter"""
        response = test_client.get("/logs?limit=invalid")
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_logs_combined_since_and_limit(self, test_client, logger_with_handler):
        """Should handle both since and limit parameters together"""
        LOG_HANDLER.logs.clear()

        for i in range(10):
            logger_with_handler.info(f"Message {i}")
            time.sleep(0.01)

        # Get timestamp after message 5
        checkpoint = time.time() - 0.04  # Rough estimate

        response = test_client.get(f"/logs?since={checkpoint}&limit=2")
        data = response.json()

        # Should get at most 2 of the newest logs after checkpoint
        assert len(data["logs"]) <= 2

    def test_logs_rejects_post(self, test_client):
        """Should reject POST requests"""
        response = test_client.post("/logs")
        assert response.status_code == 405  # Method Not Allowed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

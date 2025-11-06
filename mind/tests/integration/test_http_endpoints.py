"""Integration tests for MCP server HTTP endpoints"""

import pytest
from starlette.testclient import TestClient

from mind.interfaces.mcp.main import create_starlette_app
from mind.interfaces.mcp.server import MCPServer


@pytest.fixture
def test_client():
    """Create a test client for the Starlette app"""
    server = MCPServer("Test Server")
    mcp_server = server.mcp._mcp_server
    app = create_starlette_app(mcp_server, debug=True)
    return TestClient(app)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

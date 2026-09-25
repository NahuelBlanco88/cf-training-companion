"""The package must not advertise an unverified production MCP URL."""

import json
from pathlib import Path

import pytest

from mcp_bridge.render_plugin_config import render


def test_portable_package_and_real_url_gate(tmp_path):
    root = Path(__file__).resolve().parent.parent / "plugins" / "cf-training-companion"
    manifest = json.loads((root / "plugin.json").read_text())
    assert manifest["name"] == "cf-training-companion"
    assert (root / "skills" / "training-log" / "SKILL.md").read_text().startswith("---\nname: training-log\n")
    assert not (root / "mcp.json").exists()

    destination = tmp_path / "mcp.json"
    with pytest.raises(ValueError, match="actual deployed HTTPS endpoint"):
        render("http://127.0.0.1:8080/mcp", destination)
    assert not destination.exists()
    render("https://cf-training-mcp.example.test/mcp", destination)
    config = json.loads(destination.read_text())
    assert config["mcpServers"]["cf-training-companion"]["type"] == "streamable-http"
    assert config["mcpServers"]["cf-training-companion"]["url"].endswith("/mcp")

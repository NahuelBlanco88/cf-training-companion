"""Fill the portable plugin's MCP URL only after a real HTTPS endpoint exists."""

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse


def render(public_url: str, destination: Path) -> None:
    address = urlparse(public_url)
    if (address.scheme != "https" or not address.hostname or address.username
            or address.password or address.query or address.fragment or address.path != "/mcp"):
        raise ValueError("Provide the actual deployed HTTPS endpoint ending exactly in /mcp")
    configuration = {
        "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
        "mcpServers": {
            "cf-training-companion": {"type": "streamable-http", "url": public_url}
        },
    }
    destination.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    render(arguments.public_url, arguments.output)

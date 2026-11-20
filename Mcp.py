import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("MyTools")

MAX_FILE_BYTES = 1_000_000
MAX_URL_BYTES = 200_000


def safe_local_path(file_path: str) -> Path:
    root = os.getenv("MCP_ALLOWED_DIR", "").strip()
    if not root:
        raise PermissionError("read_file is disabled: set MCP_ALLOWED_DIR")

    base = Path(root).resolve()
    candidate = Path(file_path)
    candidate = candidate if candidate.is_absolute() else base / candidate
    candidate = candidate.resolve()

    if not candidate.is_relative_to(base):
        raise PermissionError("read_file path is outside MCP_ALLOWED_DIR")
    return candidate


def allowed_hosts() -> set[str]:
    return {
        host.strip().lower()
        for host in os.getenv("MCP_ALLOWED_HOSTS", "").split(",")
        if host.strip()
    }


@mcp.tool()
def search_recent_data(query: str) -> str:
    """Search the web for recent data based on a query."""
    search_tool = TavilySearchResults(max_results=2)
    results = search_tool.invoke(query)
    return str(results)


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b. Raises error if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@mcp.tool()
def read_file(file_path: str) -> str:
    """Read a local file, restricted to MCP_ALLOWED_DIR."""
    path = safe_local_path(file_path)
    with path.open(encoding="utf-8") as file:
        return file.read(MAX_FILE_BYTES)


@mcp.tool()
def get_datetime() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@mcp.tool()
def read_url(url: str) -> str:
    """Fetch a URL, restricted to hosts in MCP_ALLOWED_HOSTS."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise PermissionError("read_url requires an http(s) URL")
    if parsed.hostname.lower() not in allowed_hosts():
        raise PermissionError("read_url host is not in MCP_ALLOWED_HOSTS")

    response = requests.get(url, timeout=10, allow_redirects=False)
    if 300 <= response.status_code < 400:
        raise PermissionError("read_url does not follow redirects")
    response.raise_for_status()
    return response.text[:MAX_URL_BYTES]


if __name__ == "__main__":
    mcp.settings.port = int(os.getenv("MCP_PORT", "8000"))
    mcp.run(transport="streamable-http")

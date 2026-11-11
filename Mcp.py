import sys
from mcp.server.fastmcp import FastMCP
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from datetime import datetime
import requests

load_dotenv()

mcp = FastMCP("MyTools")


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
    """Read the content of a local file."""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return str(e)

@mcp.tool()
def get_datetime() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@mcp.tool()
def read_url(url: str) -> str:
    """Read the content of a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return str(e)

if __name__ == "__main__":
    mcp.settings.port = 8000
    mcp.run(transport="streamable-http")
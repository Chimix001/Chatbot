# MyChatbot — AI-Powered Conversational Assistant

An intelligent, tool-enabled conversational AI assistant built with **LangGraph, LangChain, FastAPI, and the Model Context Protocol (MCP)**.

MyChatbot combines a Groq-hosted Large Language Model (LLM) with external tools, persistent memory, and graph-based agent orchestration to deliver contextual and extensible AI conversations.

The system can retrieve recent information from the web, perform mathematical calculations, read local files, access URLs, and remember relevant information across conversations.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Memory Architecture](#memory-architecture)
- [Available MCP Tools](#available-mcp-tools)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Example Interactions](#example-interactions)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)
- [Future Improvements](#future-improvements)

---

## Overview

MyChatbot is an experimental agentic AI application designed to demonstrate how modern AI systems can combine language models, external tools, workflow orchestration, and persistent memory.

Unlike a basic chatbot that only generates text, MyChatbot can determine when external tools are needed, execute those tools through an MCP server, and incorporate their results into its responses.

The application uses LangGraph to manage the conversation lifecycle, MCP to expose external capabilities, and a hybrid memory architecture to maintain both conversational history and long-term user information.

## Features

- **Conversational AI:** Powered by Groq-hosted LLMs through LangChain.
- **Agentic tool calling:** Automatically determines when tools are required to answer user requests.
- **MCP integration:** Exposes external capabilities through the Model Context Protocol over Streamable HTTP.
- **Web search:** Retrieves recent information using Tavily Search.
- **Mathematical operations:** Supports addition, subtraction, multiplication, and division.
- **File and URL reading:** Retrieves content from local files and web URLs.
- **Long-term memory:** Stores user-specific information using PostgreSQL and ChromaDB.
- **Short-term memory:** Maintains conversation history using SQLite checkpoints.
- **Graph-based orchestration:** Uses LangGraph to coordinate memory retrieval, LLM execution, tool calling, and memory persistence.
- **REST API:** Provides an HTTP interface through FastAPI.
- **Interactive CLI:** Supports local conversational testing.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | LLM integration and tool binding |
| LangGraph | Agent workflow orchestration |
| Groq | Hosted LLM inference |
| FastAPI | REST API development |
| MCP | External tool integration |
| Tavily | Web search |
| PostgreSQL | Persistent user memory storage |
| ChromaDB | Vector storage and semantic retrieval |
| Sentence Transformers | Text embeddings |
| SQLite | Conversation checkpoint persistence |
| Pydantic | Data validation |
| Uvicorn | ASGI application server |

---

## System Architecture

MyChatbot follows a modular architecture in which the API layer, agent orchestration, memory components, and MCP tools operate independently.

### Architecture Diagram

```mermaid
flowchart TD
    A["User"] --> B["FastAPI - main.py"]
    B --> C["LangGraph Agent - Mychatbot.py"]

    subgraph Agent["LangGraph Orchestration"]
        C --> D["Load Memory"]
        D --> E["Chatbot / LLM"]
        E --> F{"Tool Calls?"}
        F -->|Yes| G["MCP Tool Node"]
        G --> E
        F -->|No| H["Token Counter"]
        H --> I["Save Memory"]
    end

    I --> J["Response to User"]

    E <-->|Inference| K["Groq LLM"]

    G <-->|Streamable HTTP| L["MCP Server"]

    L --> M["Tavily Search"]
    L --> N["Mathematical Tools"]
    L --> O["File Reader"]
    L --> P["URL Reader"]
    L --> Q["Date and Time"]

    D <-->|Semantic Retrieval| R["ChromaDB"]
    I <-->|Persistent Storage| S["PostgreSQL"]

    C <-->|Checkpoints| T["SQLite"]
```

### Architectural Components

**1. FastAPI Layer**

Provides the HTTP interface through which external applications can interact with the chatbot.

**2. LangGraph Orchestrator**

Coordinates the chatbot's execution flow, including memory retrieval, LLM invocation, conditional tool execution, and memory persistence.

**3. LLM Layer**

Uses Groq-hosted language models to understand user requests, generate responses, and decide when tools should be called.

**4. MCP Server**

Exposes external capabilities as standardized tools that the chatbot can discover and execute through the Model Context Protocol.

**5. Memory Layer**

Combines PostgreSQL, ChromaDB, and SQLite to support persistent user information and conversation continuity.

---

## Project Structure

```text
MyChatbot/
│
├── main.py                  # FastAPI application and REST endpoints
├── Mychatbot.py             # LangGraph workflow and LLM integration
├── memory.py                # Memory management and retrieval nodes
├── Mcp.py                   # MCP server and tool definitions
│
├── .env                     # API keys and environment variables
├── .env.example             # Environment variable template
├── .gitignore               # Files excluded from Git
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
│
├── checkpoints.sqlite       # SQLite conversation checkpoints
└── chroma_db/               # ChromaDB vector storage
```

> **Note:** `checkpoints.sqlite` and `chroma_db/` are generated at runtime and should not be committed to Git.

> **Important:** Avoid naming your MCP server file `mcp.py`, as this can shadow the installed `mcp` Python SDK and cause import errors.

---

## How It Works

The chatbot follows a graph-based execution workflow.

1. **User Input:** The user submits a message through the CLI or FastAPI endpoint.
2. **Memory Retrieval:** The memory node searches for relevant information associated with the user's identity.
3. **LLM Processing:** The retrieved context and conversation history are passed to the Groq-hosted LLM.
4. **Tool Decision:** The LLM determines whether the request requires an external tool.
5. **Tool Execution:** If necessary, LangGraph routes the request to the MCP ToolNode.
6. **Response Generation:** The tool results are returned to the LLM for further processing.
7. **Token Counting:** The application calculates a basic token estimate for the conversation.
8. **Memory Persistence:** Relevant information is passed to the memory-saving component.
9. **Response Delivery:** The final response is returned to the user.

The workflow supports iterative tool execution until the LLM produces a final response.

---

## Memory Architecture

MyChatbot implements two complementary memory mechanisms.

### 1. Long-Term Memory

Long-term memory allows the chatbot to retain relevant user information across separate conversations.

It uses:

- **PostgreSQL:** Persistent storage for user memory records.
- **ChromaDB:** Vector indexing for semantic similarity search.
- **Sentence Transformers:** Generates text embeddings for semantic retrieval.

Memory is associated with a `user_id`, allowing information to be retrieved across different conversation threads belonging to the same user.

### 2. Short-Term Memory

Short-term memory maintains conversation history within a particular conversation thread.

It uses:

- **SQLite:** Local checkpoint storage.
- **AsyncSqliteSaver:** Asynchronous checkpoint persistence.
- **thread_id:** Identifies and isolates individual conversation threads.

### Memory Comparison

| Feature | Long-Term Memory | Short-Term Memory |
|---|---|---|
| Purpose | Retain user information | Maintain conversation context |
| Storage | PostgreSQL + ChromaDB | SQLite |
| Identifier | `user_id` | `thread_id` |
| Retrieval | Semantic similarity | Conversation checkpoints |
| Persistence | Across conversations | Across sessions in the same thread |

---

## Available MCP Tools

The MCP server exposes tools that extend the chatbot's capabilities.

| Tool | Description |
|---|---|
| `search_recent_data` | Searches the web using Tavily |
| `add` | Adds two numbers |
| `subtract` | Subtracts one number from another |
| `multiply` | Multiplies two numbers |
| `divide` | Divides two numbers with zero-division protection |
| `read_file` | Reads the content of a local file |
| `get_datetime` | Returns the current date and time |
| `read_url` | Retrieves content from a URL |

The tools are exposed through an MCP server using the Streamable HTTP transport.

---

## Getting Started

### Prerequisites

Before installing the project, ensure you have:

- Python 3.11 or later.
- PostgreSQL installed and running.
- A Groq API key.
- A Tavily API key.
- Git installed.
- Access to the required Python packages.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd MyChatbot
```

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project's root directory.

You can use `.env.example` as a template.

---

## Configuration

The application requires API credentials and database connection details.

Example `.env.example`:

```env
# Groq
GROQ_API_KEY=your_groq_api_key

# Tavily
TAVILY_API_KEY=your_tavily_api_key

# PostgreSQL
DB_URL=postgresql://username:password@localhost:5432/mychatbot

# MCP Server
MCP_SERVER_URL=http://localhost:8000/mcp
```

Replace the placeholder values with your actual credentials and connection details.

Never commit your actual `.env` file or expose API keys in your source code.

---

## Running the Application

The MCP server and chatbot application should run in separate terminals.

### 1. Start the MCP Server

```powershell
python my_mcp_server.py
```

The MCP server should be available at:

```text
http://localhost:8000/mcp
```

### 2. Start the Chatbot CLI

Open another terminal:

```powershell
python Mychatbot.py
```

You can now interact with the chatbot directly through the terminal.

To exit the conversation, enter:

```text
exit
```

### 3. Start the FastAPI Application

To expose the chatbot through the REST API:

```powershell
uvicorn main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI interactive documentation:

```text
http://localhost:8000/docs
```

> **Note:** If the MCP server and FastAPI application run simultaneously on the same machine, configure them to use different ports. For example, keep the MCP server on port `8000` and run FastAPI on port `8001`.

---

## API Reference

MyChatbot exposes a REST API through FastAPI.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Application health or welcome endpoint |
| `POST` | `/ask` | Submit a question to the chatbot |

### Example Request

```http
POST /ask
Content-Type: application/json
```

Example JSON payload (adapt the fields to the request model defined in `main.py`):

```json
{
    "message": "What is artificial intelligence?",
    "user_id": "user_123",
    "thread_id": "conversation_001"
}
```

The endpoint processes the request through the LangGraph workflow and returns the chatbot's response.

---

## Example Interactions

Here are some example prompts to test the chatbot.

**General Conversation**

```text
Hello! What can you help me with?
```

**Web Search**

```text
Search for the latest developments in artificial intelligence.
```

**Mathematical Operations**

```text
What is 250 multiplied by 45?
```

**Date and Time**

```text
What is the current date and time?
```

**File Reading**

```text
Read the contents of my notes.txt file.
```

**URL Reading**

```text
Retrieve and summarize the content of https://example.com.
```

**Memory Testing**

```text
My name is Alex, and I am learning Python.
```

In a subsequent conversation using the same `user_id`, test whether the chatbot can retrieve the previously stored information.

---

## Troubleshooting

### 1. MCP ImportError

If you encounter:

```text
ImportError: cannot import name 'MCPServer' from 'mcp.server'
```

Use the current FastMCP interface:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MyTools")
```

Avoid importing the nonexistent `MCPServer` class.

### 2. MCP SDK Compatibility

The project uses the MCP Python SDK 1.x API.

For compatibility with the LangChain MCP adapter version used by the project, avoid unintentionally installing MCP SDK 2.x.

Example dependency constraint:

```text
mcp>=1.24,<2
```

Keep your MCP SDK and `langchain-mcp-adapters` versions compatible.

### 3. Groq Model Errors

If Groq returns a model decommissioning or unavailable-model error, check the currently supported models in the Groq documentation and update the model configured in `Mychatbot.py`.

### 4. PostgreSQL Connection Errors

Verify that:

- PostgreSQL is running.
- The database exists.
- The connection URL is correct.
- The configured database user has the required permissions.

### 5. MCP Connection Errors

Ensure that:

- The MCP server is running.
- The configured URL points to the correct endpoint.
- The Streamable HTTP transport is configured consistently on both the server and client.
- The selected port is available.

### 6. Missing API Keys

Verify that:

- The `.env` file is in the project root.
- `load_dotenv()` is called.
- The environment variable names match those used in the Python code.

---

## Security Considerations

MyChatbot interacts with external services, local files, and databases. Appropriate security controls are important.

- **API credentials:** Store API keys in environment variables.
- **File access:** Restrict file-reading tools to approved directories.
- **URL access:** Validate URLs and restrict access to private and internal network addresses to prevent SSRF.
- **Database security:** Use appropriate database permissions and secure connection credentials.
- **User isolation:** Ensure memory retrieval and persistence are scoped to the authenticated user.
- **MCP transport:** Use appropriate authentication and access controls when exposing the server beyond localhost.
- **Git security:** Exclude `.env`, virtual environments, databases, and generated vector stores from version control.

---

## Future Improvements

Potential areas for extending the application include:

- Automatic fallback between Groq and Gemini.
- Streaming responses through FastAPI.
- Improved memory management and relevance filtering.
- User authentication and authorization.
- Conversation management and deletion.
- Better token usage and cost tracking.
- Additional MCP tools and integrations.
- Docker-based deployment.
- Automated testing and evaluation.

---

## Contributing

Contributions, suggestions, and improvements are welcome.

To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Test your implementation.
5. Submit a pull request.

---



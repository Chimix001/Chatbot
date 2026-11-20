from __future__ import annotations

import asyncio
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.store.postgres import PostgresStore

from memory import (
    CHECKPOINT_DB,
    extract_text,
    memory_save_node,
    memory_search_node,
    require_db_url,
)

load_dotenv()


class MychatBot(TypedDict):
    messages: Annotated[list, add_messages]
    token_count: int


llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

mcp_client = MultiServerMCPClient({
    "my-tools": {
        "url": os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp"),
        "transport": "streamable_http",
    }
})


def tools_router(state: MychatBot):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tool_node"
    return "save_memory"


def ordered_messages(messages: list):
    """Keep system messages first, regardless of when memory retrieval ran."""
    system = [m for m in messages if isinstance(m, SystemMessage)]
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    return system + rest


def usage_tokens(message) -> int:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("total_tokens") or 0)


async def build_graph() -> StateGraph:
    mcp_tools = await mcp_client.get_tools()
    llm_with_tools = llm.bind_tools(mcp_tools)

    async def chatbot(state: MychatBot):
        response = await llm_with_tools.ainvoke(ordered_messages(state["messages"]))
        return {
            "messages": [response],
            "token_count": (state.get("token_count") or 0) + usage_tokens(response),
        }

    graph = StateGraph(MychatBot)
    graph.add_node("load_memory", memory_search_node)
    graph.add_node("chatbot", chatbot)
    graph.add_node("tool_node", ToolNode(mcp_tools))
    graph.add_node("save_memory", memory_save_node)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "chatbot")
    graph.add_conditional_edges("chatbot", tools_router)
    graph.add_edge("tool_node", "chatbot")
    graph.add_edge("save_memory", END)
    return graph


async def main():
    with PostgresStore.from_conn_string(require_db_url()) as store:
        store.setup()

        graph = await build_graph()

        async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
            app = graph.compile(checkpointer=checkpointer, store=store)
            config = {
                "configurable": {
                    "thread_id": "cli-user_123",
                    "user_id": os.getenv("CLI_USER_ID", "user_123"),
                }
            }

            while True:
                user_input = input("user: ").strip()
                if user_input in ("exit", "end"):
                    break
                if not user_input:
                    continue

                result = await app.ainvoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )
                print("AI:", extract_text(result["messages"][-1].content))
                print("Token count:", result.get("token_count", "N/A"))


if __name__ == "__main__":
    asyncio.run(main())

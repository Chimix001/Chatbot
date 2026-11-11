from __future__ import annotations

import asyncio
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.store.postgres import PostgresStore

from memory import DB_URL, memory_save_node, memory_search_node

load_dotenv()


class MychatBot(TypedDict):
    messages: Annotated[list, add_messages]
    token_count: int


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

mcp_client = MultiServerMCPClient({
    "my-tools": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http",
    }
})


def tools_router(state: MychatBot):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tool_node"
    return "get_token_counter"


def token_counter(state: MychatBot):
    tokens = str(state["messages"]).split()
    return {"token_count": len(tokens)}


async def build_graph() -> StateGraph:
    mcp_tools = await mcp_client.get_tools()
    llm_with_tools = llm.bind_tools(mcp_tools)

    def chatbot(state: MychatBot):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph = StateGraph(MychatBot)
    graph.add_node("load_memory", memory_search_node)
    graph.add_node("chatbot", chatbot)
    graph.add_node("tool_node", ToolNode(mcp_tools))
    graph.add_node("get_token_counter", token_counter)
    graph.add_node("save_memory", memory_save_node)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "chatbot")
    graph.add_conditional_edges("chatbot", tools_router)
    graph.add_edge("tool_node", "chatbot")
    graph.add_edge("get_token_counter", "save_memory")
    graph.add_edge("save_memory", END)
    return graph

async def main():
    with PostgresStore.from_conn_string(DB_URL) as store:
        store.setup()

        graph = await build_graph()

        # AsyncSqliteSaver MUST be opened as an async context manager
        async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
            app = graph.compile(checkpointer=checkpointer, store=store)

            config = {"configurable": {"thread_id": 1, "user_id": "user_123"}}

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

                ai_content = result["messages"][-1].content
                if isinstance(ai_content, list):
                    ai_text = "".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in ai_content
                    )
                else:
                    ai_text = ai_content

                print("AI:", ai_text)
                print("Token count:", result.get("token_count", "N/A"))


if __name__ == "__main__":
    asyncio.run(main())
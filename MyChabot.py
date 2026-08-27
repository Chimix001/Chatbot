from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
import sqlite3
import os

load_dotenv()

# storing the memory in sqlite
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)


class MychatBot(TypedDict):
    messages: Annotated[list, add_messages]
    token_count: int


# calling the tools
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

# calling the llm
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Binding tool with LLM model
llm_with_tool = llm.bind_tools(tools=tools)


def chatbot(state: MychatBot):
    return {
        "messages": [llm_with_tool.invoke(state["messages"])]
    }


def tools_router(state: MychatBot):
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return "get_token_counter"


tool_node = ToolNode(tools=tools)


def token_counter(state: MychatBot):
    tokens = str(state["messages"]).split()  # Split the text into words
    count_no = len(tokens)
    return {"token_count": count_no}


# creating the graph pipeline
graph = StateGraph(MychatBot)

graph.add_node("chatbot", chatbot)
graph.add_node("get_token_counter", token_counter)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("chatbot")
graph.add_conditional_edges("chatbot", tools_router)
graph.add_edge("tool_node", "chatbot")
graph.add_edge("get_token_counter", END)

app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": 1}}

while True:
    user_input = input("user: ")
    if user_input in ["exit", "end"]:
        break
    else:
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        }, config=config)

        ai_content = result["messages"][-1].content

        if isinstance(ai_content, list):
            ai_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in ai_content
            )
        else:
            ai_text = ai_content

        print("AI: " + ai_text)
        print("Token count:", result.get("token_count", "N/A"))
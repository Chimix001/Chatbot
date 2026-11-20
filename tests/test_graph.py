import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import Mychatbot


def test_tools_router():
    assert Mychatbot.tools_router({
        "messages": [AIMessage(content="", tool_calls=[{"name": "add", "args": {}, "id": "1"}])]
    }) == "tool_node"
    assert Mychatbot.tools_router({"messages": [AIMessage(content="done")]}) == "save_memory"


def test_ordered_messages_puts_system_first():
    messages = [HumanMessage(content="hi"), SystemMessage(content="memories")]
    ordered = Mychatbot.ordered_messages(messages)
    assert isinstance(ordered[0], SystemMessage)
    assert isinstance(ordered[1], HumanMessage)


def test_usage_tokens():
    usage = {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
    assert Mychatbot.usage_tokens(AIMessage(content="x", usage_metadata=usage)) == 7
    assert Mychatbot.usage_tokens(AIMessage(content="x")) == 0


@pytest.mark.asyncio
async def test_build_graph_wiring(monkeypatch):
    class FakeLLM:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            return AIMessage(content="ok", usage_metadata={"total_tokens": 2})

    class FakeClient:
        async def get_tools(self):
            return []

    monkeypatch.setattr(Mychatbot, "llm", FakeLLM())
    monkeypatch.setattr(Mychatbot, "mcp_client", FakeClient())

    graph = await Mychatbot.build_graph()
    assert set(graph.nodes) == {"load_memory", "chatbot", "tool_node", "save_memory"}


@pytest.mark.asyncio
async def test_graph_runs_with_fakes(monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    import memory

    class FakeLLM:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            assert isinstance(messages[0], SystemMessage) or len(messages) == 1
            return AIMessage(content="ok", usage_metadata={
                "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
            })

    class FakeClient:
        async def get_tools(self):
            return []

    monkeypatch.setattr(Mychatbot, "llm", FakeLLM())
    monkeypatch.setattr(Mychatbot, "mcp_client", FakeClient())
    monkeypatch.setattr(memory.Memorysaver, "search", lambda self, *a, **k: [])
    monkeypatch.setattr(memory.Memorysaver, "save", lambda self, *a, **k: "id")

    graph = await Mychatbot.build_graph()
    app = graph.compile(checkpointer=InMemorySaver(), store=InMemoryStore())
    result = await app.ainvoke(
        {"messages": [HumanMessage(content="hello there friend")]},
        config={"configurable": {"thread_id": "t", "user_id": "u"}},
    )
    assert result["messages"][-1].content == "ok"
    assert result["token_count"] == 2

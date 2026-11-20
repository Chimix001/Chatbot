import httpx
import pytest
from langchain_core.messages import AIMessage

import main


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    main.app.dependency_overrides.clear()


class FakeGraph:
    def __init__(self, error=None):
        self.config = None
        self.error = error

    async def ainvoke(self, state, config):
        self.config = config
        if self.error:
            raise self.error
        return {
            "messages": [AIMessage(content=[{"type": "text", "text": "hello"}])],
            "token_count": 5,
        }


@pytest.mark.asyncio
async def test_ask_uses_token_identity_and_scoped_thread(monkeypatch):
    monkeypatch.setenv("AUTH_TOKENS", '{"abc": "alice"}')
    fake = FakeGraph()
    main.app.state.graph = fake
    main.app.dependency_overrides[main.get_user_id] = lambda: "alice"

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/ask", json={"question": "hi there", "thread_id": "t1"}
        )

    assert response.status_code == 200
    assert response.json() == {"response": "hello", "token_count": 5}
    assert fake.config["configurable"] == {"thread_id": "alice:t1", "user_id": "alice"}


@pytest.mark.asyncio
async def test_ask_requires_auth(monkeypatch):
    monkeypatch.setenv("AUTH_TOKENS", '{"abc": "alice"}')
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/ask", json={"question": "hi"})

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_ask_handles_graph_failure(monkeypatch):
    monkeypatch.setenv("AUTH_TOKENS", '{"abc": "alice"}')
    main.app.state.graph = FakeGraph(error=RuntimeError("MCP down"))
    main.app.dependency_overrides[main.get_user_id] = lambda: "alice"

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/ask", json={"question": "hi"})

    assert response.status_code == 502

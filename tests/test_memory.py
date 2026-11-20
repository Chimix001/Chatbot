from langchain_core.messages import HumanMessage, SystemMessage

import memory


class Item:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class FakeStore:
    def __init__(self):
        self.data = {}

    def put(self, namespace, key, value):
        self.data.setdefault(namespace, {})[key] = value

    def get(self, namespace, key):
        value = self.data.get(namespace, {}).get(key)
        return Item(key, value) if value is not None else None

    def search(self, namespace, limit=10):
        items = list(self.data.get(namespace, {}).items())[:limit]
        return [Item(key, value) for key, value in items]


VECTORS = {
    "My name is Sam": [1.0, 0.0],
    "Random unrelated text": [0.0, 1.0],
    "what is my name": [1.0, 0.0],
}


def fake_embed(text):
    return VECTORS.get(text, [0.5, 0.5])


def test_cosine_similarity():
    assert memory.cosine_similarity([1, 0], [1, 0]) == 1.0
    assert memory.cosine_similarity([1, 0], [0, 1]) == 0.0
    assert memory.cosine_similarity([], [1]) == 0.0
    assert memory.cosine_similarity([1], [1, 2]) == 0.0


def test_save_stores_embedding_and_dedupes(monkeypatch):
    monkeypatch.setattr(memory, "embed", fake_embed)
    store = FakeStore()
    saver = memory.Memorysaver(store)

    first = saver.save("u1", "My name is Sam", {"nested": {"x": 1}})
    saved = store.get(("memories", "u1"), first).value
    assert saved["text"] == "My name is Sam"
    assert saved["metadata"] == {"nested": {"x": 1}}
    assert saved["embedding"] == [1.0, 0.0]

    assert saver.save("u1", "My name is Sam") == first
    assert len(store.data[("memories", "u1")]) == 1


def test_search_applies_relevance_threshold(monkeypatch):
    monkeypatch.setattr(memory, "embed", fake_embed)
    store = FakeStore()
    store.put(("memories", "u1"), "high", {
        "text": "My name is Sam", "metadata": {}, "embedding": [1.0, 0.0],
    })
    store.put(("memories", "u1"), "low", {
        "text": "Random unrelated text", "metadata": {}, "embedding": [0.0, 1.0],
    })

    results = memory.Memorysaver(store).search("u1", "what is my name")
    assert [r["memory_id"] for r in results] == ["high"]
    assert results[0]["score"] == 1.0
    assert "embedding" not in results[0]


def test_memory_save_node_skips_short_messages(monkeypatch):
    monkeypatch.setattr(memory, "embed", fake_embed)
    store = FakeStore()
    config = {"configurable": {"user_id": "u1"}}

    memory.memory_save_node({"messages": [HumanMessage(content="hi")]}, config, store=store)
    assert store.data == {}

    memory.memory_save_node(
        {"messages": [HumanMessage(content="my favorite color is blue")]}, config, store=store
    )
    assert len(store.data[("memories", "u1")]) == 1


def test_memory_search_node_injects_system_message(monkeypatch):
    monkeypatch.setattr(memory.Memorysaver, "search", lambda self, *a, **k: [
        {"memory_id": "1", "text": "Sam likes tea", "score": 0.9}
    ])
    config = {"configurable": {"user_id": "u1"}}
    state = {"messages": [HumanMessage(content="what do I like?")]}

    result = memory.memory_search_node(state, config, store=FakeStore())
    assert isinstance(result["messages"][0], SystemMessage)
    assert "Sam likes tea" in result["messages"][0].content

    monkeypatch.setattr(memory.Memorysaver, "search", lambda self, *a, **k: [])
    assert memory.memory_search_node(state, config, store=FakeStore()) == {}

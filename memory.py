from __future__ import annotations

import math
import os
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "")
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "checkpoints.sqlite")
MIN_RELEVANCE = float(os.getenv("MEMORY_MIN_SCORE", "0.3"))
SEARCH_LIMIT = int(os.getenv("MEMORY_SEARCH_LIMIT", "200"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

IGNORED_MESSAGES = {
    "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "yes",
    "no", "sure", "lol", "bye", "goodbye", "got it", "sounds good",
}

_embedder = None


def require_db_url() -> str:
    if not DB_URL:
        raise RuntimeError("DATABASE_URL is not set (see .env.example)")
    return DB_URL


def get_embedder():
    """Load the embedding model on first use so importing stays cheap."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def embed(text: str) -> list[float]:
    return get_embedder().encode(text).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_text(content) -> str:
    """Normalize str or list-of-parts content from chat messages."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def message_text(message) -> str:
    return extract_text(getattr(message, "content", message))


def is_worth_saving(text: str) -> bool:
    """Skip short acknowledgements that would only add retrieval noise."""
    normalized = " ".join(text.split()).lower()
    if normalized in IGNORED_MESSAGES:
        return False
    return len(normalized.split()) >= 3


class Memorysaver:
    """Long-term memory: Postgres stores text, metadata and embedding together."""

    def __init__(self, store: BaseStore):
        self.store = store

    def save(self, user_id: str, text: str, metadata: dict | None = None) -> str:
        name_space = ("memories", user_id)
        metadata = metadata or {}

        for item in self.store.search(name_space, limit=SEARCH_LIMIT):
            if item.value.get("text") == text:
                return str(item.key)

        memory_id = str(uuid.uuid4())
        self.store.put(name_space, memory_id, {
            "text": text,
            "metadata": metadata,
            "embedding": embed(text),
        })
        return memory_id

    def search(self, user_id: str, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = embed(query)
        name_space = ("memories", user_id)

        scored = [
            (cosine_similarity(query_embedding, item.value.get("embedding", [])), item)
            for item in self.store.search(name_space, limit=SEARCH_LIMIT)
        ]
        scored = [(score, item) for score, item in scored if score >= MIN_RELEVANCE]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        results = []
        for score, item in scored[:top_k]:
            value = {k: v for k, v in item.value.items() if k != "embedding"}
            results.append({"memory_id": str(item.key), "score": score, **value})
        return results


def memory_search_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    """Retrieve relevant long-term memories and inject them as a system message."""
    user_id = config["configurable"].get("user_id", "anonymous")
    query = message_text(state["messages"][-1])

    hits = Memorysaver(store).search(user_id, query, top_k=5)
    if not hits:
        return {}

    context = "\n".join(f"- {h['text']}" for h in hits)
    return {
        "messages": [
            SystemMessage(content=f"Relevant memories for this user:\n{context}")
        ]
    }


def memory_save_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    """Persist the most recent human message if it is worth remembering."""
    user_id = config["configurable"].get("user_id", "anonymous")
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    if last_human:
        text = message_text(last_human).strip()
        if text and is_worth_saving(text):
            Memorysaver(store).save(user_id, text)
    return {}

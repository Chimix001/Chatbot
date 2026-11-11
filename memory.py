from __future__ import annotations

import uuid

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.graph import MessagesState

DB_URL = "postgresql://postgres:postgres@localhost:5432/langgraph"

# Chroma collection owns the embedding function — never call .encode()/.tolist().
embedding_model = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="langgraph_collection",
    embedding_function=embedding_model,
    metadata={"hnsw:space": "cosine"},
)


class Memorysaver:
    """Long-term memory: Postgres is the source of truth, Chroma is the index."""

    def __init__(self, store: BaseStore):
        self.store = store

    def save(self, user_id: str, text: str, metadata: dict | None = None) -> str:
        memory_id = str(uuid.uuid4())
        name_space = ("memories", user_id)
        metadata = metadata or {}

        # 1) authoritative copy in Postgres
        self.store.put(name_space, memory_id, {"text": text, "metadata": metadata})

        # 2) searchable copy in Chroma (only primitive metadata values)
        chroma_metadata = {
            "user_id": user_id,
            **{k: v for k, v in metadata.items()
               if isinstance(v, (str, int, float, bool))},
        }
        collection.upsert(
            ids=[memory_id],
            documents=[text],            # Chroma embeds via embedding_model
            metadatas=[chroma_metadata],
        )
        return memory_id

    def search(self, user_id: str, query: str, top_k: int = 5) -> list[dict]:
        res = collection.query(
            query_texts=[query],         # Chroma embeds via embedding_model
            n_results=top_k,
            where={"user_id": user_id},
        )
        name_space = ("memories", user_id)
        results = []
        for mem_id, dist in zip(res["ids"][0], res["distances"][0]):
            item = self.store.get(name_space, mem_id)
            if item:
                results.append({"memory_id": mem_id, "score": 1 - dist, **item.value})
        return results


# ---- LangGraph nodes (module-level functions; `store` is injected) ----

def memory_search_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    """Retrieve relevant long-term memories and inject them as a system message."""
    user_id = config["configurable"].get("user_id", "anonymous")
    last = state["messages"][-1]
    query = last.content if hasattr(last, "content") else str(last)

    hits = Memorysaver(store).search(user_id, query, top_k=5)
    if not hits:
        return {}  # nothing to inject

    context = "\n".join(f"- {h['text']}" for h in hits)
    return {
        "messages": [
            SystemMessage(content=f"Relevant memories for this user:\n{context}")
        ]
    }


def memory_save_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    """Persist the most recent human message as a long-term memory."""
    user_id = config["configurable"].get("user_id", "anonymous")
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    if last_human and last_human.content:
        Memorysaver(store).save(user_id, last_human.content)
    return {}
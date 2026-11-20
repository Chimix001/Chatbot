import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.postgres import PostgresStore
from pydantic import BaseModel, Field

from Mychatbot import build_graph
from memory import CHECKPOINT_DB, extract_text, require_db_url

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


def load_auth_tokens() -> dict[str, str]:
    raw = os.getenv("AUTH_TOKENS", "").strip()
    if not raw:
        return {}
    try:
        tokens = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("AUTH_TOKENS must be a JSON object of token: user_id") from exc
    if not isinstance(tokens, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in tokens.items()
    ):
        raise RuntimeError("AUTH_TOKENS must be a JSON object of token: user_id")
    return tokens


def user_for_token(token: str, tokens: dict[str, str]) -> str | None:
    return tokens.get(token)


async def get_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    try:
        tokens = load_auth_tokens()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    if not tokens:
        raise HTTPException(503, "Authentication is not configured (set AUTH_TOKENS)")

    if credentials is None:
        raise HTTPException(
            401, "Missing bearer token", headers={"WWW-Authenticate": "Bearer"}
        )
    user_id = user_for_token(credentials.credentials, tokens)
    if user_id is None:
        raise HTTPException(
            401, "Invalid bearer token", headers={"WWW-Authenticate": "Bearer"}
        )
    return user_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    store_cm = PostgresStore.from_conn_string(require_db_url())
    store = store_cm.__enter__()
    store.setup()

    checkpointer_cm = AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB)
    checkpointer = await checkpointer_cm.__aenter__()

    compiled = (await build_graph()).compile(checkpointer=checkpointer, store=store)
    app.state.graph = compiled
    app.state.store_cm = store_cm
    app.state.checkpointer_cm = checkpointer_cm

    try:
        yield
    finally:
        await checkpointer_cm.__aexit__(None, None, None)
        store_cm.__exit__(None, None, None)


app = FastAPI(
    title="MyChatbot API",
    description="API for MyChatbot using FastAPI and LangGraph",
    lifespan=lifespan,
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    thread_id: str = Field(default="default", min_length=1)


@app.get("/")
def home():
    return {"message": "Welcome to MyChatbot API!"}


@app.post("/ask")
async def ask_question(request: QuestionRequest, user_id: str = Depends(get_user_id)):
    if not request.question.strip():
        raise HTTPException(400, "Question must not be blank")

    config = {
        "configurable": {
            "thread_id": f"{user_id}:{request.thread_id}",
            "user_id": user_id,
        }
    }
    try:
        result = await app.state.graph.ainvoke(
            {"messages": [HumanMessage(content=request.question)]},
            config=config,
        )
    except Exception:
        logger.exception("Graph invocation failed for user %s", user_id)
        raise HTTPException(502, "Chat request failed") from None

    return {
        "response": extract_text(result["messages"][-1].content),
        "token_count": result.get("token_count"),
    }

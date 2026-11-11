from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.postgres import PostgresStore

from Mychatbot import build_graph
from memory import DB_URL


# ---- Lifespan: open PostgresStore + checkpointer + compile graph once ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    store_cm = PostgresStore.from_conn_string(DB_URL)
    store = store_cm.__enter__()
    store.setup()

    checkpointer_cm = AsyncSqliteSaver.from_conn_string("checkpoints.sqlite")
    checkpointer = await checkpointer_cm.__aenter__()

    raw_graph = await build_graph()
    compiled = raw_graph.compile(checkpointer=checkpointer, store=store)

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
    question: str
    user_id: str
    thread_id: str = "default"


@app.get("/")
def home():
    return {"message": "Welcome to MyChatbot API!"}


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    graph = app.state.graph

    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": request.user_id,
        }
    }

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=request.question)]},
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

    return {
        "response": ai_text,
        "token_count": result.get("token_count"),
    }
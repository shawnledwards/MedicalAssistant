from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes.chat import router
from medical_assistant.config.settings import Settings
from medical_assistant.rag.pipeline import RAGPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.pipeline = RAGPipeline(settings=settings)
    yield
    app.state.pipeline = None


app = FastAPI(
    title="Medical Assistant RAG",
    description="RAG-powered medical Q&A with persona-aware responses",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve frontend — must be last so API routes take priority
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

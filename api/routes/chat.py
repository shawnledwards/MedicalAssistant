from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from api.schemas.models import ChatRequest, ChatResponse, HealthResponse
from medical_assistant.core.vector_store import stores_ready

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    pipeline = request.app.state.pipeline
    try:
        result = await run_in_threadpool(
            pipeline.run_query, question=body.question, persona=body.persona
        )
        result.pop("context", None)
        return ChatResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    pipeline = request.app.state.pipeline
    return HealthResponse(
        status="ok",
        model_loaded=pipeline.is_ready(),
        vector_store_ready=stores_ready(),
    )

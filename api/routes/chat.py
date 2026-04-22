from fastapi import APIRouter, HTTPException, Request

from api.schemas.models import ChatRequest, ChatResponse, HealthResponse
from medical_assistant.core.llm import is_loaded
from medical_assistant.core.vector_store import stores_ready

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    pipeline = request.app.state.pipeline
    try:
        result = pipeline.run_query(question=body.question, persona=body.persona)
        result.pop("context", None)
        return ChatResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=is_loaded(),
        vector_store_ready=stores_ready(),
    )

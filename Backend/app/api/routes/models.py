from fastapi import APIRouter
from app.services.model_registry import model_registry
from app.api.schemas.api_models import (
    ModelsListResponse,
    ModelInfo,
    CheckpointsResponse,
    CheckpointEntry
)

router = APIRouter()

@router.get("/models", response_model=ModelsListResponse)
def list_models():
    models_info = model_registry.list_models_info()
    return ModelsListResponse(
        models=[ModelInfo(**m) for m in models_info],
        count=len(models_info)
    )

@router.get("/checkpoints", response_model=CheckpointsResponse)
def get_checkpoints_ledger():
    ledger = model_registry.get_checkpoints_ledger()
    return CheckpointsResponse(
        checkpoints=[CheckpointEntry(**c) for c in ledger],
        count=len(ledger)
    )

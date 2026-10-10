import torch
from fastapi import APIRouter
from app.config import settings
from app.api.schemas.api_models import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def get_health():
    cuda_avail = torch.cuda.is_available()
    device = "CUDA (" + torch.cuda.get_device_name(0) + ")" if cuda_avail else "CPU (Standard Mode)"
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        torch_version=torch.__version__,
        cuda_available=cuda_avail,
        device=device,
        active_storage_dir=str(settings.STORAGE_DIR)
    )

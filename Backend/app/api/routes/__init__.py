from fastapi import APIRouter
from app.api.routes.health import router as health_router
from app.api.routes.models import router as models_router
from app.api.routes.image import router as image_router
from app.api.routes.inference import router as inference_router
from app.api.routes.downloads import router as downloads_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["Health"])
api_router.include_router(models_router, tags=["Models"])
api_router.include_router(image_router, tags=["Image"])
api_router.include_router(inference_router, tags=["Inference"])
api_router.include_router(downloads_router, tags=["Downloads"])

__all__ = ["api_router"]

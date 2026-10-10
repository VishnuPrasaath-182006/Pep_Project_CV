import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "CAVE-Flood Inundation & Semantic Segmentation API"
    APP_VERSION: str = "2.4.0-Production"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Path Resolution
    # Path of Backend/
    BACKEND_DIR: Path = Path(__file__).resolve().parent.parent
    # Workspace Root
    PROJECT_ROOT: Path = BACKEND_DIR.parent
    
    # Storage Directories
    STORAGE_DIR: Path = BACKEND_DIR / "storage"
    UPLOADS_DIR: Path = STORAGE_DIR / "uploads"
    PREVIEWS_DIR: Path = STORAGE_DIR / "previews"
    MASKS_DIR: Path = STORAGE_DIR / "masks"
    OVERLAYS_DIR: Path = STORAGE_DIR / "overlays"
    EXPORTS_DIR: Path = STORAGE_DIR / "exports"
    SAMPLE_CHIPS_DIR: Path = STORAGE_DIR / "sample_chips"
    
    # Model Architecture Roots (Read-Only)
    UNET_MAMBA_DIR: Path = PROJECT_ROOT / "Models" / "BaseModel and its Extension" / "UNet-Mamba"
    UNET_MAMBA_EXT_DIR: Path = PROJECT_ROOT / "Models" / "BaseModel and its Extension" / "UNet-Mamba Extension"
    UNET_DIR: Path = PROJECT_ROOT / "Models" / "unet"
    SEGFORMER_DIR: Path = PROJECT_ROOT / "Models" / "SegFormer"
    DATASETS_DIR: Path = PROJECT_ROOT / "Models" / "Datasets"
    
    # Checkpoints
    UNET_MAMBA_CHECKPOINT: Path = UNET_MAMBA_DIR / "checkpoints" / "best_model.pth"
    UNET_CHECKPOINT: Path = UNET_DIR / "checkpoints" / "best_model.pth"
    SEGFORMER_CHECKPOINT: Path = SEGFORMER_DIR / "outputs" / "checkpoints" / "best_model.pth"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
    
    # Maximum upload size (100 MB)
    MAX_UPLOAD_SIZE_BYTES: int = 100 * 1024 * 1024
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Ensure runtime storage directories exist inside Backend/storage
for directory in [
    settings.STORAGE_DIR,
    settings.UPLOADS_DIR,
    settings.PREVIEWS_DIR,
    settings.MASKS_DIR,
    settings.OVERLAYS_DIR,
    settings.EXPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

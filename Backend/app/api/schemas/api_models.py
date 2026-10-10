from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    torch_version: str
    cuda_available: bool
    device: str
    active_storage_dir: str

class ModelInfo(BaseModel):
    id: str
    name: str
    architecture: str
    status: str
    readiness: str = "Ready"
    is_available: bool
    checkpoint_path: Optional[str] = None
    artifact_filename: Optional[str] = None
    file_size_formatted: Optional[str] = None
    error_detail: Optional[str] = None
    input_channels: int
    expected_bands: List[str]
    input_shape: str
    classes: Dict[int, str]
    historical_metrics: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
    count: int

class CheckpointEntry(BaseModel):
    model_id: str
    model_name: str
    architecture: str
    artifact_filename: str
    expected_path: str
    actual_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    file_size_formatted: str
    sha256_checksum: Optional[str] = None
    verification_status: str
    deployment_readiness: str
    notes: Optional[str] = None

class CheckpointsResponse(BaseModel):
    checkpoints: List[CheckpointEntry]
    count: int

class RasterMetadata(BaseModel):
    filename: str
    format: str
    file_size_bytes: int
    file_size_formatted: str
    width: int
    height: int
    channels: int
    dtype: str
    crs: Optional[str] = None
    bounds: Optional[List[float]] = None
    resolution: Optional[List[float]] = None
    is_multispectral: bool
    has_nir: bool
    preview_url: str
    compatibility: Dict[str, bool]
    compatibility_notes: Dict[str, str]

class SingleInferenceResult(BaseModel):
    result_id: str
    model_id: str
    model_name: str
    processing_time_ms: float
    device: str
    image_width: int
    image_height: int
    valid_pixels: int
    flood_pixels: int
    background_pixels: int
    flood_coverage_percentage: float
    mean_confidence: float
    original_preview_url: str
    mask_url: str
    overlay_url: str
    geotiff_mask_url: Optional[str] = None
    metadata: RasterMetadata

class CompareResultItem(BaseModel):
    model_id: str
    model_name: str
    status: str
    is_successful: bool
    error_message: Optional[str] = None
    result: Optional[SingleInferenceResult] = None

class CompareResponse(BaseModel):
    result_id: str
    input_image_name: str
    total_processing_time_ms: float
    models_evaluated: List[CompareResultItem]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

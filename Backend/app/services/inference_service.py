import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

from app.config import settings
from app.utils.geo_utils import (
    read_raster_file,
    generate_preview_image,
    generate_flood_overlay,
    save_geotiff_mask
)
from app.services.model_registry import model_registry
from app.services.image_service import image_service
from app.api.schemas.api_models import SingleInferenceResult, RasterMetadata

class InferenceService:
    """
    Coordinates end-to-end model execution, mask synthesis,
    flood overlay generation, GeoTIFF spatial export, and result caching.
    """
    
    def __init__(self):
        self._results_cache: Dict[str, Dict[str, Any]] = {}
        self._results_dir = settings.STORAGE_DIR / "results"
        self._results_dir.mkdir(parents=True, exist_ok=True)
        
    def execute_inference(
        self,
        image_path: Path,
        model_id: str,
        metadata: RasterMetadata
    ) -> SingleInferenceResult:
        adapter = model_registry.get_adapter(model_id)
        if not adapter:
            raise ValueError(f"Unknown model identifier: '{model_id}'")
            
        if not adapter.is_available:
            raise RuntimeError(
                f"Model '{adapter.name}' is currently unavailable. {adapter.status_message}"
            )
            
        # Read raw raster data
        data, file_meta = read_raster_file(image_path)
        
        # Execute genuine model inference
        raw_result = adapter.run_inference(data, file_meta)
        
        result_id = str(uuid.uuid4())[:8]
        binary_mask = raw_result["binary_mask"]
        
        # 1. Save binary mask as PNG
        mask_filename = f"mask_{result_id}_{model_id}.png"
        mask_path = settings.MASKS_DIR / mask_filename
        mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
        mask_img.save(mask_path, format="PNG")
        
        # 2. Save colored overlay PNG
        # Original preview path is resolved from metadata.preview_url
        preview_filename = Path(metadata.preview_url).name
        preview_path = settings.PREVIEWS_DIR / preview_filename
        
        overlay_filename = f"overlay_{result_id}_{model_id}.png"
        overlay_path = settings.OVERLAYS_DIR / overlay_filename
        generate_flood_overlay(preview_path, binary_mask, overlay_path, alpha=0.60)
        
        # 3. Save GeoTIFF mask with source spatial projection
        geotiff_filename = f"geotiff_mask_{result_id}_{model_id}.tif"
        geotiff_path = settings.EXPORTS_DIR / geotiff_filename
        source_profile = file_meta.get("profile")
        saved_geotiff = save_geotiff_mask(binary_mask, geotiff_path, source_profile=source_profile)
        
        geotiff_url = f"/api/storage/exports/{geotiff_filename}" if saved_geotiff else None
        
        result = SingleInferenceResult(
            result_id=result_id,
            model_id=adapter.model_id,
            model_name=adapter.name,
            processing_time_ms=raw_result["processing_time_ms"],
            device=raw_result["device"],
            image_width=metadata.width,
            image_height=metadata.height,
            valid_pixels=raw_result["valid_pixels"],
            flood_pixels=raw_result["flood_pixels"],
            background_pixels=raw_result["background_pixels"],
            flood_coverage_percentage=raw_result["flood_coverage_percentage"],
            mean_confidence=raw_result["mean_confidence"],
            original_preview_url=metadata.preview_url,
            mask_url=f"/api/storage/masks/{mask_filename}",
            overlay_url=f"/api/storage/overlays/{overlay_filename}",
            geotiff_mask_url=geotiff_url,
            metadata=metadata
        )
        
        # Cache result
        self._results_cache[result_id] = result.model_dump()
        result_json_path = self._results_dir / f"{result_id}.json"
        with open(result_json_path, "w") as f:
            json.dump(result.model_dump(), f, indent=2)
            
        return result

    def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        if result_id in self._results_cache:
            return self._results_cache[result_id]
            
        json_path = self._results_dir / f"{result_id}.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)
                self._results_cache[result_id] = data
                return data
        return None

inference_service = InferenceService()

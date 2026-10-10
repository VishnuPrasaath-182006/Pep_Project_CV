import os
import uuid
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np

from app.config import settings
from app.utils.geo_utils import read_raster_file, generate_preview_image
from app.services.model_registry import model_registry
from app.api.schemas.api_models import RasterMetadata

class ImageService:
    """
    Manages satellite raster ingestion, storage, metadata extraction,
    preview generation, and model compatibility validation.
    """
    
    def process_upload(self, file_bytes: bytes, original_filename: str) -> Tuple[Path, RasterMetadata]:
        # Generate safe unique ID
        file_id = str(uuid.uuid4())[:8]
        safe_ext = Path(original_filename).suffix
        if not safe_ext:
            safe_ext = ".tif"
            
        stored_filename = f"{file_id}_{Path(original_filename).stem}{safe_ext}"
        upload_path = settings.UPLOADS_DIR / stored_filename
        
        with open(upload_path, "wb") as f:
            f.write(file_bytes)
            
        # Inspect raster
        data, meta = read_raster_file(upload_path)
        
        # Generate normalized preview PNG
        preview_filename = f"preview_{file_id}.png"
        preview_path = settings.PREVIEWS_DIR / preview_filename
        generate_preview_image(data, preview_path)
        
        preview_url = f"/api/storage/previews/{preview_filename}"
        
        # Test compatibility across all architectures
        compat = {}
        notes = {}
        for adapter in model_registry.list_adapters():
            is_compat, note = adapter.check_compatibility(meta)
            compat[adapter.model_id] = is_compat
            notes[adapter.model_id] = note
            
        c = meta["channels"]
        raster_metadata = RasterMetadata(
            filename=original_filename,
            format=meta["format"],
            file_size_bytes=meta["file_size_bytes"],
            file_size_formatted=meta["file_size_formatted"],
            width=meta["width"],
            height=meta["height"],
            channels=c,
            dtype=meta["dtype"],
            crs=meta["crs"],
            bounds=meta["bounds"],
            resolution=meta["resolution"],
            is_multispectral=(c >= 4),
            has_nir=(c >= 4),
            preview_url=preview_url,
            compatibility=compat,
            compatibility_notes=notes
        )
        
        return upload_path, raster_metadata

    def get_existing_upload_and_metadata(self, preview_url: str) -> Tuple[Optional[Path], Optional[RasterMetadata]]:
        preview_stem = Path(preview_url).stem.replace("preview_", "")
        matches = list(settings.UPLOADS_DIR.glob(f"{preview_stem}_*"))
        if not matches:
            return None, None
            
        upload_path = matches[0]
        data, meta = read_raster_file(upload_path)
        
        compat = {}
        notes = {}
        for adapter in model_registry.list_adapters():
            is_compat, note = adapter.check_compatibility(meta)
            compat[adapter.model_id] = is_compat
            notes[adapter.model_id] = note
            
        c = meta["channels"]
        original_filename = upload_path.name
        if original_filename.startswith(f"{preview_stem}_"):
            original_filename = original_filename[len(preview_stem) + 1:]
            
        raster_metadata = RasterMetadata(
            filename=original_filename,
            format=meta["format"],
            file_size_bytes=meta["file_size_bytes"],
            file_size_formatted=meta["file_size_formatted"],
            width=meta["width"],
            height=meta["height"],
            channels=c,
            dtype=meta["dtype"],
            crs=meta["crs"],
            bounds=meta["bounds"],
            resolution=meta["resolution"],
            is_multispectral=(c >= 4),
            has_nir=(c >= 4),
            preview_url=preview_url,
            compatibility=compat,
            compatibility_notes=notes
        )
        return upload_path, raster_metadata

image_service = ImageService()

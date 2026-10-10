from pathlib import Path
from typing import List, Dict
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.config import settings
from app.services.image_service import image_service
from app.api.schemas.api_models import RasterMetadata

router = APIRouter()

@router.post("/image/inspect", response_model=RasterMetadata)
async def inspect_uploaded_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty")
        
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
    if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_BYTES / 1e6} MB"
        )
        
    try:
        _, metadata = image_service.process_upload(content, file.filename)
        return metadata
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to inspect satellite raster: {str(e)}")

@router.get("/image/samples", response_model=List[Dict[str, str]])
def list_sample_chips():
    """Lists available authentic PlanetScope test scenes from the research dataset."""
    sample_dir = settings.SAMPLE_CHIPS_DIR
    if not sample_dir.exists():
        return []
        
    samples = []
    for f in sorted(sample_dir.glob("*.tif")):
        samples.append({
            "id": f.stem,
            "filename": f.name,
            "description": f"PlanetScope 4-band Scene ({f.stat().st_size / 1e6:.1f} MB)",
            "event": "Bangladesh Monsoon Flood (PlanetScope 4-Channel)"
        })
    return samples

@router.post("/image/load-sample", response_model=RasterMetadata)
def load_sample_chip(sample_id: str = Query(..., description="ID of sample scene")):
    sample_dir = settings.SAMPLE_CHIPS_DIR
    candidate = sample_dir / f"{sample_id}.tif"
    if not candidate.exists():
        # Try matching by stem
        matches = list(sample_dir.glob(f"*{sample_id}*.tif"))
        if matches:
            candidate = matches[0]
        else:
            raise HTTPException(status_code=404, detail=f"Sample scene '{sample_id}' not found.")
        
    with open(candidate, "rb") as f:
        content = f.read()
        
    _, metadata = image_service.process_upload(content, candidate.name)
    return metadata

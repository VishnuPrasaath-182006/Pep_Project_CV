import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from app.config import settings
from app.services.inference_service import inference_service

router = APIRouter()

@router.get("/download/{result_id}/{asset_type}")
def download_asset(result_id: str, asset_type: str):
    # Sanitize inputs
    clean_id = "".join([c for c in result_id if c.isalnum() or c in "-_"])
    if not clean_id:
        raise HTTPException(status_code=400, detail="Invalid result ID")
        
    result_data = inference_service.get_result(clean_id)
    if not result_data:
        raise HTTPException(status_code=404, detail="Inference result not found")
        
    model_id = result_data.get("model_id", "model")
    
    if asset_type == "mask":
        filename = f"mask_{clean_id}_{model_id}.png"
        file_path = settings.MASKS_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Mask file not found")
        return FileResponse(file_path, media_type="image/png", filename=f"flood_mask_{clean_id}.png")
        
    elif asset_type == "overlay":
        filename = f"overlay_{clean_id}_{model_id}.png"
        file_path = settings.OVERLAYS_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Overlay file not found")
        return FileResponse(file_path, media_type="image/png", filename=f"flood_overlay_{clean_id}.png")
        
    elif asset_type == "geotiff":
        filename = f"geotiff_mask_{clean_id}_{model_id}.tif"
        file_path = settings.EXPORTS_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="GeoTIFF export file not found")
        return FileResponse(file_path, media_type="image/tiff", filename=f"flood_mask_{clean_id}.tif")
        
    elif asset_type == "report":
        report_json = json.dumps(result_data, indent=2)
        return Response(
            content=report_json,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=flood_report_{clean_id}.json"}
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown asset type '{asset_type}'. Supported types: 'mask', 'overlay', 'geotiff', 'report'."
        )

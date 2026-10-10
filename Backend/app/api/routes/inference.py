import time
from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query

from app.config import settings
from app.services.image_service import image_service
from app.services.inference_service import inference_service
from app.services.model_registry import model_registry
from app.api.schemas.api_models import (
    SingleInferenceResult,
    CompareResponse,
    CompareResultItem
)

router = APIRouter()

@router.post("/predict", response_model=SingleInferenceResult)
async def predict_single(
    file: Optional[UploadFile] = File(None),
    model_id: str = Form("unet_mamba"),
    preview_url: Optional[str] = Form(None)
):
    upload_path = None
    metadata = None
    
    if preview_url:
        upload_path, metadata = image_service.get_existing_upload_and_metadata(preview_url)
        
    if upload_path is None and file is not None and file.filename:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        upload_path, metadata = image_service.process_upload(content, file.filename)
    elif upload_path is None:
        raise HTTPException(
            status_code=400,
            detail="Must provide either an inspected preview_url or a valid satellite image file."
        )
        
    try:
        result = inference_service.execute_inference(upload_path, model_id, metadata)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")

@router.post("/compare", response_model=CompareResponse)
async def compare_models(
    file: Optional[UploadFile] = File(None),
    preview_url: Optional[str] = Form(None),
    model_ids: Optional[str] = Form("unet_mamba,unet,segformer")
):
    start_time = time.time()
    
    upload_path = None
    metadata = None
    
    if preview_url:
        upload_path, metadata = image_service.get_existing_upload_and_metadata(preview_url)
        
    if upload_path is None and file is not None and file.filename:
        content = await file.read()
        upload_path, metadata = image_service.process_upload(content, file.filename)
    elif upload_path is None:
        raise HTTPException(
            status_code=400,
            detail="Must provide either an inspected preview_url or a valid satellite image file."
        )
        
    requested_models = [m.strip() for m in model_ids.split(",") if m.strip()]
    evaluated_items: List[CompareResultItem] = []
    
    for m_id in requested_models:
        adapter = model_registry.get_adapter(m_id)
        if not adapter:
            evaluated_items.append(CompareResultItem(
                model_id=m_id,
                model_name=m_id,
                status="Unknown Model",
                is_successful=False,
                error_message=f"Architecture '{m_id}' is not registered."
            ))
            continue
            
        if not adapter.is_available:
            evaluated_items.append(CompareResultItem(
                model_id=adapter.model_id,
                model_name=adapter.name,
                status=adapter.status_message,
                is_successful=False,
                error_message=f"Trained checkpoint not found. Architecture preserved in {adapter.get_model_info().get('notes', '')}"
            ))
            continue
            
        try:
            res = inference_service.execute_inference(upload_path, m_id, metadata)
            evaluated_items.append(CompareResultItem(
                model_id=adapter.model_id,
                model_name=adapter.name,
                status="Completed",
                is_successful=True,
                result=res
            ))
        except Exception as e:
            evaluated_items.append(CompareResultItem(
                model_id=adapter.model_id,
                model_name=adapter.name,
                status="Failed",
                is_successful=False,
                error_message=str(e)
            ))
            
    total_time_ms = round((time.time() - start_time) * 1000.0, 2)
    return CompareResponse(
        result_id=metadata.preview_url.split("/")[-1].replace(".png", ""),
        input_image_name=metadata.filename,
        total_processing_time_ms=total_time_ms,
        models_evaluated=evaluated_items
    )

@router.get("/results/{result_id}", response_model=SingleInferenceResult)
def get_prediction_result(result_id: str):
    data = inference_service.get_result(result_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Inference result '{result_id}' not found.")
    return SingleInferenceResult(**data)

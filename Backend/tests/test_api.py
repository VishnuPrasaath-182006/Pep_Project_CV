import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "torch_version" in data
    assert "cuda_available" in data

def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 3
    model_ids = [m["id"] for m in data["models"]]
    assert "unet_mamba" in model_ids
    assert "unet" in model_ids
    assert "segformer" in model_ids
    
    # Verify UNet-Mamba is detected and ready
    mamba_info = next(m for m in data["models"] if m["id"] == "unet_mamba")
    assert mamba_info["is_available"] is True
    assert mamba_info["readiness"] == "Ready"
    assert "Ready" in mamba_info["status"]
    assert mamba_info["artifact_filename"] == "best_model.pth"
    
    # Verify U-Net is activated and ready
    unet_info = next(m for m in data["models"] if m["id"] == "unet")
    assert unet_info["is_available"] is True
    assert unet_info["readiness"] == "Ready"
    assert "Active / Ready" in unet_info["status"]

    # Verify SegFormer is activated and ready
    segformer_info = next(m for m in data["models"] if m["id"] == "segformer")
    assert segformer_info["is_available"] is True
    assert segformer_info["readiness"] == "Ready"
    assert "Active / Ready" in segformer_info["status"]

def test_checkpoints_ledger_endpoint():
    response = client.get("/api/checkpoints")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    
    # Verify UNet-Mamba verified entry
    mamba_entry = next(c for c in data["checkpoints"] if c["model_id"] == "unet_mamba")
    assert mamba_entry["artifact_filename"] == "best_model.pth"
    assert mamba_entry["file_size_bytes"] == 171815178
    assert mamba_entry["verification_status"] == "SHA-256 Verified"
    assert mamba_entry["deployment_readiness"] == "Verified & Ready"
    assert mamba_entry["sha256_checksum"] == "58674c84e97e16c7ab86f2e218ba8738c7aaf69a8c6fd8c384de00beec9eb76d"

    # Verify U-Net entry
    unet_entry = next(c for c in data["checkpoints"] if c["model_id"] == "unet")
    assert unet_entry["artifact_filename"] == "best_val_iou.pth"
    assert unet_entry["deployment_readiness"] in ["Verified & Ready", "Ready", "Preserved (Checkpoint Pending)"]

    # Verify SegFormer entry
    seg_entry = next(c for c in data["checkpoints"] if c["model_id"] == "segformer")
    assert seg_entry["artifact_filename"] == "fold_00_best.pt"
    assert seg_entry["deployment_readiness"] in ["Verified & Ready", "Ready", "Preserved (Checkpoint Pending)"]

def test_sample_chips_listing():
    response = client.get("/api/image/samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) > 0
    assert any("Bangladesh" in s["id"] for s in samples)

def test_load_sample_inspection():
    samples = client.get("/api/image/samples").json()
    sample_id = samples[0]["id"]
    response = client.post(f"/api/image/load-sample?sample_id={sample_id}")
    assert response.status_code == 200
    meta = response.json()
    assert meta["width"] == 512
    assert meta["height"] == 512
    assert meta["channels"] == 4
    assert meta["is_multispectral"] is True
    assert meta["compatibility"]["unet_mamba"] is True

def test_predict_and_results_and_downloads():
    samples = client.get("/api/image/samples").json()
    sample_id = samples[0]["id"]
    load_resp = client.post(f"/api/image/load-sample?sample_id={sample_id}")
    assert load_resp.status_code == 200
    meta = load_resp.json()
    
    # 1. Run prediction
    pred_resp = client.post(
        "/api/predict",
        data={
            "model_id": "unet_mamba",
            "preview_url": meta["preview_url"]
        }
    )
    assert pred_resp.status_code == 200
    res = pred_resp.json()
    result_id = res["result_id"]
    assert res["model_id"] == "unet_mamba"
    assert res["valid_pixels"] == 512 * 512
    assert res["flood_pixels"] >= 0
    assert 0.0 <= res["flood_coverage_percentage"] <= 100.0
    assert res["mask_url"].endswith(".png")
    assert res["overlay_url"].endswith(".png")
    
    # 2. Retrieve result by ID
    get_res = client.get(f"/api/results/{result_id}")
    assert get_res.status_code == 200
    assert get_res.json()["result_id"] == result_id

    # 3. Test asset downloads
    mask_dl = client.get(f"/api/download/{result_id}/mask")
    assert mask_dl.status_code == 200
    assert len(mask_dl.content) > 0

    overlay_dl = client.get(f"/api/download/{result_id}/overlay")
    assert overlay_dl.status_code == 200
    assert len(overlay_dl.content) > 0

    report_dl = client.get(f"/api/download/{result_id}/report")
    assert report_dl.status_code == 200
    assert len(report_dl.content) > 0

def test_compare_endpoint():
    samples = client.get("/api/image/samples").json()
    sample_id = samples[0]["id"]
    load_resp = client.post(f"/api/image/load-sample?sample_id={sample_id}")
    assert load_resp.status_code == 200
    meta = load_resp.json()

    compare_resp = client.post(
        "/api/compare",
        data={
            "preview_url": meta["preview_url"],
            "model_ids": "unet_mamba,unet,segformer"
        }
    )
    assert compare_resp.status_code == 200
    comp_data = compare_resp.json()
    assert len(comp_data["models_evaluated"]) == 3
    
    # UNet-Mamba succeeded
    mamba_res = next(m for m in comp_data["models_evaluated"] if m["model_id"] == "unet_mamba")
    assert mamba_res["is_successful"] is True
    assert mamba_res["result"] is not None

    # U-Net succeeded
    unet_res = next(m for m in comp_data["models_evaluated"] if m["model_id"] == "unet")
    assert unet_res["is_successful"] is True
    assert unet_res["result"] is not None

    # SegFormer succeeded
    seg_res = next(m for m in comp_data["models_evaluated"] if m["model_id"] == "segformer")
    assert seg_res["is_successful"] is True
    assert seg_res["result"] is not None

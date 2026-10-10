"""Unit tests for SegFormer forward pass, backpropagation, and checkpointing (Test 5, Test 6, Test 11)."""

import os
import tempfile
import torch
import pytest

from src.models.segformer import SegFormerFloodModel


def test_model_forward_pass_and_shape():
    """Verify that SegFormer-B2 processes input correctly and produces resized logits."""
    model = SegFormerFloodModel(pretrained=False)
    model.eval()

    # Input: [B, 3, H, W]
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)

    logits = out["logits"]
    assert logits.shape == (1, 2, 512, 512), f"Expected (1, 2, 512, 512), got {logits.shape}"


def test_model_backpropagation():
    """Verify that backpropagation computes finite gradients and updates parameters."""
    model = SegFormerFloodModel(pretrained=False)
    model.train()

    x = torch.randn(2, 3, 256, 256)
    y = torch.randint(0, 2, (2, 256, 256)).long()
    y[0, :5, :5] = 255  # ignore pixels

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    out = model(x, labels=y)
    loss = out["loss"]
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    loss.backward()

    # Check that at least some parameters have non-zero gradients
    grads_exist = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in model.parameters())
    assert grads_exist, "No gradients were computed during backpropagation!"

    optimizer.step()


def test_checkpoint_saving_and_loading():
    """Verify that model weights can be saved, restored, and used for identical inference."""
    model_orig = SegFormerFloodModel(pretrained=False)
    model_orig.eval()
    x = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        logits_orig = model_orig(x)["logits"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test_checkpoint.pt")
        torch.save({"model_state_dict": model_orig.state_dict()}, ckpt_path)

        # Restore into new model
        model_loaded = SegFormerFloodModel(pretrained=False)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_loaded.load_state_dict(ckpt["model_state_dict"])
        model_loaded.eval()

        with torch.no_grad():
            logits_loaded = model_loaded(x)["logits"]

        assert torch.allclose(logits_orig, logits_loaded, atol=1e-5), "Restored model output mismatch!"

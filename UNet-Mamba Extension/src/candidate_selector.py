"""
Candidate-Region Selector & RemoteCLIP Guidance Module
Implements:
1. Division of 512x512 patch into sixteen 128x128 regions (4x4 grid).
2. Calculation of normalized binary entropy and flood-probability gradient magnitude.
3. Region filtering: requires >= 50% valid/source coverage.
4. Normalized rank averaging (0.5 entropy + 0.5 gradient).
5. Top 4 candidate region selection (binary candidate mask C).
6. RemoteCLIP (ViT-B/32) vision-language guidance margin calculation and score scaling.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Fixed Balanced Prompt Set (Section 7)
FLOOD_PROMPTS = [
    "Satellite image of land inundated by floodwater.",
    "Satellite image of floodwater covering agricultural fields.",
    "Satellite image of floodwater covering roads and buildings.",
    "Remote sensing image showing a flooded landscape."
]

NONFLOOD_PROMPTS = [
    "Satellite image of land without visible inundation.",
    "Satellite image of non-flooded agricultural fields.",
    "Satellite image of dry roads and buildings.",
    "Remote sensing image showing a non-flooded landscape."
]


def compute_entropy_and_gradient(flood_prob, eps=1e-7):
    """
    Computes normalized binary entropy and gradient magnitude for a 2D flood probability map.
    Input: flood_prob (H, W) in [0, 1]
    Output:
        entropy_norm (H, W) in [0, 1]
        grad_mag (H, W) in [0, 1]
    """
    p = np.clip(flood_prob, eps, 1.0 - eps)
    entropy = - (p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    # Normalized binary entropy (max is 1.0 at p=0.5)
    entropy_norm = np.clip(entropy, 0.0, 1.0)
    
    # 2D spatial gradients
    gy, gx = np.gradient(flood_prob)
    grad_mag = np.sqrt(gy**2 + gx**2)
    # Theoretical max spatial gradient between adjacent pixels in [0, 1] is ~0.707
    grad_norm = np.clip(grad_mag / 0.7071, 0.0, 1.0)
    
    return entropy_norm.astype(np.float32), grad_norm.astype(np.float32)


def select_candidate_regions(
    flood_prob,
    valid_mask=None,
    patch_size=512,
    region_size=128,
    top_k=4,
    min_coverage=0.5
):
    """
    Divides 512x512 patch into sixteen 128x128 regions, filters by coverage >= 50%,
    ranks eligible regions by average of normalized entropy and gradient ranks,
    and returns top_k candidate mask and metadata.
    """
    h, w = flood_prob.shape
    assert h == patch_size and w == patch_size, f"Expected shape ({patch_size}, {patch_size}), got ({h}, {w})"
    
    if valid_mask is None:
        valid_mask = np.ones((h, w), dtype=bool)
        
    entropy_map, grad_map = compute_entropy_and_gradient(flood_prob)
    
    n_regions_per_dim = patch_size // region_size  # 4
    total_region_pixels = region_size * region_size  # 16,384
    
    eligible_regions = []
    
    for r in range(n_regions_per_dim):
        for c in range(n_regions_per_dim):
            top = r * region_size
            left = c * region_size
            
            reg_valid = valid_mask[top:top+region_size, left:left+region_size]
            valid_count = int(np.sum(reg_valid))
            coverage = valid_count / total_region_pixels
            
            if coverage >= min_coverage and valid_count > 0:
                mean_h = float(np.mean(entropy_map[top:top+region_size, left:left+region_size][reg_valid]))
                mean_g = float(np.mean(grad_map[top:top+region_size, left:left+region_size][reg_valid]))
                eligible_regions.append({
                    "r": r,
                    "c": c,
                    "top": top,
                    "left": left,
                    "coverage": coverage,
                    "valid_count": valid_count,
                    "mean_entropy": mean_h,
                    "mean_gradient": mean_g
                })
                
    cand_mask = np.zeros((h, w), dtype=np.float32)
    selected_regions = []
    
    M = len(eligible_regions)
    if M > 0:
        h_vals = [reg["mean_entropy"] for reg in eligible_regions]
        g_vals = [reg["mean_gradient"] for reg in eligible_regions]
        
        # Rank each cue (higher value -> higher rank)
        rank_h = scipy.stats.rankdata(h_vals) / float(M)
        rank_g = scipy.stats.rankdata(g_vals) / float(M)
        
        for i, reg in enumerate(eligible_regions):
            reg["score"] = float(0.5 * rank_h[i] + 0.5 * rank_g[i])
            
        selected_regions = sorted(eligible_regions, key=lambda x: x["score"], reverse=True)[:top_k]
        
        for reg in selected_regions:
            cand_mask[reg["top"]:reg["top"]+region_size, reg["left"]:reg["left"]+region_size] = 1.0
            
    return {
        "candidate_mask": cand_mask,
        "selected_regions": selected_regions,
        "eligible_count": M,
        "entropy_map": entropy_map,
        "gradient_map": grad_map
    }


class RemoteCLIPGuidance:
    """
    Manages frozen RemoteCLIP ViT-B/32 text and image embedding extraction,
    margin calculation, and score scaling.
    """
    def __init__(self, device="cpu", scaler_path=None):
        import open_clip
        from huggingface_hub import hf_hub_download
        
        self.device = torch.device(device)
        ckpt_path = hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=ckpt_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
            
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self._encode_prompts()
        
        # Scaler parameters
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.mean = 0.0
        self.std = 1.0
        if self.scaler_path and self.scaler_path.exists():
            self.load_scaler(self.scaler_path)

    def _encode_prompts(self):
        with torch.no_grad():
            f_tokens = self.tokenizer(FLOOD_PROMPTS).to(self.device)
            nf_tokens = self.tokenizer(NONFLOOD_PROMPTS).to(self.device)
            
            f_embeds = self.model.encode_text(f_tokens)
            nf_embeds = self.model.encode_text(nf_tokens)
            
            self.flood_text_embeds = f_embeds / f_embeds.norm(dim=-1, keepdim=True)
            self.nonflood_text_embeds = nf_embeds / nf_embeds.norm(dim=-1, keepdim=True)

    def compute_region_margin(self, region_rgb_uint8):
        """
        Computes raw margin = mean_flood_sim - mean_nonflood_sim for an RGB (H, W, 3) uint8 image.
        """
        pil_img = Image.fromarray(region_rgb_uint8)
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            img_feat = self.model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            
            f_sims = (img_feat @ self.flood_text_embeds.T).mean().item()
            nf_sims = (img_feat @ self.nonflood_text_embeds.T).mean().item()
            
        raw_margin = float(f_sims - nf_sims)
        return raw_margin

    def fit_scaler(self, raw_margins, save_path=None):
        margins = np.array(raw_margins, dtype=np.float32)
        self.mean = float(np.mean(margins))
        self.std = float(np.std(margins))
        if self.std < 1e-6:
            self.std = 1.0
            
        scaler_data = {
            "mean": self.mean,
            "std": self.std,
            "count": len(margins),
            "min": float(np.min(margins)),
            "max": float(np.max(margins))
        }
        
        if save_path:
            save_path = Path(save_path)
            os.makedirs(save_path.parent, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(scaler_data, f, indent=2)
            self.scaler_path = save_path
            print(f"Saved RemoteCLIP score scaler to: {save_path}")
            print(f"Scaler parameters: Mean={self.mean:.4f}, Std={self.std:.4f} (fitted on {len(margins)} regions)")
        return scaler_data

    def load_scaler(self, scaler_path):
        with open(scaler_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.mean = data["mean"]
        self.std = data["std"]
        self.scaler_path = Path(scaler_path)
        print(f"Loaded RemoteCLIP score scaler from: {scaler_path} (Mean={self.mean:.4f}, Std={self.std:.4f})")

    def scale_margin(self, raw_margin):
        return (raw_margin - self.mean) / (self.std + 1e-6)

    def build_guidance_map(self, patch_img_raw, selected_regions, region_size=128):
        """
        Builds 512x512 VLM guidance map: scaled margin broadcast over selected regions, 0 outside.
        patch_img_raw: (4, 512, 512) raw PlanetScope uint16 (channels: R, G, B, NIR)
        """
        h, w = patch_img_raw.shape[1], patch_img_raw.shape[2]
        guidance_map = np.zeros((h, w), dtype=np.float32)
        
        # Robust RGB scaling to uint8 (using channels 0, 1, 2)
        rgb_raw = patch_img_raw[:3].astype(np.float32)
        rgb_uint8 = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            c_min = np.percentile(rgb_raw[c], 1)
            c_max = np.percentile(rgb_raw[c], 99)
            if c_max > c_min:
                norm_c = np.clip((rgb_raw[c] - c_min) / (c_max - c_min), 0.0, 1.0)
            else:
                norm_c = np.zeros_like(rgb_raw[c])
            rgb_uint8[:, :, c] = (norm_c * 255.0).astype(np.uint8)
            
        region_margins = []
        for reg in selected_regions:
            top = reg["top"]
            left = reg["left"]
            reg_rgb = rgb_uint8[top:top+region_size, left:left+region_size]
            
            raw_margin = self.compute_region_margin(reg_rgb)
            scaled_margin = self.scale_margin(raw_margin)
            
            guidance_map[top:top+region_size, left:left+region_size] = scaled_margin
            region_margins.append({"top": top, "left": left, "raw_margin": raw_margin, "scaled_margin": scaled_margin})
            
        return guidance_map, region_margins

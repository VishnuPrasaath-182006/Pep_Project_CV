import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import numpy as np
import torch
from config import load_config
from evaluate_source_chips import compute_binary_metrics
from casr import CASRAdapter

def analyze():
    config = load_config()
    val_dir = Path('artifacts/CAVE_Flood_Antigravity/cache/candidate_regions/val')
    val_files = sorted(list(val_dir.glob('*.npz')))

    ckpt_e2 = torch.load('UNet-Mamba/checkpoints/best_e2_model.pth', weights_only=True)
    ckpt_e1 = torch.load('UNet-Mamba/checkpoints/best_e1_model.pth', weights_only=True)

    scenes = {}
    for f in val_files:
        stem_key = f.stem.rsplit('_p', 1)[0]
        p_idx = int(f.stem.rsplit('_p', 1)[1])
        if stem_key not in scenes:
            scenes[stem_key] = {}
        scenes[stem_key][p_idx] = f

    quadrant_coords = {0: (0, 0), 1: (0, 512), 2: (512, 0), 3: (512, 512)}
    norm_mean = np.array(config['data']['norm_mean'], dtype=np.float32).reshape(4, 1, 1)
    norm_std = np.array(config['data']['norm_std'], dtype=np.float32).reshape(4, 1, 1)

    adapter_e2 = CASRAdapter(in_channels=9, hidden_channels=32, num_classes=2, lambda_res=1.0)
    adapter_e2.load_state_dict(ckpt_e2['casr_state_dict'])
    adapter_e2.eval()

    adapter_e1 = CASRAdapter(in_channels=9, hidden_channels=32, num_classes=2, lambda_res=1.0)
    adapter_e1.load_state_dict(ckpt_e1['casr_state_dict'])
    adapter_e1.eval()

    scene_results = []
    for stem_key, p_map in scenes.items():
        if len(p_map) != 4:
            continue
        
        full_mask = np.zeros((1024, 1024), dtype=np.int64)
        full_img = np.zeros((4, 1024, 1024), dtype=np.int64)
        e0_logits = np.zeros((2, 1024, 1024), dtype=np.float32)
        e1_logits = np.zeros((2, 1024, 1024), dtype=np.float32)
        e2_logits = np.zeros((2, 1024, 1024), dtype=np.float32)
        
        for p_idx, p_path in p_map.items():
            r, c = quadrant_coords[p_idx]
            data = np.load(p_path)
            full_mask[r:r+512, c:c+512] = data['raw_mask']
            full_img[:, r:r+512, c:c+512] = data['raw_image']
            
            base_logits = data['patch_logits']
            e0_logits[:, r:r+512, c:c+512] = base_logits
            
            raw_img = data['raw_image']
            cand_mask = data['candidate_mask']
            flood_prob = data['flood_prob']
            entropy_map = data['entropy_map']
            grad_map = data['gradient_map']
            img_norm = (raw_img.astype(np.float32) - norm_mean) / (norm_std + 1e-6)
            
            # E1
            feats_e1 = np.concatenate([
                img_norm, flood_prob[np.newaxis, ...], entropy_map[np.newaxis, ...],
                grad_map[np.newaxis, ...], cand_mask[np.newaxis, ...], data['guidance_e1'][np.newaxis, ...]
            ], axis=0).astype(np.float32)
            
            # E2
            feats_e2 = np.concatenate([
                img_norm, flood_prob[np.newaxis, ...], entropy_map[np.newaxis, ...],
                grad_map[np.newaxis, ...], cand_mask[np.newaxis, ...], data['guidance_e2'][np.newaxis, ...]
            ], axis=0).astype(np.float32)
            
            with torch.no_grad():
                cand_t = torch.from_numpy(cand_mask).unsqueeze(0).unsqueeze(0).float()
                _, _, res1 = adapter_e1(torch.from_numpy(feats_e1).unsqueeze(0).float(), cand_t)
                _, _, res2 = adapter_e2(torch.from_numpy(feats_e2).unsqueeze(0).float(), cand_t)
                
                z1 = torch.from_numpy(base_logits).unsqueeze(0) + res1
                z2 = torch.from_numpy(base_logits).unsqueeze(0) + res2
                e1_logits[:, r:r+512, c:c+512] = z1.squeeze(0).numpy()
                e2_logits[:, r:r+512, c:c+512] = z2.squeeze(0).numpy()

        valid_pixels = (full_img != 0).any(axis=0) & (full_mask != -1)
        if valid_pixels.sum() == 0:
            continue
            
        y_true = full_mask[valid_pixels]
        def get_metrics(y_true, y_pred):
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            return compute_binary_metrics(tp, fp, fn, tn)

        m0 = get_metrics(y_true, np.argmax(e0_logits, axis=0)[valid_pixels])
        m1 = get_metrics(y_true, np.argmax(e1_logits, axis=0)[valid_pixels])
        m2 = get_metrics(y_true, np.argmax(e2_logits, axis=0)[valid_pixels])
        
        scene_results.append({
            'scene': stem_key,
            'event': stem_key.split('_')[0],
            'e0_iou': m0['flood_iou'],
            'e1_iou': m1['flood_iou'],
            'e2_iou': m2['flood_iou'],
            'diff_e2': m2['flood_iou'] - m0['flood_iou'],
            'diff_e1': m1['flood_iou'] - m0['flood_iou'],
            'flood_pixels': int((y_true == 1).sum()),
            'valid_pixels': int(valid_pixels.sum())
        })

    print(f"Total evaluated scenes: {len(scene_results)}")
    
    # Sort by E2 gain
    scene_results.sort(key=lambda x: x['diff_e2'], reverse=True)
    print("\n--- TOP 5 GAINS (E2 vs E0) ---")
    for s in scene_results[:5]:
        print(f"{s['scene']}: E0={s['e0_iou']:.4f} -> E2={s['e2_iou']:.4f} (E1={s['e1_iou']:.4f}) | Diff E2: +{s['diff_e2']:.4f} | Flood Px: {s['flood_pixels']}")

    print("\n--- US-DAKOTA SCENES ---")
    dakota = [s for s in scene_results if s['event'] == 'US-Dakota']
    dakota.sort(key=lambda x: x['diff_e2'], reverse=True)
    for s in dakota[:5]:
        print(f"{s['scene']}: E0={s['e0_iou']:.4f} -> E2={s['e2_iou']:.4f} (E1={s['e1_iou']:.4f}) | Diff E2: {s['diff_e2']:+.4f} | Flood Px: {s['flood_pixels']}")

    print("\n--- MODERATE / TYPICAL SCENES ---")
    # Bolivia scenes with high flood content
    bolivia_flood = [s for s in scene_results if s['event'] == 'Bolivia' and s['flood_pixels'] > 50000]
    bolivia_flood.sort(key=lambda x: x['diff_e2'], reverse=True)
    for s in bolivia_flood[:5]:
        print(f"{s['scene']}: E0={s['e0_iou']:.4f} -> E2={s['e2_iou']:.4f} (E1={s['e1_iou']:.4f}) | Diff E2: {s['diff_e2']:+.4f} | Flood Px: {s['flood_pixels']}")

if __name__ == '__main__':
    analyze()

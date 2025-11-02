import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ============================================================
# 1️⃣ Simple Overlay Visualizer
# ============================================================
def save_visuals(model, loader, out_dir: Path, device="cuda", limit=10):
    """
    Save simple overlay images (Input + Prediction).
    Each overlay shows model-predicted mask overlaid on the grayscale image.

    Args:
        model: trained segmentation model
        loader: PyTorch DataLoader (image, mask)
        out_dir: Path to save visualization results
        device: 'cuda' or 'cpu'
        limit: number of samples to save (default=10)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for idx, (img, mask) in enumerate(tqdm(loader, desc="Saving overlays")):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            pred_bin = (torch.sigmoid(pred) > 0.5).float()

            # Convert tensors to uint8 numpy arrays
            img_np = (img[0, 0].cpu().numpy() * 255).astype(np.uint8)
            pred_np = (pred_bin[0, 0].cpu().numpy() * 255).astype(np.uint8)

            # Overlay (blend)
            overlay = cv2.addWeighted(img_np, 0.7, pred_np, 0.3, 0)
            cv2.imwrite(str(out_dir / f"sample_{idx:03d}.png"), overlay)

            if idx + 1 >= limit:
                break

    print(f"✅ Saved {min(limit, len(loader))} overlay samples to {out_dir}")


# ============================================================
# 2️⃣ GT vs Pred vs Overlay Visualizer (Paper-Quality)
# ============================================================
def save_visuals_triplet(model, loader, out_dir: Path, device="cuda", limit=10, layout="horizontal"):
    """
    Save triplet visuals: [Ground Truth | Prediction | Overlay].
    Used for paper-quality qualitative results.

    Args:
        model: trained segmentation model
        loader: PyTorch DataLoader (image, mask)
        out_dir: Path to save visualization results
        device: 'cuda' or 'cpu'
        limit: number of samples to save (default=10)
        layout: 'horizontal' or 'vertical'
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for idx, (img, mask) in enumerate(tqdm(loader, desc="Saving GT–Pred–Overlay visuals")):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            pred_bin = (torch.sigmoid(pred) > 0.5).float()

            # Convert to uint8 numpy arrays
            img_np = (img[0, 0].cpu().numpy() * 255).astype(np.uint8)
            gt_np = (mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
            pr_np = (pred_bin[0, 0].cpu().numpy() * 255).astype(np.uint8)

            # Convert to RGB
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            gt_rgb = cv2.cvtColor(gt_np, cv2.COLOR_GRAY2BGR)
            pr_rgb = cv2.cvtColor(pr_np, cv2.COLOR_GRAY2BGR)

            # Overlay (red prediction)
            overlay = img_rgb.copy()
            red_mask = np.zeros_like(overlay)
            red_mask[..., 2] = 255
            alpha = 0.35
            overlay = np.where(pr_rgb > 128, (1 - alpha) * overlay + alpha * red_mask, overlay).astype(np.uint8)

            # Label helper
            def label(img, text):
                return cv2.putText(
                    img.copy(), text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                )

            gt_labeled = label(gt_rgb, "Ground Truth")
            pr_labeled = label(pr_rgb, "Prediction")
            ov_labeled = label(overlay, "Overlay")

            # Combine horizontally or vertically
            if layout == "horizontal":
                combined = np.hstack([gt_labeled, pr_labeled, ov_labeled])
            else:
                combined = np.vstack([gt_labeled, pr_labeled, ov_labeled])

            # Save output
            out_path = out_dir / f"vis_{idx:03d}.png"
            cv2.imwrite(str(out_path), combined)

            if idx + 1 >= limit:
                break

    print(f"✅ Saved {min(limit, len(loader))} triplet visuals to {out_dir}")

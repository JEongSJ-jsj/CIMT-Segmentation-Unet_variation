import os, cv2, json, shutil, argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image  # ✅ Full TIFF compatibility

# ======================================================
# 1️⃣ Generate segmentation masks (LI/MA → PNG)
# ======================================================
def generate_masks(base_dir, annotator="Manual-A1"):
    """
    Convert LI/MA boundary text files into binary segmentation masks.
    Each mask is filled between LI (lumen–intima) and MA (media–adventitia) contours.
    """
    seg_dir = Path(base_dir) / "SEGMENTATIONS" / annotator
    img_dir = Path(base_dir) / "IMAGES"
    out_dir = Path(base_dir) / f"masks_{annotator}"
    out_dir.mkdir(parents=True, exist_ok=True)

    li_files = sorted(seg_dir.glob("*-LI.txt"))
    print(f"[1/4] Generating masks from boundary coordinates ({len(li_files)} files)...")

    for li_path in tqdm(li_files):
        stem = li_path.stem.replace("-LI", "")
        ma_path = seg_dir / f"{stem}-MA.txt"
        img_path = img_dir / f"{stem}.tiff"

        # --- Try alternate extensions if .tiff not found ---
        if not img_path.exists():
            for ext in [".tif", ".png", ".jpg"]:
                alt = img_dir / f"{stem}{ext}"
                if alt.exists():
                    img_path = alt
                    break
        if not img_path.exists() or not ma_path.exists():
            continue

        try:
            img = Image.open(str(img_path)).convert("L")
            h, w = img.size[1], img.size[0]
        except:
            continue

        li_pts = np.loadtxt(li_path, dtype=np.float32)
        ma_pts = np.loadtxt(ma_path, dtype=np.float32)

        mask = np.zeros((h, w), dtype=np.uint8)
        contour = np.vstack([li_pts, ma_pts[::-1]]).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [contour], 255)
        cv2.imwrite(str(out_dir / f"{stem}.png"), mask)

    print(f"✅ Masks saved to: {out_dir}")
    return out_dir


# ======================================================
# 2️⃣ Apply CLAHE enhancement + intensity standardization
# ======================================================
def apply_clahe_and_standardize(base_dir, annotator="Manual-A1"):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to enhance local contrast and normalize grayscale intensities.
    Saves standardized image/mask pairs under 'data_std/'.
    """
    img_dir = Path(base_dir) / "IMAGES"
    mask_dir = Path(base_dir) / f"masks_{annotator}"
    out_dir = Path(base_dir) / "data_std"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    print(f"[2/4] Applying CLAHE and standardizing dataset...")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ✅ Include all major image extensions (recursive)
    img_paths = sorted(list(img_dir.rglob("*.tif")) +
                       list(img_dir.rglob("*.tiff")) +
                       list(img_dir.rglob("*.png")) +
                       list(img_dir.rglob("*.jpg")))

    if len(img_paths) == 0:
        print(f"⚠️ No images found in {img_dir}")
        return out_dir

    for img_path in tqdm(img_paths):
        fname = img_path.name
        mask_path = mask_dir / (Path(fname).stem + ".png")

        # Fallback to .tif mask if necessary
        if not mask_path.exists():
            alt_mask = mask_dir / (Path(fname).stem + ".tif")
            if alt_mask.exists():
                mask_path = alt_mask
            else:
                continue

        # --- Read image using PIL (TIFF-safe) ---
        try:
            img = Image.open(str(img_path)).convert("L")
            img = np.array(img, dtype=np.uint8)
        except:
            print(f"⚠️ Could not read {img_path.name}")
            continue

        # --- Read corresponding mask (OpenCV fallback) ---
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            try:
                mask = np.array(Image.open(str(mask_path)).convert("L"), dtype=np.uint8)
            except:
                print(f"⚠️ Could not read mask {mask_path.name}")
                continue

        # --- Apply CLAHE enhancement ---
        img_clahe = clahe.apply(img)
        cv2.imwrite(str(out_dir / "images" / (Path(fname).stem + ".png")), img_clahe)
        cv2.imwrite(str(out_dir / "masks" / (Path(fname).stem + ".png")), mask)

    print(f"✅ data_std created at: {out_dir}")
    return out_dir


# ======================================================
# 3️⃣ Verify image–mask pair consistency
# ======================================================
def verify_pairs(data_std_dir):
    """
    Check if each image has a corresponding mask of the same dimensions.
    Saves any errors to verify_report.json.
    """
    img_dir = Path(data_std_dir) / "images"
    mask_dir = Path(data_std_dir) / "masks"
    errors = []

    print(f"[3/4] Verifying image–mask pairs...")
    for img_path in tqdm(sorted(img_dir.glob("*.png"))):
        fname = img_path.name
        mask_path = mask_dir / fname
        if not mask_path.exists():
            errors.append(f"Missing mask for {fname}")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img.shape != mask.shape:
            errors.append(f"Shape mismatch {fname}: {img.shape} vs {mask.shape}")

    report_path = Path(data_std_dir) / "verify_report.json"
    json.dump(errors, open(report_path, "w"), indent=2)
    print(f"✅ Verification complete. Errors: {len(errors)} → saved at {report_path}")
    return errors


# ======================================================
# 4️⃣ Split dataset into train/val/test subsets (6:2:2)
# ======================================================
def split_dataset(data_std_dir, ratios=(0.6, 0.2, 0.2), seed=42):
    """
    Split the standardized dataset into train/validation/test sets.
    Default ratio = 6:2:2. Copies images and masks into separate folders.
    """
    img_dir = Path(data_std_dir) / "images"
    mask_dir = Path(data_std_dir) / "masks"
    img_paths = sorted(img_dir.glob("*.png"))
    if len(img_paths) == 0:
        raise RuntimeError("No images found in data_std/images")

    print(f"[4/4] Splitting dataset into train/val/test ({ratios})...")
    train_paths, temp_paths = train_test_split(img_paths, test_size=(1 - ratios[0]), random_state=seed)
    val_ratio_adj = ratios[1] / (ratios[1] + ratios[2])
    val_paths, test_paths = train_test_split(temp_paths, test_size=(1 - val_ratio_adj), random_state=seed)

    def copy_set(paths, subset):
        for p in paths:
            fname = p.name
            img_out = Path(data_std_dir) / subset / "images" / fname
            mask_out = Path(data_std_dir) / subset / "masks" / fname
            img_out.parent.mkdir(parents=True, exist_ok=True)
            mask_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, img_out)
            shutil.copy2(mask_dir / fname, mask_out)

    copy_set(train_paths, "train")
    copy_set(val_paths, "val")
    copy_set(test_paths, "test")

    summary = {"train": len(train_paths), "val": len(val_paths), "test": len(test_paths)}
    json.dump(summary, open(Path(data_std_dir) / "split_summary.json", "w"), indent=2)
    print(f"✅ Split complete: {summary}")
    return summary


# ======================================================
# 🧩 Main execution
# ======================================================
def main():
    """
    Full CIMT preprocessing pipeline:
      1. Generate segmentation masks
      2. Apply CLAHE & intensity normalization
      3. Verify image–mask integrity
      4. Split dataset into train/val/test subsets
    """
    parser = argparse.ArgumentParser(description="Full CIMT preprocessing pipeline (generate + CLAHE + verify + split)")
    parser.add_argument("--base_dir", type=str, required=True, help="Base dataset directory containing IMAGES/ and SEGMENTATIONS/")
    parser.add_argument("--annotator", type=str, default="Manual-A1", help="SEGMENTATIONS subfolder name")
    args = parser.parse_args()

    base_dir = args.base_dir
    annotator = args.annotator

    generate_masks(base_dir, annotator)
    data_std_dir = apply_clahe_and_standardize(base_dir, annotator)
    verify_pairs(data_std_dir)
    split_dataset(data_std_dir)
    print("\n✅ All preprocessing steps completed successfully!")


if __name__ == "__main__":
    main()

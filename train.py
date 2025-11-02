import os, time, csv, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import get_model
from dataset import CIMTDataset
from utils.metrics import dice_coeff, iou_score, precision_score, recall_score, accuracy_score
from utils.logger import save_training_curves, save_runtime_summary
from utils.visualize import save_visuals
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================
# Default hyperparameter settings per model (from parametercheck)
# ==============================================================
_DEFAULTS = {
    "unet": dict(base=33),
    "unetpp": dict(base=32),
    "resunet": dict(base=36),
    "attention_unet": dict(base_c=64),
    "attention_resunet": dict(base=35),
    "seunet": dict(base=33),
    " denseunet": dict(base=36, growth=20, layers=6),
    "inceptionunet": dict(base=60),
    "transunet": dict(embed_dim=256, depth=6, heads=4, mlp_ratio=4),
  # ≈ 15.18M params
    "unext": dict(base=40, expansion=4),
}

# ==============================================================
# Model selection & training configuration
# ==============================================================
model_list = [
    "unet", "unetpp", "resunet", "attention_unet", "attention_resunet",
    "seunet", "denseunet", "inceptionunet", "transunet", "unext"
]

print("\n==============================")
print(" CIMT Segmentation Training ")
print("==============================\n")

print("Select models to train:")
for i, m in enumerate(model_list, 1):
    print(f"{i}. {m}")
print("\n💡 Example:")
print(" - Comma separated: 1,4,9")
print(" - Range selection: 1-10 (means all models)\n")

raw_input_str = input("Enter model indices: ").strip()

# Parse model selection input (supports 1-5 and 1,4,9 formats)
selected_indices = []
for part in raw_input_str.split(","):
    if "-" in part:
        start, end = map(int, part.split("-"))
        selected_indices.extend(range(start, end + 1))
    elif part.strip().isdigit():
        selected_indices.append(int(part.strip()))

selected_models = [model_list[i - 1] for i in selected_indices if 1 <= i <= len(model_list)]

epochs = input("Enter number of epochs (default=100): ").strip()
EPOCHS = int(epochs) if epochs else 100

print(f"\n✅ Selected Models: {', '.join(selected_models)}")
print(f"✅ Training Epochs: {EPOCHS}\n")

# ==============================================================
# Dataset paths / environment setup
# ==============================================================
DATA_ROOT = Path("CIMT_Dataset/data_std")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
LR = 1e-4
RESULTS_BASE = Path("results")
RESULTS_BASE.mkdir(exist_ok=True)

summary_records = []

# ==============================================================
# Training loop per model
# ==============================================================
for model_name in selected_models:
    print(f"\n🚀 Training model: {model_name}")
    RESULTS_DIR = RESULTS_BASE / model_name
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Dataset preparation ---
    train_ds = CIMTDataset(DATA_ROOT / "train")
    val_ds = CIMTDataset(DATA_ROOT / "val")
    test_ds = CIMTDataset(DATA_ROOT / "test")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=1)

    # --- Model initialization (with default hyperparams) ---
    model_kwargs = _DEFAULTS.get(model_name, {})
    model = get_model(model_name, **model_kwargs).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f" - Parameters: {param_count:,}")

    # --- Loss functions ---
    bce = nn.BCEWithLogitsLoss()

    def dice_loss(pred, target):
        smooth = 1e-6
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    def hybrid_loss(pred, target):
        return 0.5 * bce(pred, target) + 0.5 * dice_loss(pred, target)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # --- Training log CSV ---
    csv_path = RESULTS_DIR / "train_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "dice", "iou", "precision", "recall", "accuracy", "lr", "epoch_time"])

    best_dice = 0.0
    total_train_time = 0.0
    train_start = time.time()

    # ==============================================================
    # Epoch loop
    # ==============================================================
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        train_loss = 0

        for img, mask in tqdm(train_loader, desc=f"[Train {model_name} {epoch+1}/{EPOCHS}]"):
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            pred = model(img)
            loss = hybrid_loss(pred, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss, dices, ious, precs, recs, accs = 0, [], [], [], [], []
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                pred = model(img)
                val_loss += hybrid_loss(pred, mask).item()
                pred_bin = (torch.sigmoid(pred) > 0.5).float()
                dices.append(dice_coeff(pred_bin, mask))
                ious.append(iou_score(pred_bin, mask))
                precs.append(precision_score(pred_bin, mask))
                recs.append(recall_score(pred_bin, mask))
                accs.append(accuracy_score(pred_bin, mask))
        val_loss /= len(val_loader)

        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time

        avg_dice = torch.mean(torch.tensor(dices)).item()
        avg_iou  = torch.mean(torch.tensor(ious)).item()
        avg_prec = torch.mean(torch.tensor(precs)).item()
        avg_rec  = torch.mean(torch.tensor(recs)).item()
        avg_acc  = torch.mean(torch.tensor(accs)).item()
        lr = optimizer.param_groups[0]["lr"]

        # --- Log current epoch ---
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, val_loss, avg_dice, avg_iou,
                             avg_prec, avg_rec, avg_acc, lr, epoch_time])

        print(f"Epoch {epoch+1:03d}/{EPOCHS} "
              f"Loss={train_loss:.4f}/{val_loss:.4f} "
              f"Dice={avg_dice:.4f} IoU={avg_iou:.4f}")

        # --- Save best model ---
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), RESULTS_DIR / "best_model.pth")

    # ==============================================================
    # Post-training summary
    # ==============================================================
    total_train_time = time.time() - train_start
    time_per_epoch = total_train_time / EPOCHS

    # --- Inference time evaluation ---
    model.eval()
    inference_times = []
    with torch.no_grad():
        for img, _ in test_loader:
            img = img.to(DEVICE)
            t0 = time.time()
            _ = model(img)
            t1 = time.time()
            inference_times.append(t1 - t0)
    inference_time_avg = np.mean(inference_times)

    # --- Save TorchScript version (.pt) ---
    example_input = torch.randn(1, 1, 256, 256).to(DEVICE)
    traced = torch.jit.trace(model, example_input)
    traced.save(str(RESULTS_DIR / "final_model.pt"))

    print(f"\n✅ Saved:\n - best_model.pth\n - final_model.pt")

    # --- Save metrics summary record ---
    summary_records.append({
        "model": model_name,
        "params": param_count,
        "total_time": round(total_train_time, 2),
        "time_per_epoch": round(time_per_epoch, 2),
        "inference_time": round(float(inference_time_avg), 5),
        "best_dice": round(best_dice, 4)
    })

# ==============================================================
# Save global training summary
# ==============================================================
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(RESULTS_BASE / "summary.csv", index=False)
print("\n📊 Training complete for all selected models.")
print("Results summary saved to results/summary.csv")

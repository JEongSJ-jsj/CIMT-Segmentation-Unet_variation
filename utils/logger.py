import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def save_training_curves(df: pd.DataFrame, out_dir: Path):
    # ---- Loss Curve ----
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training/Validation Loss Curve")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=200)
    plt.close()

    # ---- Metrics Curve ----
    plt.figure()
    plt.plot(df["epoch"], df["dice"], label="Dice")
    plt.plot(df["epoch"], df["iou"], label="IoU")
    plt.xlabel("Epoch"); plt.ylabel("Score")
    plt.title("Validation Metrics Curve")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_curve.png", dpi=200)
    plt.close()

def save_runtime_summary(summary_df, out_dir: Path):
    """Runtime summary bar plot"""
    plt.figure(figsize=(8,4))
    plt.bar(summary_df["model"], summary_df["time_per_epoch"])
    plt.ylabel("Time per Epoch (s)")
    plt.title("Model Runtime Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_bar.png", dpi=200)
    plt.close()

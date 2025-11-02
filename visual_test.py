# visual_test.py
import torch
from pathlib import Path
from visualize import save_visuals_triplet
from dataset import CIMTDataset
from models import get_model

# === 1. Load model ===
model = get_model("attention_unet", base_c=64)
model.load_state_dict(torch.load("results/attention_unet/best_model.pth"))
model.eval().cuda()

# === 2. Load validation data ===
val_dataset = CIMTDataset("CUBS/data_std/val")

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# === 3. Visualize ===
save_visuals_triplet(
    model=model,
    loader=val_loader,
    out_dir=Path("results/vis_attention_unet"),
    device="cuda",
    limit=10,
    layout="horizontal"
)

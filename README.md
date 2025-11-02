# 


# 🧠 CIMT Segmentation — Pure Python Run Guide

This repository implements **10 U-Net-based architectures** for **Carotid Intima-Media Thickness (CIMT)** ultrasound segmentation using **only Python**, with no Conda or additional setup tools required.

---

## 📂 Folder Structure

```
cimtseg_models_v4/
 ├─ models/               # All U-Net variants
 ├─ utils/                # Metrics, visualization, logger
 ├─ dataset.py            # Dataset loader
 ├─ train.py              # Model training script
 ├─ preprocess_full.py    # Data preprocessing pipeline
 ├─ visualize.py          # Visualization functions
 ├─ visual_test.py        # Visualization test runner
 ├─ results/              # Trained weights and logs
 └─ CUBS/                 # Dataset folder (after preprocessing)
```

---

## ⚙️ 1. Install Requirements

Make sure Python ≥ 3.10 is installed.

Install the required packages directly with `pip`:

```bash
pip install torch torchvision torchaudio
pip install opencv-python albumentations pillow tqdm numpy pandas matplotlib scikit-learn
```

No virtual environment is required — these are standard PyPI packages only.

---
📦 Dataset: CUBS (Carotid Ultrasound B-mode Segmentation)

This project uses the CUBS dataset published on Mendeley Data
.

🔗 Download Instructions

Go to the dataset page:
👉 https://data.mendeley.com/datasets/fpv535fss7/1

Click the "Download All Files" button on the right side.
This will download a .zip file (≈ several hundred MB).

Extract the archive anywhere, for example:

C:\Users\<username>\Desktop\CUBS


The extracted folder should contain:

CUBS/
├── IMAGES/
│   ├── *.tif / *.tiff (ultrasound images)
├── SEGMENTATIONS/
│   ├── Manual-A1/
│   │   ├── *-LI.txt
│   │   ├── *-MA.txt

🧰 Preprocessing Before Training

Run the preprocessing script to generate masks, standardize images (CLAHE), verify data, and split into train/val/test:

python preprocess_full.py --base_dir "C:\Users\<username>\Desktop\CUBS" --annotator "Manual-A1"


This will automatically create:

CUBS/
└── data_std/
    ├── images/
    ├── masks/
    ├── train/
    ├── val/
    ├── test/
## 🧮 2. Dataset Preprocessing

Place your dataset in the following structure:

```
CUBS/
 ├─ IMAGES/
 └─ SEGMENTATIONS/Manual-A1/
```

Run the preprocessing pipeline:

```bash
python preprocess_full.py --base_dir CUBS --annotator Manual-A1
```

This will automatically:

1. Generate mask PNGs from LI/MA boundary text files  
2. Apply CLAHE (contrast enhancement)  
3. Verify image–mask shape consistency  
4. Split into `train`, `val`, and `test` subsets (6 : 2 : 2)

Output structure:

```
CUBS/data_std/
 ├─ train/
 ├─ val/
 ├─ test/
 └─ split_summary.json
```

---

## 🚀 3. Train Models

Run training interactively from the command line:

```bash
python train.py
```

Example prompt:

```
==============================
 CIMT Segmentation Training
==============================

Select models to train:
1. unet
2. unetpp
3. resunet
4. attention_unet
5. attention_resunet
6. seunet
7. denseunet
8. inceptionunet
9. transunet
10. unext

💡 Example:
 - Comma separated: 1,4,9
 - Range selection: 1-10 (means all models)

Enter model indices: 1-10
Enter number of epochs (default=100): 2
```

Training results are saved to:

```
results/
 ├─ [model_name]/best_model.pth
 ├─ [model_name]/final_model.pt
 ├─ [model_name]/train_log.csv
 └─ summary.csv
```

---

## 🎨 4. Visualization

After training, visualize model predictions and overlays:

```bash
python visual_test.py
```

The script loads the trained model and saves visualization triplets (`Input`, `Ground Truth`, `Prediction`) under:

```
results/vis_[model_name]/
```

Each output shows:
- Left → original ultrasound image  
- Middle → ground-truth segmentation  
- Right → predicted overlay

---

## 🧾 5. Independent Script Usage

| Task | Command |
|------|----------|
| Run preprocessing only | `python preprocess_full.py --base_dir CUBS` |
| Train one or multiple models | `python train.py` |
| Generate visualization outputs | `python visual_test.py` |

All scripts are fully self-contained; no external configs or notebooks are required.

---

## 🧠 Models Implemented

| Type | Model | Description |
|------|--------|-------------|
| Base | **U-Net** | Standard encoder–decoder |
| Nested | **U-Net++** | Dense skip connections |
| Residual | **ResU-Net** | Residual convolution blocks |
| Attention | **Attention U-Net** | Spatial attention in skip paths |
| Hybrid | **Attention ResU-Net** | Residual + attention mechanism |
| Squeeze | **SE-U-Net** | Channel attention (Squeeze-Excitation) |
| Dense | **DenseU-Net** | Dense connections per block |
| Inception | **Inception U-Net** | Multi-scale inception features |
| Transformer | **TransU-Net** | CNN + Transformer encoder |
| Lightweight | **UNeXt** | Efficient MLP-based segmentation |

---

## 📊 Evaluation Metrics

Each epoch logs the following:

- Binary Cross-Entropy (BCE) + Dice hybrid loss  
- Dice Coefficient  
- Intersection over Union (IoU)  
- Precision / Recall  
- Accuracy  
- Runtime per epoch and inference time

All values are recorded in `train_log.csv` and summarized in `results/summary.csv`.

---

## 💡 Notes

- To adjust dataset path, edit `DATA_ROOT` in **train.py**:  
  ```python
  DATA_ROOT = Path("CUBS/data_std")
  ```
- To reduce memory usage:  
  ```python
  BATCH_SIZE = 4
  ```
- All `.pth` files are standard PyTorch checkpoints, compatible with both CPU and GPU.

---

## 🏁 Quick Command Summary

| Step | Command | Description |
|------|----------|-------------|
| 1️⃣ Preprocess | `python preprocess_full.py --base_dir CUBS` | Prepare dataset |
| 2️⃣ Train | `python train.py` | Train selected models |
| 3️⃣ Visualize | `python visual_test.py` | Save overlay results |

---

## 🧾 License

Released under the **MIT License**.  
Use freely for research, education, and development.

# CIMT Segmentation with U-Net Variants  
*(⚠ This implementation does not perfectly match the parameters reported in the original paper. It is a functional and optimized version for practical experimentation.)*

---

## 📘 Overview
This repository provides a **complete Python-only pipeline** for Carotid Intima–Media Thickness (CIMT) ultrasound segmentation using **10 U-Net variants**, including:
- U-Net  
- U-Net++  
- ResU-Net  
- Attention U-Net  
- Attention ResU-Net  
- SE-U-Net  
- DenseU-Net  
- Inception U-Net  
- TransUNet  
- UNeXt  

The project covers:
- Dataset preprocessing (mask generation, CLAHE, verification, splitting)
- Model training and evaluation
- Automatic logging and visualization
- Lightweight visualization pipeline for GT–Prediction–Overlay images

---

## ⚙️ Requirements
Python ≥ 3.9  
Recommended environment: **conda**

```bash
conda create -n cimt python=3.10
conda activate cimt
pip install -r requirements.txt
```

> (If `requirements.txt` is missing, install core libs manually)
> ```bash
> pip install torch torchvision albumentations opencv-python pillow tqdm scikit-learn matplotlib
> ```

---

## 📦 Dataset: CUBS (Carotid Ultrasound B-mode Segmentation)

This project uses the **CUBS dataset** from Mendeley Data:  
🔗 [https://data.mendeley.com/datasets/fpv535fss7/1](https://data.mendeley.com/datasets/fpv535fss7/1)

### 🔽 Download Instructions
1. Visit the dataset page above.  
2. Click **“Download All Files”** on the right-hand side.  
3. Extract the downloaded archive, for example:
   ```
   C:\Users\<username>\Desktop\CUBS
   ```
4. After extraction, ensure the folder structure looks like:
   ```
   CUBS/
   ├── IMAGES/
   │   ├── *.tif / *.tiff (ultrasound images)
   ├── SEGMENTATIONS/
   │   ├── Manual-A1/
   │   │   ├── *-LI.txt
   │   │   ├── *-MA.txt
   ```

---

## 🧰 Preprocessing (Mask + CLAHE + Split)

Run the following command to automatically:
- Generate segmentation masks from LI/MA boundaries  
- Apply CLAHE and normalize the dataset  
- Verify image-mask pairs  
- Split into **train/val/test (6:2:2)**

```bash
python preprocess_full.py --base_dir "C:\Users\<username>\Desktop\CUBS" --annotator "Manual-A1"
```

This creates a standardized dataset:
```
CUBS/
└── data_std/
    ├── images/
    ├── masks/
    ├── train/
    ├── val/
    ├── test/
```

---

## 🧠 Model Training

Start the unified training script:
```bash
python train.py
```

You’ll be prompted to:
- Select which models to train (`1-10` or comma-separated list)
- Specify training epochs

Example:
```
Enter model indices: 1-10
Enter number of epochs (default=100): 50
```

Each model will:
- Train automatically on GPU (if available)
- Save `best_model.pth` and `final_model.pt` to the project root
- Log metrics (Loss, Dice, IoU) per epoch

---

## 🎨 Visualization (GT–Prediction–Overlay)

After training, you can visualize sample predictions using:

```bash
python visual_test.py
```

This will create a folder like:
```
results/
└── vis_attention_unet/
    ├── sample_000_triplet.png
    ├── sample_001_triplet.png
    └── ...
```

Each visual shows:
1. **Ground Truth (GT)**  
2. **Predicted Mask**  
3. **Overlay (Image + Prediction)**

---

## 🗂 Folder Structure
```
cimtseg_models_v4/
│
├─ models/              ← all U-Net variants
│
├─ utils/
│   ├─ metrics.py
│   ├─ visualize.py
│   ├─ logger.py
│   └─ __init__.py
│
├─ dataset.py           ← dataset loader
├─ train.py             ← main training script
├─ preprocess_full.py   ← dataset preprocessing pipeline
├─ visual_test.py       ← visual result generation
├─ README.md
└─ requirements.txt
```

---

## 📋 Notes
- The parameter counts are adjusted for GPU efficiency (not exactly identical to original papers).  
- The code automatically detects GPU via `torch.cuda.is_available()`.  
- All logs and model checkpoints are saved in the project root.  

---

## 📜 Citation
If you use this implementation or part of it, please cite the original **CUBS dataset**:

> Carotid ultrasound B-mode images for segmentation and analysis (CUBS dataset),  
> Mendeley Data, V1, DOI: [10.17632/fpv535fss7.1](https://doi.org/10.17632/fpv535fss7.1)

---

## ✍️ Author
Developed by **Jeong Seung Ju (KMOU, Dept. of AI Engineering)**  
For research on CIMT segmentation using U-Net variants.

---

✅ *This project is a clean, self-contained Python-only implementation ready for GitHub release.*

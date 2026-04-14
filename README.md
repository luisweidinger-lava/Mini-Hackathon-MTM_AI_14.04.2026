# FloodNet — Flood Damage Semantic Segmentation
### Mini Hackathon | MTM AI | 14.04.2026

> **Task:** Label every pixel in a drone photograph taken after Hurricane Harvey (2017) into one of 10 classes: flooded buildings, water, roads, trees, vehicles, pools, grass and more.

---

## Repository Structure

```
├── eda.ipynb          # Exploratory Data Analysis — run locally in VS Code
├── train.ipynb        # Model training — run in Google Colab (T4 GPU)
├── PLAN.md            # Original project plan and dataset documentation
└── README.md          # This file
```

---

## Quick Start

### EDA (local, no GPU needed)
```bash
# 1. Create virtual environment
python3 -m venv venv
venv/bin/pip install pillow numpy matplotlib seaborn scipy ipykernel

# 2. Register kernel for VS Code
venv/bin/python3 -m ipykernel install --user --name floodnet-eda --display-name "FloodNet EDA (Python 3.9)"

# 3. Open eda.ipynb in VS Code and select "FloodNet EDA (Python 3.9)" kernel
```

### Training (Google Colab T4 GPU)
Upload `train.ipynb` to Colab, set runtime to T4 GPU, run all cells.

---

## Dataset

**FloodNet Supervised v1.0** — Drone imagery after Hurricane Harvey, Texas, 2017.

| Split | Images | % of total |
|---|---|---|
| Train | 1,445 | 61.7% |
| Val | 450 | 19.2% |
| Test | 448 | 19.1% |
| **Total** | **2,343** | |

### 10 Classes

| ID | Class | Pixel % | Images containing it | Colour |
|---|---|---|---|---|
| 0 | Background | ~1.7% | 6.8% | Black |
| 1 | **Bldg Flooded** | ~1.8% | **10.3%** | Red |
| 2 | Bldg Non-Flooded | ~3.0% | 37.4% | Green |
| 3 | Road Flooded | ~3.1% | 11.2% | Orange |
| 4 | Road Non-Flooded | ~5.4% | 49.2% | Grey |
| 5 | Water | ~5.1% | 46.2% | Blue |
| 6 | Tree | ~17.5% | 80.0% | Forest green |
| 7 | Vehicle | ~0.2% | 34.3% | Yellow |
| 8 | Pool | ~0.2% | 22.9% | Cyan |
| 9 | Grass | ~35.5% | 92.1% | Light green |

---

## EDA — Key Findings

### Q1: Dataset size
- Train: 1,445 | Val: 450 | Test: 448 | Total: 2,343
- All image-mask pairs are matched — no missing labels.

### Q2: Image dimensions
- Variable — ~3000×4000 px (~12 MP).
- Must resize to 512×512 before training.
- NEAREST interpolation on masks (no blending).

### Q3: Class imbalance
- **SEVERE.** Grass/Tree/Background dominate (>60% combined).
- Bldg Flooded: ~2–3% | Vehicle: <0.5% | Pool: <0.5%
- Imbalance ratio: ~100–500× between dominant and rare classes.

### Q4: How to fix imbalance
- **(a)** Inverse-frequency class weights in CrossEntropyLoss
- **(b)** DiceLoss — optimises region overlap, handles rare classes
- **(c)** Combined: `DiceLoss + CrossEntropyLoss(weight=weights_tensor)`

### Q5: Evaluation metric
- mIoU (mean Intersection over Union) across 10 classes.
- Pixel accuracy is misleading under imbalance — mIoU is fair.
- Key headline: **IoU for class 1 (Bldg Flooded)**.

### Q6: Why transfer learning?
- 1,445 images is small. EfficientNet-B4 on ImageNet already knows edges, textures, shapes.
- Fine-tuning beats training from scratch and prevents overfitting.

### Q7: Why U-Net?
- Skip connections pass spatial detail from encoder → decoder.
- Preserves sharp boundaries — critical for small objects like vehicles and pools.

### Q8: Augmentations and why
- Flip H/V + Rotate90: drone photos have no fixed orientation.
- BrightnessContrast: vary lighting at inference time.
- GaussianBlur: camera blur robustness.
- Applied **identically** to image AND mask (same random seed).

### Q9: Baseline to beat
- Predict all pixels as dominant class → ~30% pixel accuracy, ~3% mIoU, 0% on flooded buildings.
- Our model must beat this.

### Q10: Data quality
- No missing files. Class 0 (Background) marks unannotated edges.
- All masks use values 0–9 only — no corrupted labels found.

### Problems handled in training

| Finding | Impact | Fix |
|---|---|---|
| Grass 175× more common than Vehicle/Pool | Model ignores rare classes | Inverse-frequency class weights |
| Bldg Flooded only 149 train images (10.3%) | Model rarely sees flooding | Weighted sampler: 3× oversample flood images |
| Background (class 0) = image edge artifact | Wastes model capacity | Set background weight = 0 in loss |
| Vehicle + Pool = 0.2% pixels each | Will score low IoU regardless | Accept this, report honestly |

---

## Technical Briefing

### 1. Task
For every single pixel in a drone photograph, the model outputs one of 10 labels. This is **semantic segmentation** — harder than image classification because a 512×512 image has 262,144 pixels each needing its own label. The model must simultaneously understand global context and fine-grained local boundaries.

### 2. Data Pipeline
- `FloodNetDataset` extends PyTorch's `Dataset` class
- Images loaded as NumPy arrays, transforms applied identically to image and mask
- Resize to 512×512 (not crop) — preserves full scene content
- NEAREST interpolation on masks prevents boundary corruption
- Validation uses Resize + Normalize only (no augmentation)

### 3. Weighted Sampler
`WeightedRandomSampler` gives flood images 3× draw probability. Without this, only 10.3% of each epoch involves the most important class. With 3×, flood images appear ~27% of the time.

### 4. Model Architecture — U-Net

```
Input (512×512×3)
    ↓
ENCODER: [512] → [256] → [128] → [64] → [32]   (ResNet50, pretrained ImageNet)
    ↓
BOTTLENECK
    ↓
DECODER: [32] → [64] → [128] → [256] → [512]   (skip connections from encoder)
    ↓
Output mask (512×512×10)
```

**Why U-Net over alternatives:**
| Architecture | Why not used |
|---|---|
| DeepLab v3+ | Slower, marginal gain at 512px |
| SegFormer | Needs far more data; overkill for 1,445 images |
| FCN | No skip connections → poor boundary accuracy |

### 5. Encoder — ResNet-50 (ImageNet pretrained)
- 50-layer CNN pretrained on 1.2M ImageNet photos
- Already detects edges, textures, shapes — fine-tuned on FloodNet
- Transfer learning is essential with only 1,445 training images

### 6. Loss Function

```python
def combined_loss(pred, target):
    return dice_loss(pred, target) + ce_loss(pred, target)
```

- **CrossEntropyLoss**: stable per-pixel gradients, fast convergence
- **DiceLoss**: optimises region overlap directly, handles class imbalance
- **Class weights**: rare classes weighted higher; Background weight = 0 (artifact)

```python
raw_weights = torch.tensor([
    0.0,  # Background  — excluded (image edge artifact)
    4.0,  # Bldg Flooded  — most important
    2.5,  # Bldg Non-Flooded
    3.5,  # Road Flooded
    1.5,  # Road Non-Flooded
    2.0,  # Water
    0.5,  # Tree
    4.5,  # Vehicle — 0.2% of pixels
    4.5,  # Pool    — 0.2% of pixels
    0.3,  # Grass   — 35.5% of pixels
])
```

### 7. Optimiser and Scheduler
- **AdamW**: fixes weight decay bug in Adam — cleaner regularisation for fine-tuning
- **OneCycleLR**: warm-up for first 10% of training, then cosine annealing to near zero
- LR = 3e-4, weight_decay = 1e-4

### 8. Evaluation Metric — mIoU

```
IoU(c) = TP(c) / (TP(c) + FP(c) + FN(c))
mIoU   = mean across all 9 non-background classes
```

Pixel accuracy is useless here (predicting all Grass = 35% accuracy, 0% on flooded buildings). mIoU weights every class equally.

**NaN in per-class output** = class absent from both predictions and ground truth in that batch.

### 9. Known Limitations

| Limitation | Detail |
|---|---|
| Vehicle + Pool IoU will be low | Only 0.2% of pixels — inherent dataset constraint |
| 7–20 epochs insufficient | Model still improving — longer run needed post-hackathon |
| No differential LR | Encoder and decoder trained at same rate — simplification |
| No test-time augmentation | TTA would add ~2–4 mIoU points at zero training cost |

### 10. Expected Results (7–20 epochs)

| Class | Expected IoU |
|---|---|
| Bldg Flooded | 0.25 – 0.45 |
| Bldg Non-Flooded | 0.40 – 0.60 |
| Road Flooded | 0.30 – 0.50 |
| Water | 0.50 – 0.70 |
| Tree + Grass | 0.60 – 0.80 |
| Vehicle + Pool | 0.05 – 0.25 |
| **mIoU** | **0.35 – 0.50** |

Baseline (predict all Grass): ~3% mIoU.

---

## Model Config

```python
IMG_SIZE   = 512
BATCH_SIZE = 8
NUM_EPOCHS = 20        # 7 for quick run
LR         = 3e-4
NUM_CLASSES= 10
ENCODER    = "resnet50"
DEVICE     = "cuda" (Colab T4) / "mps" (Apple M-series)
```

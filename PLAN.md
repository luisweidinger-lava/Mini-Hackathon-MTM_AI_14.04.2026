# Hackathon Plan: Flood Image Segmentation — FloodNet

## What are we building?
A deep learning model that looks at an aerial (drone) photo taken after a flood and colours every pixel by what it shows:
flooded building, non-flooded building, water, road, tree, etc.

This is called **semantic segmentation** — "semantic" means meaningful categories, "segmentation" means dividing the image pixel by pixel.

---

## Dataset: FloodNet Supervised v1.0
GitHub: https://github.com/BinaLab/FloodNet-Supervised_v1.0
Download (Google Drive): https://drive.google.com/drive/folders/1sZZMJkbqJNbHgebKvHzcXYZHJd6ss4tH

### What's in it?
| Detail | Value |
|--------|-------|
| Total images | 2,343 drone (UAV) photos |
| Training set | 1,445 images |
| Validation set | 450 images |
| Test set | 448 images |
| Image type | High-resolution colour JPEG (RGB) |
| Mask type | PNG where each pixel = a number 0–9 |
| Captured after | Hurricane Harvey (Texas, 2017) |
| Camera | DJI Mavic Pro drone |
| Ground resolution | 1.5 cm per pixel (extremely detailed) |

### Folder structure (after download)
```
FloodNet-Supervised_v1.0/
├── train/
│   ├── train-org-img/      ← the actual drone photos (.jpg)
│   └── train-label-img/    ← the masks (.png, pixel values 0–9)
├── val/
│   ├── val-org-img/
│   └── val-label-img/
└── test/
    ├── test-org-img/
    └── test-label-img/
```

### The 10 classes (what each pixel number means)
| Number | Class | What it looks like from above |
|--------|-------|-------------------------------|
| 0 | Background | Sky edges, unlabelled areas |
| 1 | Building Flooded | Building roof with water around/inside |
| 2 | Building Non-Flooded | Normal building roof |
| 3 | Road Flooded | Road covered in water |
| 4 | Road Non-Flooded | Normal dry road |
| 5 | Water | Rivers, standing flood water |
| 6 | Tree | Tree canopy |
| 7 | Vehicle | Cars, trucks |
| 8 | Pool | Swimming pools |
| 9 | Grass | Lawn, fields |

### Class imbalance (critical problem!)
The classes are NOT equally represented. Expected distribution (approximate):
- Grass (9) and Tree (6) dominate — they cover large areas
- Background (0) is also frequent
- Building Flooded (1) is RARE — maybe 1–3% of pixels
- Vehicle (7) and Pool (8) are the rarest — tiny objects

This means the model will naturally ignore rare classes unless we fix it with class weights.

---

## Phase 0: Colab Setup — Step by Step

### What you need before starting
- A Google account (for Colab + Drive)
- The dataset downloaded to your Google Drive

### Cell 1 — Install all libraries
```python
!pip install segmentation-models-pytorch albumentations pillow scipy wandb
```

**What this does:**
- `segmentation-models-pytorch` — gives us ready-made U-Net and other segmentation models. Without this we'd have to code the model from scratch (hundreds of lines)
- `albumentations` — a fast library for image augmentation (flipping, rotating, changing brightness). Makes our small dataset feel bigger
- `pillow` — reads image files (JPG, PNG) into Python
- `scipy` — scientific tools; we use it to analyse mask boundaries
- `wandb` — "Weights & Biases", a free website that shows real-time training charts
- The `!` at the start means "run this as a terminal command, not Python code"

---

### Cell 2 — Connect Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/floodnet', exist_ok=True)
```

**What this does:**
- `drive.mount(...)` — links your Google Drive to the Colab virtual machine so files on Drive appear at `/content/drive/MyDrive/`
- Without this, any files you download disappear when the Colab session ends
- `os.makedirs(...)` — creates a folder called `floodnet` in your Drive to store everything
- `exist_ok=True` — don't crash if the folder already exists

---

### Cell 3 — Download the dataset
```python
# The dataset is hosted on Google Drive by the FloodNet authors
# Download it manually:
# 1. Open this link: https://drive.google.com/drive/folders/1sZZMJkbqJNbHgebKvHzcXYZHJd6ss4tH
# 2. Download the ZIP to your computer
# 3. Upload it to /content/drive/MyDrive/floodnet/ via the Colab file browser

# Then unzip it:
%cd /content/drive/MyDrive/floodnet
!unzip -q FloodNet-Supervised_v1.0.zip
!ls   # should show the train/ val/ test/ folders
```

**What this does:**
- `%cd` — change directory (move into the floodnet folder)
- `!unzip -q` — extract the ZIP file; `-q` means quiet (don't print every filename)
- `!ls` — list files to verify the unzip worked

---

### Cell 4 — Define your paths
```python
import os

BASE     = "/content/drive/MyDrive/floodnet/FloodNet-Supervised_v1.0"
TRAIN_IMG  = os.path.join(BASE, "train/train-org-img")
TRAIN_MASK = os.path.join(BASE, "train/train-label-img")
VAL_IMG    = os.path.join(BASE, "val/val-org-img")
VAL_MASK   = os.path.join(BASE, "val/val-label-img")
TEST_IMG   = os.path.join(BASE, "test/test-org-img")
TEST_MASK  = os.path.join(BASE, "test/test-label-img")

train_imgs  = sorted(os.listdir(TRAIN_IMG))
train_masks = sorted(os.listdir(TRAIN_MASK))
print(f"Train images : {len(train_imgs)}")    # expect ~1445
print(f"Train masks  : {len(train_masks)}")   # expect ~1445
```

**What this does:**
- `os.path.join(...)` — safely builds a file path by combining folder names with `/`
- `os.listdir(...)` — lists all files in a folder
- `sorted(...)` — sorts alphabetically so images[0] always matches masks[0] (same photo and its label)

---

### Cell 5 — Sanity check (one image + one mask)
```python
import numpy as np
from PIL import Image

img  = Image.open(os.path.join(TRAIN_IMG, train_imgs[0]))
mask = Image.open(os.path.join(TRAIN_MASK, train_masks[0]))

print(f"Image size : {img.size}")        # e.g. (3000, 4000) width x height
print(f"Image mode : {img.mode}")        # RGB = 3 colour channels
print(f"Mask size  : {mask.size}")       # should match image size
print(f"Mask mode  : {mask.mode}")       # L = grayscale (single channel)

mask_arr = np.array(mask)
print(f"Mask unique values: {np.unique(mask_arr)}")  # should be subset of 0–9
print(f"Mask dtype: {mask_arr.dtype}")               # uint8 (integers 0–255)
```

**What this does:**
- `Image.open(...)` — loads the file into memory as a PIL Image object
- `img.size` — returns (width, height) in pixels
- `img.mode` — `RGB` means 3 channels (red, green, blue); `L` means grayscale (1 channel used for masks)
- `np.array(mask)` — converts the PIL image into a 2D NumPy array of integers
- `np.unique(...)` — finds all distinct values in the array; for a mask these should be 0–9
- **Key check**: if you see values like 255 in the mask, that means unlabelled pixels — handle them

---

### Cell 6 — Import all libraries at the top
```python
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from collections import defaultdict
from scipy.ndimage import binary_erosion
import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")   # should be True on Colab T4
```

**What this does:**
- `matplotlib.pyplot` — draws charts and plots
- `matplotlib.patches` — lets us create a colour legend for the segmentation maps
- `collections.defaultdict` — a dictionary that auto-creates missing keys (useful for counting)
- `torch.cuda.is_available()` — checks if a GPU is available; training on CPU would be ~50× slower

---

## Phase 1: EDA — Exploratory Data Analysis

EDA means systematically understanding your data BEFORE training anything.
Goal: produce charts and tables that explain the dataset to your audience.

---

### EDA Step 1 — Dataset split overview

```python
splits = {
    "train": (TRAIN_IMG, TRAIN_MASK),
    "val":   (VAL_IMG,   VAL_MASK),
    "test":  (TEST_IMG,  TEST_MASK),
}

for split, (img_dir, mask_dir) in splits.items():
    n_imgs  = len(os.listdir(img_dir))
    n_masks = len(os.listdir(mask_dir))
    print(f"{split:5s} | images: {n_imgs:4d} | masks: {n_masks:4d} | match: {n_imgs == n_masks}")
```

**What this does:**
- Loops through each of the 3 splits (train / val / test)
- Counts files in the image folder and mask folder
- Checks they match — if not, the dataset download is broken
- Expected output: train=1445, val=450, test=448

---

### EDA Step 2 — Image size check

```python
sizes = set()
for fname in train_imgs[:50]:    # check first 50 (checking all 1445 takes 2 min)
    img = Image.open(os.path.join(TRAIN_IMG, fname))
    sizes.add(img.size)

print(f"Unique sizes found: {sizes}")
# FloodNet images are NOT all the same size — they vary by drone altitude
```

**What this does:**
- Opens 50 images and collects their (width, height) into a `set` (sets only store unique values)
- If the set has 1 entry → all images are the same size (easy)
- If multiple → images vary in size. This matters because the model needs fixed-size inputs, so we must resize in the data pipeline
- FloodNet images are typically around 3000×4000 px but may vary

---

### EDA Step 3 — Class pixel distribution (the most important EDA step)

```python
CLASS_NAMES = [
    "Background",        # 0
    "Bldg Flooded",      # 1
    "Bldg Non-Flooded",  # 2
    "Road Flooded",      # 3
    "Road Non-Flooded",  # 4
    "Water",             # 5
    "Tree",              # 6
    "Vehicle",           # 7
    "Pool",              # 8
    "Grass",             # 9
]

pixel_counts = np.zeros(10, dtype=np.int64)

for fname in train_masks:
    mask = np.array(Image.open(os.path.join(TRAIN_MASK, fname)))
    for c in range(10):
        pixel_counts[c] += (mask == c).sum()

total_pixels = pixel_counts.sum()
pixel_pct    = pixel_counts / total_pixels * 100

# Print table
print(f"{'Class':<22} {'Pixels':>12} {'Percent':>8}")
print("-" * 44)
for i, (name, cnt, pct) in enumerate(zip(CLASS_NAMES, pixel_counts, pixel_pct)):
    print(f"{i} {name:<20} {cnt:>12,} {pct:>7.2f}%")

# Bar chart (log scale because imbalance is extreme)
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(CLASS_NAMES, pixel_pct, color='steelblue')
ax.set_yscale('log')     # log scale so rare classes are visible
ax.set_ylabel("% of total pixels (log scale)")
ax.set_title("FloodNet — Pixel-level class distribution (training set)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("eda_class_distribution.png", dpi=150)
plt.show()
```

**What this does:**
- Loops through ALL 1,445 training masks
- For each class (0–9), counts how many pixels belong to it using `(mask == c).sum()`
- Divides by total to get percentages
- Plots a bar chart with **log scale** — necessary because without it, rare classes (Vehicle=0.1%) would be invisible next to Grass (30%)
- **Key finding for presentation**: Building Flooded is likely <3% of pixels. This is why training fails without class weights

---

### EDA Step 4 — Per-image class presence (which images have flooded buildings?)

```python
has_flooded = []   # list of filenames that contain class 1

for fname in train_masks:
    mask = np.array(Image.open(os.path.join(TRAIN_MASK, fname)))
    if (mask == 1).any():      # .any() = True if at least one pixel is class 1
        has_flooded.append(fname)

pct_with_flood = len(has_flooded) / len(train_masks) * 100
print(f"Images containing flooded buildings : {len(has_flooded)}")
print(f"Percentage                          : {pct_with_flood:.1f}%")
```

**What this does:**
- `(mask == 1).any()` — checks if ANY pixel in the mask equals 1 (flooded building)
- Builds a list of images that actually show flooded buildings
- **Why this matters**: if only 30% of images have flooded buildings, you might want to oversample those images during training

---

### EDA Step 5 — Visualise sample images with colour-coded masks

```python
# Define a colour for each class
COLORMAP = np.array([
    [0,   0,   0  ],   # 0 Background — black
    [255, 0,   0  ],   # 1 Building Flooded — red
    [0,   128, 0  ],   # 2 Building Non-Flooded — green
    [255, 165, 0  ],   # 3 Road Flooded — orange
    [128, 128, 128],   # 4 Road Non-Flooded — grey
    [0,   0,   255],   # 5 Water — blue
    [34,  139, 34 ],   # 6 Tree — forest green
    [255, 255, 0  ],   # 7 Vehicle — yellow
    [0,   255, 255],   # 8 Pool — cyan
    [144, 238, 144],   # 9 Grass — light green
], dtype=np.uint8)

def mask_to_rgb(mask_array):
    """Convert a 2D mask (values 0–9) to a 3-channel RGB colour image."""
    rgb = COLORMAP[mask_array]   # fancy indexing: each pixel value → its colour
    return rgb

# Plot 6 examples: original image | colour mask | overlay
n_show = 6
fig, axes = plt.subplots(n_show, 3, figsize=(15, n_show * 4))

for row, fname in enumerate(train_imgs[:n_show]):
    mask_fname = fname.replace(".jpg", "_lab.png")   # adjust if naming differs
    img  = np.array(Image.open(os.path.join(TRAIN_IMG, fname)))
    mask = np.array(Image.open(os.path.join(TRAIN_MASK, mask_fname)))

    rgb_mask = mask_to_rgb(mask)

    # Resize for display (full res is huge)
    from PIL import Image as PILImage
    img_small  = np.array(PILImage.fromarray(img).resize((512, 512)))
    mask_small = np.array(PILImage.fromarray(mask, mode='L').resize((512, 512), PILImage.NEAREST))
    rgb_small  = mask_to_rgb(mask_small)

    overlay = (img_small * 0.6 + rgb_small * 0.4).astype(np.uint8)

    axes[row, 0].imshow(img_small);  axes[row, 0].set_title("Original image"); axes[row, 0].axis("off")
    axes[row, 1].imshow(rgb_small);  axes[row, 1].set_title("Mask (colour coded)"); axes[row, 1].axis("off")
    axes[row, 2].imshow(overlay);    axes[row, 2].set_title("Overlay"); axes[row, 2].axis("off")

# Add legend
legend_patches = [mpatches.Patch(color=COLORMAP[i]/255, label=f"{i}: {CLASS_NAMES[i]}") for i in range(10)]
fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=9)
plt.suptitle("FloodNet — Sample images with segmentation masks", y=1.01)
plt.tight_layout()
plt.savefig("eda_samples.png", dpi=150, bbox_inches='tight')
plt.show()
```

**What this does:**
- `COLORMAP` — a lookup table: pixel value 0 → black, 1 → red, etc.
- `COLORMAP[mask_array]` — NumPy "fancy indexing": each pixel value is used as an index to get its colour. Converts a 2D array of numbers into a 3D RGB image in one line
- `.resize(..., PILImage.NEAREST)` — resizes without blending pixels. NEAREST is critical for masks — blending would create values like 1.5 which don't exist as a class
- `img * 0.6 + rgb_mask * 0.4` — blends the photo with the colour mask at 60/40 ratio to create a see-through overlay
- `mpatches.Patch(...)` — creates a coloured square for the legend

---

### EDA Step 6 — RGB channel statistics (needed for normalisation)

```python
channel_sums   = np.zeros(3)
channel_sq_sum = np.zeros(3)
n_pixels = 0

for fname in train_imgs:
    img = np.array(Image.open(os.path.join(TRAIN_IMG, fname))) / 255.0
    channel_sums   += img.mean(axis=(0, 1))
    channel_sq_sum += (img ** 2).mean(axis=(0, 1))
    n_pixels       += 1

mean = channel_sums / n_pixels
std  = np.sqrt(channel_sq_sum / n_pixels - mean ** 2)
print(f"Dataset RGB mean : {mean}")   # e.g. [0.47, 0.46, 0.38]
print(f"Dataset RGB std  : {std}")    # e.g. [0.21, 0.20, 0.19]
# Save these values — you'll use them in transforms.Normalize(mean, std)
```

**What this does:**
- Divides pixels by 255 to scale them from [0–255] to [0.0–1.0] (neural networks work best with small numbers)
- Calculates the average brightness of each colour channel (R, G, B) across the whole dataset
- Calculates standard deviation (how spread out the values are)
- **Why needed**: normalisation (subtracting mean, dividing by std) makes training faster and more stable. Without it, the model starts far from good weights

---

### EDA Step 7 — Class co-occurrence (which classes appear together?)

```python
# For each pair of classes, count how many images contain BOTH
co_occurrence = np.zeros((10, 10), dtype=int)

for fname in train_masks:
    mask = np.array(Image.open(os.path.join(TRAIN_MASK, fname)))
    present = [c for c in range(10) if (mask == c).any()]
    for i in present:
        for j in present:
            co_occurrence[i, j] += 1

import seaborn as sns
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(co_occurrence, annot=True, fmt='d', xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES, cmap='Blues', ax=ax)
ax.set_title("Class co-occurrence: how many images contain both classes")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("eda_cooccurrence.png", dpi=150)
plt.show()
```

**What this does:**
- For each mask, builds a list of which classes are present
- For every pair of classes that appear together, increments a counter
- Displays as a heatmap: dark blue = these two classes always appear together, white = never together
- **Insight for presentation**: Water and Building Flooded should co-occur often (flooded areas). If they don't, the dataset has labelling inconsistencies

---

### EDA Step 8 — Compute class weights (for fixing imbalance)

```python
# Inverse frequency weighting: rare classes get higher weight
# So the model pays more attention to them during training

class_weights = 1.0 / (pixel_counts + 1)    # +1 to avoid division by zero
class_weights = class_weights / class_weights.sum() * 10   # normalise to sum=10

print("Class weights for loss function:")
for i, (name, w) in enumerate(zip(CLASS_NAMES, class_weights)):
    print(f"  {i} {name:<22} weight = {w:.4f}")

# Convert to PyTorch tensor (needed for training)
import torch
weights_tensor = torch.FloatTensor(class_weights)
```

**What this does:**
- `1.0 / pixel_counts` — rare classes (few pixels) get a high weight; common classes get a low weight
- This tells the loss function: "I care 10× more about getting a flooded building pixel right than a grass pixel"
- Without this, the model learns to predict all-grass and gets 30% accuracy "for free" without ever learning flooded buildings
- The resulting `weights_tensor` will be passed directly to the loss function during training

---

### EDA Summary Table
| Metric | Expected value |
|--------|---------------|
| Training images | 1,445 |
| Validation images | 450 |
| Test images | 448 |
| Image format | JPG, variable size (~3000×4000 px) |
| Mask format | PNG, pixel values 0–9 |
| Most common class | Grass / Tree / Background |
| Rarest classes | Vehicle, Pool, Building Flooded |
| % images with flooded buildings | ~30–50% |

---

## Phase 2: Data Pipeline

After EDA, you build the code that feeds images to the model.

### Key insight from EDA: images are huge
At 3000×4000 px, each image is 36 MP. The model can't take that as input — it would run out of GPU memory. Solution: **resize** all images to 512×512 or 1024×1024 before training.

```python
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FloodNetDataset(Dataset):
    """
    PyTorch Dataset for FloodNet segmentation.
    Loads one image + its mask, applies augmentations, returns tensors.
    """
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        # PyTorch calls this to know how many samples exist
        return len(self.img_files)

    def __getitem__(self, idx):
        # PyTorch calls this with an index to get one sample
        img_fname  = self.img_files[idx]
        mask_fname = img_fname.replace(".jpg", "_lab.png")  # match naming

        img  = np.array(Image.open(os.path.join(self.img_dir,  img_fname)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, mask_fname)))

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img  = augmented["image"]   # now a PyTorch tensor [3, H, W]
            mask = augmented["mask"]    # now a PyTorch tensor [H, W]

        return img, mask.long()         # .long() = 64-bit integer (needed for loss function)
```

**What this does:**
- `Dataset` is a PyTorch base class — any class that inherits it and defines `__len__` and `__getitem__` works with PyTorch's DataLoader automatically
- `__len__` — tells PyTorch how many samples there are
- `__getitem__(idx)` — loads and returns one image-mask pair given an index number
- `.convert("RGB")` — forces 3 channels in case any image is accidentally saved as grayscale
- `mask.long()` — converts mask to 64-bit integer. CrossEntropyLoss requires integer class indices, not floats

### Augmentations
```python
IMG_SIZE = 512   # resize everything to 512×512

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),               # shrink huge drone images to 512×512
    A.HorizontalFlip(p=0.5),                    # randomly mirror left-right (50% chance)
    A.VerticalFlip(p=0.5),                      # randomly mirror top-bottom
    A.RandomRotate90(p=0.5),                    # randomly rotate 0/90/180/270 degrees
    A.RandomBrightnessContrast(p=0.3),          # vary brightness & contrast
    A.GaussianBlur(p=0.2),                      # slight blur (simulates camera shake)
    A.Normalize(mean=mean, std=std),             # subtract dataset mean, divide by std
    ToTensorV2(),                               # convert numpy array to PyTorch tensor
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=mean, std=std),
    ToTensorV2(),
])
```

**What each augmentation does:**
- `Resize` — makes all images the same size (required: model input must be fixed size)
- `HorizontalFlip` / `VerticalFlip` — flips the image. A flooded building is still a flooded building if mirrored. Doubles/quadruples effective dataset size
- `RandomRotate90` — rotates by 90° increments. Drone photos have no "up" — valid from any angle
- `RandomBrightnessContrast` — simulates different lighting conditions
- `GaussianBlur` — simulates slight camera blur; makes model more robust
- `Normalize` — subtracts mean and divides by std (from EDA Step 6). Centres data around 0
- `ToTensorV2` — converts numpy [H,W,C] to PyTorch tensor [C,H,W] and to float32

---

## Phase 3: Model

```python
import segmentation_models_pytorch as smp
import torch.nn as nn

model = smp.Unet(
    encoder_name    = "efficientnet-b4",  # pretrained feature extractor
    encoder_weights = "imagenet",          # start with weights learned on 1M photos
    in_channels     = 3,                  # RGB input
    classes         = 10,                 # 10 output classes
)

loss_fn = smp.losses.DiceLoss(mode='multiclass') + nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

**What the model does:**
U-Net has two halves:
1. **Encoder** (left side of the U) — repeatedly shrinks the image while extracting features. Like zooming out to understand context. EfficientNet-B4 does this step, pretrained on ImageNet so it already knows how to detect edges, textures, shapes
2. **Decoder** (right side) — repeatedly enlarges back to original size while predicting class labels. Uses "skip connections" that borrow detail from the encoder to keep precise boundaries

**Why DiceLoss + CrossEntropy?**
- `CrossEntropyLoss` — standard classification loss. Penalises wrong class predictions. Fast and stable
- `DiceLoss` — measures overlap between predicted and true regions. Handles class imbalance better. Works well for rare classes like Building Flooded
- Combining both gives the benefits of both

---

## Phase 4–6 Overview
| Phase | Task | Output |
|-------|------|--------|
| 4 — Train | 30–50 epochs, monitor val mIoU via wandb | Saved model checkpoint |
| 5 — Evaluate | Per-class IoU on test set, confusion matrix | Results table |
| 6 — Present | EDA charts + training curves + visual predictions | Slides |

**Key metric: mIoU** (mean Intersection over Union)
- For each class: IoU = (pixels correctly predicted as this class) / (all pixels predicted OR actually this class)
- Ranges 0–1; higher = better
- Average across all 10 classes = mIoU
- Focus on **IoU for class 1** (Building Flooded) — this is your headline number

---

## Files to create
| File | Purpose |
|------|---------|
| `eda.ipynb` | All EDA steps above |
| `dataset.py` | FloodNetDataset class + augmentations |
| `model.py` | Model, loss, optimiser |
| `train.ipynb` | Training loop in Colab |
| `evaluate.py` | Metrics + visualisation |

---

## Sources
- [FloodNet GitHub (BinaLab)](https://github.com/BinaLab/FloodNet-Supervised_v1.0)
- [FloodNet Challenge 2021](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021)
- [FloodNet paper — arXiv](https://arxiv.org/abs/2012.02951)
- [Dataset Ninja — FloodNet stats](https://datasetninja.com/floodnet)
- [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)

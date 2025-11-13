# Project 3 - Image Segmentation Implementation

## Overview
This project implements image segmentation for Phase Contrast (PhC) microscopy data. The implementation includes multiple neural network architectures and loss functions specifically designed for semantic segmentation tasks.

## Implemented Components

### 1. Neural Network Models

#### UNet (`lib/model/UNetModel.py`)
- **Classic U-Net architecture** with encoder-decoder structure and skip connections
- Features:
  - Encoder: 4 downsampling blocks with max pooling
  - Decoder: 4 upsampling blocks with bilinear interpolation or transpose convolution
  - Skip connections preserve fine-grained details
  - Batch normalization for stable training
  - Input channels: 3 (RGB), Output channels: 1 (binary mask)

#### UNet2 (`lib/model/UNetModel.py`)
- **Lighter variant of U-Net** with fewer parameters
- Features:
  - Reduced number of channels (32-512 vs 64-1024)
  - Added dropout (0.1) for regularization
  - Same architectural principle as UNet but more efficient
  - Suitable for smaller datasets or limited computational resources

#### DilatedNet (`lib/model/DilatedNetModel.py`)
- **Dilated convolution network** for maintaining spatial resolution
- Features:
  - Uses dilated convolutions with rates: 2, 4, 8
  - Preserves spatial dimensions throughout the network
  - Captures multi-scale context without pooling
  - Simpler architecture compared to U-Net

### 2. Loss Functions

#### Dice Loss (`lib/losses.py`)
- **Dice coefficient-based loss** for handling class imbalance
- Formula: `Loss = 1 - (2 * intersection + smooth) / (sum(pred) + sum(true) + smooth)`
- Advantages:
  - Directly optimizes the Dice coefficient metric
  - Robust to class imbalance
  - Works well for binary segmentation

#### Focal Loss (`lib/losses.py`)
- **Focal loss** for addressing hard negative mining
- Formula: `FL = -α(1-p_t)^γ * log(p_t)`
- Parameters:
  - α (alpha): 1.0 - balancing factor
  - γ (gamma): 2.0 - focusing parameter
- Advantages:
  - Focuses learning on hard examples
  - Reduces the impact of easy negatives
  - Effective for imbalanced datasets

#### BCE Loss with Total Variation (`lib/losses.py`)
- **Binary Cross Entropy + Total Variation regularization**
- Combines standard BCE loss with smoothness penalty
- Total variation encourages smooth segmentation boundaries
- Regularization weight: 0.1

### 3. Dataset

#### PhC Dataset (`lib/dataset/PhCDataset.py`)
- **Phase Contrast microscopy dataset loader**
- Features:
  - Supports train/test splits
  - Loads RGB images and binary masks
  - Configurable transforms (resize, normalization)
  - Fixed path issue to use relative paths

## Usage Examples

### Basic Training
```python
from lib.model.UNetModel import UNet
from lib.losses import DiceLoss

model = UNet(n_channels=3, n_classes=1)
loss_fn = DiceLoss()
# ... training loop
```

### Testing Different Configurations
```python
# Run the test script
python test_models.py

# Run training with different models
python train.py  # Uses EncDec by default
```

### Model Selection Guide

| Model | Parameters | Memory | Best For |
|-------|------------|--------|----------|
| UNet | ~31M | High | High accuracy, sufficient data |
| UNet2 | ~7M | Medium | Limited resources, regularization |
| DilatedNet | ~1M | Low | Fast inference, simple cases |
| EncDec | ~260K | Very Low | Baseline, quick experiments |

### Loss Function Selection Guide

| Loss Function | Best For | Characteristics |
|---------------|----------|----------------|
| Dice Loss | Imbalanced classes | Direct metric optimization |
| Focal Loss | Hard examples | Focuses on difficult pixels |
| BCE + TV | Smooth boundaries | Regularized segmentation |
| BCE Loss | Balanced data | Standard baseline |

## Data Structure
```
phc_data/
├── train/
│   ├── images/  (151 RGB images)
│   └── labels/  (151 binary masks)
└── test/
    ├── images/  (500 RGB images)
    └── labels/  (500 binary masks)
```

## Key Implementation Details

1. **Input/Output**: RGB images (3 channels) → Binary masks (1 channel)
2. **Image size**: Resized to 128×128 for training efficiency
3. **Activation**: No final activation in models (raw logits)
4. **Loss computation**: Sigmoid applied within loss functions
5. **Metrics**: Loss values track training progress

## Training Results
- EncDec model achieves ~0.105 BCE loss after 20 epochs
- All models successfully handle 151 training images
- Memory efficient implementation supports batch processing

## Files Modified/Created
- `lib/losses.py` - Implemented DiceLoss and FocalLoss
- `lib/model/UNetModel.py` - Implemented UNet and UNet2
- `lib/model/DilatedNetModel.py` - Implemented DilatedNet
- `lib/dataset/PhCDataset.py` - Fixed path issues
- `test_models.py` - Comprehensive testing script

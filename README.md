# 🚗 ConvLSTM Turn Prediction Model

A deep learning pipeline for asisistive navigation that predicts maneuvers from video sequences using a ConvLSTM architecture. This model takes 3-second video clips and outputs turn predictions (Front, Left, Right) with user intent integration.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)

## 🎯 Project Overview

This repository implements a Convolutional LSTM network for real-time user maneuver prediction. The model:
- Processes 3-second video clips at 10 FPS (30 frames)
- Integrates optional intent signals
- Predicts three maneuver classes: Front (straight), Left turn, Right turn
- Operates at 128×128 resolution for efficient inference

### Key Features
- **Temporal Understanding**: LSTM cells capture motion patterns across time
- **Spatial Feature Extraction**: Convolutional layers detect road features
- **Intent Integration**: 3-channel driver intent signal 
- **Class Imbalance Handling**: Weighted loss function for better generalization

## 📂 Project Structure

```text
ConvLSTM/
├── models/
│   └── conv_lstm_classifier.py     # Neural network architecture
│
├── notebooks/                      # Development & experimentation
│   ├── prototype_1.ipynb           # Intent-enabled version (6 channels)
│   └── prototype_2.ipynb           # Baseline version (3 channels)
│
├── dataset.py                      # PyTorch Dataset implementation
├── utils.py                        # Configuration & hyperparameters
├── train.py                        # Training script
├── tester.py                       # Evaluation script
├── best_convlstm.pth               # Trained model weights (generated)
└── README.md
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install opencv-python pandas numpy scikit-learn scipy
pip install tqdm pillow jupyter
```

## 📊 Dataset Preparation

### Expected Directory Structure
```text
Dataset_Folder/
├── videos/                         # 3-second video clips
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
│
└── labels/                         # Frame-level annotations
    ├── video_001_labels.csv        # Standard naming
    ├── video_002_labels_A.csv      # Alternative naming supported
    └── ...
```

### Video Specifications
- **Duration**: Exactly 3 seconds
- **Frame Rate**: 10 FPS (30 frames total)
- **Resolution**: 128×128 pixels
- **Format**: MP4

### Label CSV Format
Each CSV must contain a `label_id_corrected` column (or fallback to last column):
- `0`: Front/Straight
- `1`: Left turn
- `2`: Right turn

The model uses the last 24 frames (last second) for label determination.

## ⚙️ Configuration

### Update Paths (utils.py)
```python
VIDEO_DIR = r'C:\path\to\your\videos'
LABEL_DIR = r'C:\path\to\your\labels'
```

### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `HEIGHT` | 128 | Frame height |
| `WIDTH` | 128 | Frame width |
| `CHANNELS` | 6 (with intent) / 3 (baseline) | Input channels |
| `FPS` | 10 | Frames per second |
| `DURATION` | 3 | Clip duration (seconds) |
| `SEQ_LEN` | 30 | Total frames per sequence |

### Training Configuration (train.py)
```python
BATCH = 5                           # Batch size
NUM_EPOCHS = 20                     # Training epochs
LEARNING_RATE = 1e-4                # Adam optimizer learning rate
SEED = 8                            # Random seed for reproducibility
```

### Model Parameters
```python
PARAMS = {
    'input_dim': 6,                 # 3 RGB + 3 intent channels
    'hidden_dim': [64, 32],         # LSTM hidden dimensions per layer
    'kernel_size': (3, 3),          # Convolutional kernel size
    'num_layers': 2,                # Number of LSTM layers
    'num_classes': 3                # Front, Left, Right
}
```

## 🚀 Usage

### 1. Training
Trains the model using 60/20/20 train/validation/test split:
```bash
python train.py
```

**Output:**
- `best_convlstm.pth`: Best model based on validation accuracy
- Console logs: Training loss, accuracy per epoch
- Intent positions: `val_intent_positions.npy` (generated once)

### 2. Evaluation
Tests the trained model on the held-out test set:
```bash
python tester.py
```

**Output:**
- Performance metrics (accuracy, precision, recall, F1-score)
- Per-class classification report
- Latency measurements (avg ms per clip, FPS)
- `test_results.csv`: Detailed predictions

### 3. Notebooks (Optional)
For experimentation and visualization:
```bash
jupyter notebook notebooks/prototype_1.ipynb
```

**Notebook Variants:**
- `prototype_1.ipynb`: Full implementation with intent integration
- `prototype_2.ipynb`: Baseline model without intent

## 🧠 Model Architecture

### ConvLSTM Overview
The model combines spatial and temporal learning:

```
Input Video (30 frames)
    ↓
ConvLSTMCell Layer 1 (64 hidden dims)
    ↓
ConvLSTMCell Layer 2 (32 hidden dims)
    ↓
Final Hidden State (last frame)
    ↓
Fully Connected Layer
    ↓
Output (3 classes)
```

### Components

#### 1. ConvLSTMCell
- Processes one frame at a time
- Combines convolutional feature extraction with LSTM memory
- Maintains hidden state (h) and cell state (c)
- Uses 4 gates: input, forget, output, cell

#### 2. ConvLSTM Module
- Stacks multiple ConvLSTMCell layers
- Processes all 30 frames sequentially
- Returns final temporal representation

#### 3. ConvLSTMModel
- Wraps ConvLSTM + classification head
- Flattens final hidden state
- Linear layer maps to 3 class logits

### With Intent Integration

Hannahhh add ka here

## 📈 Results

### Sample Output (from test_results.csv)
```csv
Actual_Label,Predicted_Label,Actual_Text,Predicted_Text,Correct
2,0,Right,Front,False
0,0,Front,Front,True
1,0,Left,Front,False
```

### Metrics Provided
- **Overall Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Latency**: Average inference time per clip (ms)
- **Throughput**: Clips processed per second

### Performance Notes
Based on the provided test results...

## 🔍 Troubleshooting

### Common Issues

**1. Directory Not Found**
```
WARNING: Directory not found: [PATH]
```
→ Update `VIDEO_DIR` and `LABEL_DIR` in `utils.py`

**2. CSV File Mismatch**
```
WARNING: Label file not found for video [NAME]
```
→ Ensure every `.mp4` has a matching `_labels.csv` or `_labels_A.csv`

**3. CUDA Out of Memory**
→ Reduce `BATCH` size in `train.py`

**4. Model File Not Found**
```
Model file not found at best_convlstm.pth
```
→ Run `train.py` first to generate the model

## 🔮 Future Work

- [ ] Implement data augmentation (rotation, brightness, etc.)
- [ ] Explore attention mechanisms for better temporal modeling
- [ ] Add online learning capabilities for continuous adaptation
- [ ] Optimize for edge deployment (quantization, pruning)
- [ ] Multi-modal fusion (LiDAR, GPS, IMU)

---

**Note**: This is a research prototype. Real-world deployment requires extensive safety validation and testing.

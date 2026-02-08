# 🚗 ConvLSTM Turn Prediction Model

This repository contains the Deep Learning Pipeline for autonomous navigation. 
It is designed to work directly with the output of the Data Preparation & Labeling Pipeline, 
using a ConvLSTM architecture to predict vehicle maneuvers from 3-second video sequences.

# 📂 Project Structure

```text
Project_Root/
├── data_preparation/            # (The folder from your first pipeline)
│   ├── labels/                  # Generated MVO CSVs
│   └── videos/                  # Cleaned 128x128 10fps videos
│
├── models/
│   └── conv_lstm_classifier.py  # The Brain (Neural Network Architecture)
│
├── dataset.py                   # The Conveyor (3-second clip data loader)
├── utils.py                     # The Settings (Hyperparameters & Paths)
├── train.py                     # The Teacher (Learning & Optimization script)
└── tester.py                    # The Proctor (Final Evaluation script)
```

# 🚀 Workflow Execution Order
To ensure the AI learns correctly, follow these steps in order:
1. Pipeline Completion: Run your 8-step Data Preparation pipeline first. Ensure one has the .mp4 files in videos/ and matching .csv files in labels/.
2. Configuration (utils.py): Double-check that the VIDEO_DIR and LABEL_DIR paths are correct.
3. Training (train.py): Run this to begin the learning process. This script will generate the best_convlstm.pth model file.
4. Testing (tester.py): Run this to evaluate the trained model on unseen data and get a final accuracy score.

# ⚙️ Component Breakdown

## 🛠️ The Settings (utils.py)

This file centralizes all constants to ensure the AI "sees" the data the same way the pipeline "cleaned" it.

Prediction Logic
| Constant | Value | Description ID |
|----------|----------|----------|
| FPS | 10 | Matches the cleaner.py output
| Duration | 3s | Matches the segmenter.py output
| Sequence | 30 | "Total frames per ""thought"" (3s * 10fps)"
| Resolution | 128x128 | Standardized input size

MVO Label ID Mapping:
* 0: FRONT
* 1: LEFT
* 2: RIGHT
* Convolutional Layers: Scan each individual frame to detect visual features like road curves, lane markings, and obstacles.
* LSTM Cells: Link the frames together in a sequence to understand motion. It remembers if the car was already turning or if it just started.
* Linear Classifier: Converts the final temporal memory into a navigation command (Left, Right, or Front).

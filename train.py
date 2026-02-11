import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import random 

# Custom project imports
from models.conv_lstm_classifier import ConvLSTMModel
from dataset import MVOVideoDataset
from utils import *

# Configurations
BATCH = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
SAVED_MODEL_PATH = "best_convlstm.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setting a fixed random seed to ensure that 
# we get the exact same data split every time we run the script
SEED = 8
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Model Parameters
PARAMS = {
    'input_dim': 4,
    'hidden_dim': [64, 32], 
    'kernel_size': (3, 3),
    'num_layers': 2,
    'height': HEIGHT,
    'width': WIDTH,
    'num_classes': 3 
}

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, leave=True)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.float().to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        running_loss += loss.item()
        _, predictions = scores.max(1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
        
        loop.set_description(f"Loss: {loss.item():.4f}")

    return running_loss / len(loader), 100 * correct / total

def main():
    # Data Setup
    transforms_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((HEIGHT, WIDTH))
    ])
    
    full_dataset = MVOVideoDataset(VIDEO_DIR, LABEL_DIR, transforms=transforms_train)
    
    # Train/Val/Test Split
    total_size = len(full_dataset)
    train_size = int(0.6 * total_size)                  # 60% for Training
    val_size = int(0.2 * total_size)                    # 20% for Validation
    test_size = total_size - train_size - val_size      # Remaining 20% for Testing

    # The generator with our fixed seed so the split is always the same
    generator = torch.Generator().manual_seed(SEED)
    
    # Split the dataset into three parts
    train_dataset, val_dataset, _ = random_split(
        full_dataset, 
        [train_size, val_size, test_size], 
        generator=generator
    )
    # Note: The last part is '_' because we don't touch the test set in train.py
    
    print(f"Data Split -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test (Unused): {test_size}")
    
    train_dataset.set_split_type('TRAIN', len(train_dataset))
    val_dataset.set_split_type('VALIDATION', len(val_dataset))

    # Calculate class instances for class weights
    label_counts, total_count = train_dataset.class_counter()
    # Add 1 to avoid division by zero
    front_weight = total_count / (label_counts[0] + 1) 
    left_weight = total_count / (label_counts[1] + 1)
    right_weight = total_count / (label_counts[2] + 1)

    print(f"Front class instances: {label_counts[0]} -> Front weight: {front_weight}")
    print(f"Left class instances: {label_counts[1]} -> Left weight: {left_weight}")
    print(f"Right class instances: {label_counts[2]} -> Right weight: {right_weight}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=0)

    # Model Setup
    model = ConvLSTMModel(
        input_dim=PARAMS['input_dim'],
        hidden_dim=PARAMS['hidden_dim'],
        kernel_size=PARAMS['kernel_size'],
        num_layers=PARAMS['num_layers'],
        height=PARAMS['height'],
        width=PARAMS['width'],
        num_classes=PARAMS['num_classes']
    ).to(DEVICE)

    # CrossEntropyLoss is used to handle class imbalance
    class_weights = torch.FloatTensor([front_weight,left_weight,right_weight]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    best_acc = 0
    print(f"Training on {DEVICE} with {len(train_dataset)} videos.")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.float().to(DEVICE)
                y = y.to(DEVICE)
                scores = model(x)
                _, preds = scores.max(1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        
        val_acc = 100 * val_correct / val_total
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVED_MODEL_PATH)
            print(f"New best model saved! ({val_acc:.2f}%)")

if __name__ == "__main__":
    main()
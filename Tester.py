import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Assuming these are your custom local files
from models import * 
from utils import *
from Preprocessor import *
from dataset import *

"""
The "Exam Proctor." 
It takes a trained brain (model_path), sets up the testing room (device/resize),
and runs through the test questions (videos) to see how many it gets right.
"""

class Tester:
    def __init__(self, model_path, device, model_params):
        """
        model_params: A dictionary containing hidden_dim, kernel_size, num_layers, etc.
        """
        self.model_path = model_path
        self.device = device
        self.params = model_params
        
        # Standardizing image size for the model
        self.val_transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((self.params['height'], self.params['width']))
        ])
        self.set_up()
    
    def set_up(self):
        # Building the brain and loading the memories
        self.model = ConvLSTMModel(
            input_dim=self.params['input_dim'],
            hidden_dim=self.params['hidden_dim'],
            kernel_size=self.params['kernel_size'],
            num_layers=self.params['num_layers'],
            height=self.params['height'],
            width=self.params['width'],
            num_classes=self.params['num_classes']
        )
        
        # Load the saved "knowledge" (.pth file)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval() # Set to evaluation mode (turns off Dropout/Batchnorm)

    def validate(self, val_loader, val_dataset):
        # Watch the videos and make a guess
        val_num_correct = 0
        predictions = []
        
        for i, (vx, vy) in enumerate(val_loader):
            # ConvLSTM expects: [Batch, Time, Channel, Height, Width]
            vx = vx.float().to(self.device)
            vy = vy.to(self.device)

            with torch.no_grad():
                outputs = self.model(vx) # Forward pass
            
            # Get the index of the highest score (the "guess")
            preds = torch.argmax(outputs, dim=1)
            
            # Store results
            predictions.append(preds.cpu().detach().numpy())
            val_num_correct += int((preds == vy).sum())
            
            # Clean up memory
            del vx, outputs 

        predictions = np.concatenate(predictions)
        acc = 100 * val_num_correct / (len(val_dataset))
        print(f"Validation Accuracy: {acc:.04f}%")
        return predictions, acc

    def validation_pipeline(self, video_ids):
        # Prepares the files, runs the test, and saves the results to a CSV.
        preprocessing_pipeline(video_ids)
        
        for video_id in video_ids:
            # Load video metadata/labels
            X, y, df = prep_video_test(osp.join(DATA_SAVE_PATH, video_id + ".csv"))
            
            # Creates the data conveyor belt (Dataset -> Loader)
            test_dataset = FrameDataset(X, y, transforms=self.val_transforms, base_path=PROCESSED_PATH)
            
            # Uses pin_memory and num_workers if using a GPU (cuda)
            is_cuda = self.device.type == 'cuda'
            val_args = dict(shuffle=False, batch_size=BATCH, num_workers=2, pin_memory=True) if is_cuda else dict(shuffle=False, batch_size=BATCH)
            
            test_loader = DataLoader(test_dataset, **val_args)
            
            print(f"Testing Video: {video_id} | Total Frames/Sequences: {len(test_dataset)}")
            
            predictions, acc = self.validate(test_loader, test_dataset)
            
            # Save the guesses back into the original dataframe
            df['predictions'] = predictions
            save_name = f"predictions_{video_id}_acc={acc:.2f}.csv"
            df.to_csv(save_name, index=None)
            print(f"Results saved to: {save_name}")
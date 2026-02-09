import torch
import cv2
import pandas as pd
import os
from torch.utils.data import Dataset

class MVOVideoDataset(Dataset):
    """
    This takes a 3-second video and turns it into 
    a 'data packet' for the AI to study.
    """
    def __init__(self, video_folder, label_folder, transforms=None):
        self.video_folder = video_folder
        self.label_folder = label_folder
        self.transforms = transforms
        # List all the 3-second videos
        self.video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        video_path = os.path.join(self.video_folder, video_name)
        
        # Load the 30 frames from the 3-second video
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(30): # We already know it's exactly 30 frames
            ret, frame = cap.read()
            if not ret:
                # If a video is shorter than 3s, pad with a black frame
                frame = torch.zeros((3, 128, 128))
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transforms:
                    from PIL import Image
                    frame = Image.fromarray(frame)
                    frame = self.transforms(frame)
            frames.append(frame)
        cap.release()

        # Convert list to a 5D tensor [1, 30, 3, 128, 128]
        video_tensor = torch.stack(frames, dim=0)

        #  Get the Label from the matching CSV file
        csv_name = video_name.replace('.mp4', '.csv')
        csv_path = os.path.join(self.label_folder, csv_name)
        
        # Read the label
        # For a 3s clip, the label is the maximum turn label present in the last second (24 frames in CSV)
        df = pd.read_csv(csv_path)
        df_lbl_count = []
        for i in range(0, 3):
            df_lbl_count = df['label'].tail(24).count(i)

        if df_lbl_count[0] == 24:
            label = 0 # Front
        elif df_lbl_count[1] > df_lbl_count[2]:
            label = 1 # Left
        elif df_lbl_count[1] < df_lbl_count[2]:
            label = 2 # Right
        else: # If turn counts are equal
            label = df['label'].tail(12).mode()[0]
            
        """
        Hannah check this plz
        
        label = df['label'].iloc[-1] # Taking the final decision of the MVO
        last_second = df['label'].tail(10)
        label = last_second.mode()[0]
        """

        return video_tensor, torch.tensor(label).long()
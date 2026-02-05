import torch
import torch.utils.data as data
import numpy as np
import os
import gzip
import urllib.request

class MovingMNIST(data.Dataset):
    """
    This class handles loading the 'Moving MNIST' video dataset.
    If the data isn't found, it downloads it from the internet.
    """
    def __init__(self, root, train=True, seq_len=20, image_size=64, transform=None):
        self.root = root
        self.train = train
        self.seq_len = seq_len
        self.image_size = image_size
        self.transform = transform
        self.url = 'http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy'
        
        # Download data if missing
        self.download()
        
        # Load the data into memory
        self.data = np.load(os.path.join(root, 'mnist_test_seq.npy'))
        
        # Swap axes to match (Sequence Length, Batch, Height, Width)
        self.data = self.data.transpose(1, 0, 2, 3)

    def __getitem__(self, index):
        # This function fetches one video sequence
        
        # Get the video data
        seq = self.data[index, :self.seq_len]
        
        # Normalize pixel values to be between 0 and 1
        seq = seq.astype(np.float32) / 255.0
        
        # Add a channel dimension (1 for grayscale)
        seq = torch.from_numpy(seq).unsqueeze(1) 
        
        return seq

    def __len__(self):
        # Returns total number of videos
        return len(self.data)

    def download(self):
        # Checks if file exists, if not, downloads it
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        
        path = os.path.join(self.root, 'mnist_test_seq.npy')
        if not os.path.exists(path):
            print("Downloading Moving MNIST dataset...")
            urllib.request.urlretrieve(self.url, path)
            print("Download complete.")
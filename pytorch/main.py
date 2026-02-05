# py -m pip install torch torchvision torchaudio matplotlib pytorch-lightning
# py -m pip install "lightning[extra]" tensorboard

import os
import torch
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from models.seq2seq_ConvLSTM import EncoderDecoderConvLSTM
from data.MovingMNIST import MovingMNIST
import argparse

# Parse command line arguments (settings for the run)
"""

Slower but more accurate

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs') # epochs
parser.add_argument('--n_hidden_dim', type=int, default=64, help='hidden dim')
opt, unknown = parser.parse_known_args()
"""

# Ngl tinamad lang ako maghintay kaya eto
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs') # epochs
parser.add_argument('--n_hidden_dim', type=int, default=32, help='hidden dim')
opt, unknown = parser.parse_known_args()

class MovingMNISTLightning(pl.LightningModule):
    """
    The Training Manager.
    It connects the Data, the Model, and the Optimization logic.
    """
    def __init__(self, model):
        super(MovingMNISTLightning, self).__init__()
        self.model = model
        self.criterion = torch.nn.MSELoss() # Measures error (Mean Squared Error)
        self.n_steps_past = 10
        self.n_steps_ahead = 10 

    def forward(self, x):
        # Runs the model to get a prediction
        return self.model(x, future_seq=self.n_steps_ahead)

    def training_step(self, batch, batch_idx):
        # PAST frames (the input)
        x = batch[:, 0:self.n_steps_past, :, :, :] 
        # FUTURE frames (the ground truth/answer key)
        y = batch[:, self.n_steps_past:, :, :, :]
        
        # The model predicts the future
        y_hat = self.forward(x)
        
        # To match the prediction (y_hat) so the computer can compare them.
        y = y.permute(0, 2, 1, 3, 4)
        
        # Calculate error
        loss = self.criterion(y_hat, y)
        
        # Log error to progress bar
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Sets up the Adam optimizer
        return torch.optim.Adam(self.parameters(), lr=opt.lr)

    def train_dataloader(self):
        # Loads the training data
        train_data = MovingMNIST(root='./data_download', train=True, seq_len=20)
        return DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2)

if __name__ == '__main__':
    # 1. Initialize the Model
    conv_lstm_model = EncoderDecoderConvLSTM(nf=opt.n_hidden_dim, in_chan=1)
    
    # 2. Wrap it in Lightning
    model = MovingMNISTLightning(model=conv_lstm_model)

    # 3. Setup Trainer (Manages the GPU and loop)
    # Note: 'accelerator="auto"' will automatically pick GPU if available
    trainer = Trainer(max_epochs=opt.epochs, accelerator="auto")

    # 4. Start Training
    trainer.fit(model)
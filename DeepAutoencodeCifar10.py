# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:20:39 2024

@author: Mahdi
"""


## Standard libraries
import os
import json
import math
import numpy as np

## Imports for plotting
import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats
from statistics import mean 
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()
sns.set()

## Progress bar
# from tqdm.notebook import tqdm
import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
# # PyTorch Lightning
# try:
#     import pytorch_lightning as pl
# except ModuleNotFoundError: # Google Colab does not have PyTorch Lightning installed by default. Hence, we do it here if necessary
#     !pip install --quiet pytorch-lightning>=1.4
#     import pytorch_lightning as pl
# from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Tensorboard extension (for visualization purposes later)
from torch.utils.tensorboard import SummaryWriter
# %load_ext tensorboard

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "saved_models"

# # Setting the seed
# pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# print("Device:", device)


import urllib.request
from urllib.error import HTTPError
# Github URL where saved models are stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial9/"
# Files to download
pretrained_files = ["cifar10_64.ckpt", "cifar10_128.ckpt", "cifar10_256.ckpt", "cifar10_384.ckpt"]
# Create checkpoint path if it doesn't exist yet
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# For each file, check whether it already exists. If not, try downloading it.
for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print("Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n", e)



def get_train_images(num):
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)


class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            # nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            # act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            # nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            # act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*c_hid, latent_dim)
        )

    def forward(self, x):
        return self.net(x)
    
    
class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*16*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            # act_fn(),
            # nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            # act_fn(),
            # nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x
    
class Autoencoder(nn.Module):

    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 num_input_channels: int = 3,
                 width: int = 32,
                 height: int = 32):
        super().__init__()
        # Saving hyperparameters of autoencoder
        # self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)
        
# def compare_imgs(img1, img2, title_prefix=""):
#     # Calculate MSE loss between both images
#     loss = F.mse_loss(img1, img2, reduction="sum")
#     # Plot images for visual comparison
#     grid = torchvision.utils.make_grid(torch.stack([img1, img2], dim=0), nrow=2, normalize=True, value_range=(-1,1))
#     grid = grid.permute(1, 2, 0)
#     plt.figure(figsize=(4,2))
#     plt.title(f"{title_prefix} Loss: {loss.item():4.2f}")
#     plt.imshow(grid)
#     plt.axis('off')
#     plt.show()

# for i in range(2):
#     # Load example image
#     img, _ = train_dataset[i]
#     img_mean = img.mean(dim=[1,2], keepdims=True)

#     # Shift image by one pixel
#     SHIFT = 1
#     img_shifted = torch.roll(img, shifts=SHIFT, dims=1)
#     img_shifted = torch.roll(img_shifted, shifts=SHIFT, dims=2)
#     img_shifted[:,:1,:] = img_mean
#     img_shifted[:,:,:1] = img_mean
#     compare_imgs(img, img_shifted, "Shifted -")

#     # Set half of the image to zero
#     img_masked = img.clone()
#     img_masked[:,:img_masked.shape[1]//2,:] = img_mean
#     compare_imgs(img, img_masked, "Masked -")


def train(base_channel_size, latent_dim, num_epochs):
    
    model = Autoencoder(base_channel_size, latent_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3, weight_decay=1e-5)
    
    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pretrained = False
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        checkpoint = torch.load(pretrained_filename, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        trained_epochs = checkpoint['epoch']
        pretrained = True
        num_epochs = 1 # no need for many epochs as we are not going to train
    
    model.to(device)   
    
    # Initialize lists to store the loss values
    train_losses = []
    val_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        if not pretrained:
            model.train()    
            train_loss = 0.0    
            
            # Wrap the training data loader with tqdm for a progress bar
            # for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", mininterval=0.1, leave=True):
            for i, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                images, _ = batch
                images = images.to(device)
        
                # Forward pass
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
        
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                train_loss += loss.item() * images.size(0)
        
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")
    
        # Validation step
        model.eval()
        val_loss = 0.0
    
        with torch.no_grad():
            # Wrap the validation data loader with tqdm for a progress bar
            for i, batch in enumerate(tqdm.tqdm(val_loader, desc="Validating", unit="batch")):
                images, _ = batch
                images = images.to(device)
    
                # Forward pass
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
    
                val_loss += loss.item() * images.size(0)
    
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Test step
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            # Wrap the validation data loader with tqdm for a progress bar
            for i, batch in enumerate(tqdm.tqdm(test_loader, desc="Testing", unit="batch")):
                images, _ = batch
                images = images.to(device)
    
                # Forward pass
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
    
                test_loss += loss.item() * images.size(0)                
    
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(f"Test Loss: {test_loss:.4f}")
        
    val_loss = [{'test_loss': mean(val_losses)}]
    test_loss = [{'test_loss': mean(test_losses)}]
    
    result = {"test": val_loss, "val": test_loss}
    print(result)
    
    if not pretrained:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.show()
    return model, result
    
def visualize_reconstructions(model, input_imgs, latent_dim):
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(input_imgs.to(device))
    reconst_imgs = reconst_imgs.cpu()

    # Plotting
    imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, value_range=(-1,1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(7,4.5))
    plt.title(f"Reconstructed from {latent_dim} latents")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Transformations applied on each image => only make them a tensor
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,))])
    
    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=transform, download=True)
    
    train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])
    
    # Loading the test set
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=transform, download=True)
    
    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)

    latent_dim = 64
    base_channel_size=32
    Max_num_epochs = 10
    
    model_dict = {}
    for latent_dim in [64, 128, 256, 384]:
        model_ld, result_ld = train(base_channel_size, latent_dim, Max_num_epochs)
        model_dict[latent_dim] = {"model": model_ld, "result": result_ld}
        # break
    


    ################### Plot the reconstruction error ########################    
    latent_dims = sorted([k for k in model_dict])
    val_scores = [model_dict[k]["result"]["val"][0]["test_loss"] for k in latent_dims]

    fig = plt.figure(figsize=(6,4))
    plt.plot(latent_dims, val_scores, '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16)
    plt.xscale("log")
    plt.xticks(latent_dims, labels=latent_dims)
    plt.title("Reconstruction error over latent dimensionality", fontsize=14)
    plt.xlabel("Latent dimensionality")
    plt.ylabel("Reconstruction error")
    plt.minorticks_off()
    plt.ylim(0,0.5)
    plt.show()
    
    #################### Visualaize reconstruction ############################
    input_imgs = get_train_images(4)
    for latent_dim in model_dict:
        visualize_reconstructions(model_dict[latent_dim]["model"], input_imgs, latent_dim)

    
    

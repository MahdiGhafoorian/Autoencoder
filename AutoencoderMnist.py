# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:52:51 2024

@author: Mahdi
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


transform = transforms.ToTensor()

mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                          batch_size=64,
                                          shuffle=True)
dataiter = iter(data_loader)
images , labels = next(dataiter)
print(torch.min(images), torch.max(images)) # shows the input range

class Autoencoder_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        # N, 784 (Mnist image size 28*28)
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3))
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()) # we use sigmoid because the input was between 0 and 1
            # if input is in [-1, 1] use Tanh
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
def train(model, criterion, optimizer, num_epochs, model_type):
    outputs=[]
    model.to(device)
    for epoch in range(num_epochs):
        for (img,_) in data_loader:
            if model_type == 0:
                img = img.reshape(-1, 28*28)
            img = img.to(device)
            recon = model(img)
            loss = criterion(recon, img)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        print(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}")
        outputs.append((epoch, img, recon))
    return outputs

class Autoencoder_CNN(nn.Module):
    def __init__(self):
        super().__init__()        
        # N, 1, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # -> N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # -> N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7) # -> N, 64, 1, 1
        )
        
        # N , 64, 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7), # -> N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # N, 16, 14, 14 (N,16,13,13 without output_padding)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), # N, 1, 28, 28  (N,1,27,27)
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
   
 # Note: nn.MaxPool2d -> use nn.MaxUnpool2d, or use different kernelsize, stride etc to compensate...
# Input [-1, +1] -> use nn.Tanh
            

if __name__ == '__main__':
    
    model_type = 1 # 0: Linear model,  1: CNN model
    
    if(model_type == 0):
        model = Autoencoder_Linear()
    else:
        model = Autoencoder_CNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    num_epochs = 10
    outputs = train(model, criterion, optimizer, num_epochs, model_type)
    
    for k in range(0 , 10, 4): # show img and reconstructed version of it during the traing
        plt.figure(figsize = (9, 2))
        plt.gray()
        img = outputs[k][1].cpu().detach().numpy()
        recon = outputs[k][2].cpu().detach().numpy()
        for i, item in enumerate(img):
            if i >= 9: break
            plt.subplot(2, 9, i+1)
            if model_type == 0:
                item = item.reshape(-1 , 28,28)
            # item: 1, 28 , 28
            plt.imshow(item[0])  
            
        for i, item in enumerate(recon):
            if i >= 9: break
            plt.subplot(2, 9, 9+i+1)
            if model_type == 0:
                item = item.reshape(-1 , 28,28)
            # item: 1, 28 , 28
            plt.imshow(item[0])
    
            
            
            
            
            
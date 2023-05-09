# required imports
import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import model

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

matplotlib.style.use('ggplot')

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='# of epochs to train for')
parser.add_argument('-b', '--batch-size', type=int, default=128,
                    help='batch size for training')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('-n', '--num-features', type=int, default=16,
                    help='# of features in the latent space')
# bad design choice, should be set using torch.cuda.is_available()
# parser.add_argument('-d', '--device', type=str, default='cuda',
#                     choices=['cpu', 'cuda'], help='device to use for training')
args = vars(parser.parse_args())

# learning parameters
epochs = args['epochs']
batch_size = args['batch_size']
lr = args['learning_rate']
num_features = args['num_features']
# device used for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Prepare the dataset for training.
'''
# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# MNIST dataset
train_dataset = datasets.MNIST(root='../data/',
                               train=True,
                               transform=transform,
                               download=True)
test_dataset = datasets.MNIST(root='../data/',
                              train=False,
                              transform=transform,
                              download=True)

# dataloaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
val_loader = DataLoader(dataset=test_dataset,
                        batch_size=batch_size,
                        shuffle=False)

# model initialization
model = model.LinearVAE(num_features).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
# TODO: check if this is the correct loss function
# see: https://jamesmccaffrey.wordpress.com/2021/03/18/why-you-can-use-either-mean-squared-error-or-binary-cross-entropy-for-mnist-data/
criterion = nn.BCELoss(reduction='sum') # binary cross entropy loss

'''
Full loss function.
'''
def compute_loss(bce_loss, mu, logvar):
    '''
    This function will add the reconstruction loss (bce_loss) and the KL-Divergence loss.
    params:
        bce_loss: the binary cross entropy loss
        mu: the mean of the latent space
        logvar: the log variance of the latent space
    returns:
        the sum of the reconstruction loss and the KL-Divergence loss
    '''
    # KL-Divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return bce_loss + kld_loss


'''
The training function.
'''
def fit(model, dataloader):
    '''
    This function will train the model for one epoch.
    params:
        model: the model to train
        dataloader: the dataloader to use for training
    returns:
        the average loss for the epoch
    '''
    model.train()
    running_loss = 0.0
    
    for i, data in tqdm(enumerate(dataloader),
                        total=int(len(train_dataset)/dataloader.batch_size)):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        bce_loss = criterion(recon_batch, data)
        loss = compute_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    train_loss = running_loss/len(dataloader.dataset)
    
    return train_loss

'''
The validation function.
'''
def validate(model, dataloader):
    '''
    This function will validate the model for one epoch.
    params:
        model: the model to validate
        dataloader: the dataloader to use for validation
    returns:
        the average loss for the epoch
    '''
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader),
                            total=int(len(test_dataset)/dataloader.batch_size)):
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)
            recon_batch, mu, logvar = model(data)
            bce_loss = criterion(recon_batch, data)
            loss = compute_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            
            # save the last batch input and output for every epoch
            if i == int(len(test_dataset)/dataloader.batch_size) - 1:
                num_rows = 8
                both = torch.cat((data.view(batch_size, 1, 28, 28)[:8],
                                  recon_batch.view(batch_size, 1, 28, 28)[:8]))
                save_image(both.cpu(), f'../results/linear-vae/reconstruction_{epoch}.png', nrow=num_rows)
            
        val_loss = running_loss/len(dataloader.dataset)
        
    return val_loss

'''
Training loop.
'''
train_loss = []
val_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model, train_loader)
    val_epoch_loss = validate(model, val_loader)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {val_epoch_loss:.4f}')
    print('-------------------')

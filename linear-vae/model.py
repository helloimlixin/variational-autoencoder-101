# required imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

'''
The Linear VAE module.
'''
class LinearVAE(nn.Module):
    def __init__(self, num_features):
        super(LinearVAE, self).__init__()
        
        '''
        The Architecture of the Linear VAE is as follows:
        '''
        
        self.latent_dim = num_features # output feature dimension of the encoder
        
        # encoder layers
        
        # corresponds to the input image size (MNIST Digits: 28*28*1)
        self.enc1 = nn.Linear(in_features=784, out_features=512)
        # last layer of the encoder, outputs the mean and log variance of the latent space
        self.enc2 = nn.Linear(in_features=512, out_features=num_features * 2)
        
        # decoder layers
        # stacked in reverse order as compared to the encoder
        self.dec1 = nn.Linear(in_features=num_features, out_features=512)
        self.dec2 = nn.Linear(in_features=512, out_features=784)
    
    def reparameterize(self, mu, log_var):
        '''
        This function takes in the mean and log variance of the latent space and
        returns a sample from the latent space.
        params:
            mu: the mean of the latent space
            log_var: the log variance of the latent space
        returns:
            a sample from the latent space
        '''
        std = torch.exp(0.5*log_var) # compute standard deviation using log-var
        eps = torch.randn_like(std) # gaussian noise
        sample = mu + (eps*std) # sampling with the reparemeterization trick
        
        return sample
    
    def forward(self, x):
        # encoding
        print(f'x.shape: {x.shape}')
        x = F.relu(self.enc1(x))
        print(f'h.shape: {x.shape}')
        x = self.enc2(x).view(-1, 2, self.latent_dim)
        
        print(f'x.shape after two encoder layers: {x.shape}')
        
        # get the mean and log variance of the latent space
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the second feature values as log variance
        
        print(f'mu.shape: {mu.shape}')
        print(f'log_var.shape: {log_var.shape}')
        
        # get a sample from the latent space through reparameterization
        z = self.reparameterize(mu, log_var)
        
        print(f'z.shape: {z.shape}')
        
        time.sleep(1000000)
        
        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        
        return reconstruction, mu, log_var
        
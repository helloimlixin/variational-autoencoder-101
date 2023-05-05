# required imports
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The Linear VAE module.
'''
class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self, num_features).__init__()
        
        '''
        The Architecture of the Linear VAE is as follows:
        '''
        
        self.latent_dim = num_features
        
        # encoder layers
        self.enc1 = nn.Linear(in_features=784, out_features=512)
        self.enc2 = nn.Linear(in_features=512, out_features=latent_dim * 2)
        
        # decoder layers
        self.dec1 = nn.Linear(in_features=latent_dim, out_features=512)
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
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps*std) # sampling with the reparemeterization trick
        
        return sample
    
    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, self.latent_dim)
        
        # get the mean and log variance of the latent space
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the second feature values as log variance
        
        # get a sample from the latent space through reparameterization
        z = self.reparameterize(mu, log_var)
        
        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        
        return reconstruction, mu, log_var
        
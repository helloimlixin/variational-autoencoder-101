import torch
import torch.nn as nn
import torch.nn.functional as F

kernel_size = 4 # (4, 4) convolution kernel
init_channels = 8 # initial number of filters
image_channels = 1 # MNIST images are grayscale
latent_dim = 16 # latent dimension for sampling

# define a Convolutional VAE class
class ConvolutionalVAE(nn.Module):
    def __init__(self) -> None:
        """Constructo method for ConvolutionalVAE class.
        The <code>kernel_size</code> is 4 x 4 for all the layers, and the stride
        length is 2 for all layers. Having a large kernel size and stride length
        of 2 will ensure that each time we are capturing a lot of spatial
        information and we are doing that repeatedly as well. This will help us
        to learn all the spatial information of the input image.
        """
        super(ConvolutionalVAE, self).__init__()
        
        # encoder
        self.enc1 = nn.Conv2d(in_channels=image_channels,
                              out_channels=init_channels,
                              kernel_size=kernel_size,
                              stride=2, padding=1)
        self.enc2 = nn.Conv2d(in_channels=init_channels,
                              out_channels=init_channels*2,
                              kernel_size=kernel_size,
                              stride=2, padding=1)
        self.enc3 = nn.Conv2d(in_channels=init_channels*2,
                              out_channels=init_channels*4,
                              kernel_size=kernel_size,
                              stride=2, padding=1)
        self.enc4 = nn.Conv2d(in_channels=init_channels*4,
                              out_channels=64, kernel_size=kernel_size,
                              stride=2, padding=0)
        
        # fully connected layers for learning representations
        # The fully connected dense features will help the model to learn all
        # the interesting representations of the input image. A dense bottleneck
        # will give our model a good overall view of the whole data and thus may
        # help in better reconstruction.
        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.fc_mu = nn.Linear(in_features=128, out_features=latent_dim)
        self.fc_log_var = nn.Linear(in_features=128, out_features=latent_dim)
        self.fc2 = nn.Linear(in_features=latent_dim, out_features=64)
        
        # decoder
        self.dec1 = nn.ConvTranspose2d(in_channels=64,
                                       out_channels=init_channels*8,
                                       kernel_size=kernel_size,
                                       stride=1, padding=0)
        self.dec2 = nn.ConvTranspose2d(in_channels=init_channels*8,
                                       out_channels=init_channels*4,
                                       kernel_size=kernel_size,
                                       stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(in_channels=init_channels*4,
                                       out_channels=init_channels*2,
                                       kernel_size=kernel_size,
                                       stride=2, padding=1)
        self.dec4 = nn.ConvTranspose2d(in_channels=init_channels*2,
                                       out_channels=image_channels,
                                       kernel_size=kernel_size,
                                       stride=2, padding=1)
    
    def reparameterize(self, mu, log_var):
        '''
        This function will sample the latent vector from the distribution
        characterized by mu and log_var.
        
        :param mu: mean from the latent vector
        :param log_var: log variance from the latent vector
        '''
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        
        return sample
    
    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        
        z = z.view(-1, 64, 1, 1)
        
        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        
        return reconstruction, mu, log_var

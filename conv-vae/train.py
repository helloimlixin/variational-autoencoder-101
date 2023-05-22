import torch
import torch.optim as optim
import torch.nn as nn
import model
import torchvision.transforms as transforms
import torchvision
import matplotlib

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from engine import train, validate
from utils import save_reconstructed_images, images_to_gif, save_loss_plot

matplotlib.style.use('ggplot')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the model
model = model.ConvolutionalVAE().to(device)

# set the learning parameters
learning_rate = 1e-3
epochs = 100
batch_size = 64
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss(reduction='sum')

# save reconstructed images in PyTorch grid format
grid_images = []

# define the image transformations and dataloaders
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# training set and train data loader
train_dataset = torchvision.datasets.MNIST(
    root='../data', train=True, transform=transform, download=True
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# validation set and validation data loader
val_dataset = torchvision.datasets.MNIST(
    root='../data', train=False, transform=transform, download=True
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, train_loader, train_dataset, device, optimizer, criterion)
    valid_epoch_loss, reconstructed_images = validate(
        model, val_loader, val_dataset, device, criterion)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    
    # save the reconstructed images from the validation loop
    save_reconstructed_images(reconstructed_images, epoch+1)
    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(reconstructed_images.detach().cpu())
    grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")
    
# save the reconstructions as a .gif file
images_to_gif(grid_images)

# save the loss plots to disk
save_loss_plot(train_loss, valid_loss)

print('DONE TRAINING')
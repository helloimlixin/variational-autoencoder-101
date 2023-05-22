from tqdm import tqdm
import torch

def compute_loss(bce_loss, mu, logvar):
    '''
    This function will add the reconstruction loss (BCELoss) and the KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    '''
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return bce_loss + KLD

def train(model, dataloader, dataset, device, optimizer, criterion):
    model.train()
    running_loss = 0.0 # keep track of the batch-wise loss values
    counter = 0 # keep track of the total number of training steps
    
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data = data[0] # capture only the image data, not the associated label
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = compute_loss(bce_loss, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    
    train_loss = running_loss / counter
    
    return train_loss

def validate(model, dataloader, dataset, device, criterion):
    model.eval()
    running_loss = 0.0 # keep track of the batch-wise loss values
    counter = 0 # keep track of the total number of training steps
    
    with torch.no_grad(): # temporarily set all the requires_grad flag to false
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data = data[0] # capture only the image data, not the associated label
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = compute_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            
            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                reconstructions = reconstruction
    
    val_loss = running_loss / counter
    
    return val_loss, reconstructions
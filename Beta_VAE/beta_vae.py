import numpy as np

from sklearn.metrics import mutual_info_score

import torch
from torch import nn
from torchvision.models import resnet18
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

### Annealing the samples_per_batch parameter (gradually increase)

class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNetEncoder, self).__init__()
        resnet = resnet18()
        self.latent_dim = latent_dim
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, latent_dim * 2)  # For mu and logvar

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        mu = x[:, :self.latent_dim]
        logvar = x[:, self.latent_dim:]
        return mu, logvar

class ResNetDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNetDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512)  # Adjust as needed
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 52, stride=2, padding=1),  # Assuming 3-channel output
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1,3,64,64))

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 512, 1, 1)  # Adjust shape to match the start of deconv layers
        return self.deconv(z)

class BetaVAE(nn.Module):
    def __init__(self, latent_dim=20, beta=1.0):
        super(BetaVAE, self).__init__()
        self.encoder = ResNetEncoder(latent_dim)
        self.decoder = ResNetDecoder(latent_dim)
        self.beta = beta
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar_z, logvar_x, beta):
    sigma_z = torch.exp(0.5 * logvar_z)
    sigma_x = torch.exp(0.5 * logvar_x)
    log_likelihood = nn.functional.mse_loss(recon_x, x, reduction='sum')/(2*sigma_x.sum()**2)
    KLD = -0.5 * torch.sum(1 + logvar_x - mu.pow(2) - logvar_x.exp())
    return log_likelihood + beta * KLD


class TeapotsDatasetNPZ(Dataset):
    def __init__(self, npz_file, transform=None):
        data = np.load(npz_file)
        self.images = data['images']  # Assuming 'images' is the key for images
        self.gts = data['gts']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1) / 255.0  # Convert to PyTorch tensor and normalize
        gts = self.gts[idx]
        if self.transform:
            image = self.transform(image)
        return image, gts


def train(epoch, model, data_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(data_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar, model.beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(data_loader.dataset):.4f}')
    
    return model


def evaluate(model, device, data_loader):
    def calculate_R_ij(latent_representations, ground_truths):
        num_latent_dims = latent_representations.shape[1]
        num_factors = ground_truths.shape[1]
        R = np.zeros((num_latent_dims, num_factors))
        
        for i in range(num_latent_dims):
            for j in range(num_factors):
                # Calculate mutual information between the i-th latent dimension
                # and the j-th ground truth factor
                weights = lasso(latent_representations[:, i], ground_truths[:, j])
                R[i, j] = np.abs(weights) / np.sum(np.abs(weights))
        return R

    def calculate_P_ij(R):
        # Normalize R_ij over all j for each i to get P_ij
        P_ij = R / R.sum(axis=1, keepdims=True)
        return P_ij    

    def entropy(P):
        return -np.sum(P * np.log(P + 1e-9), axis=1)  # Add small value to prevent log(0)

    def disentanglement_score(P):
        H = entropy(P)
        D = 1 - H / np.log(P.shape[1])  # Normalize by the log of the number of factors
        return D
    
    latent_representations = []
    ground_truths = []

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for inputs, labels in data_loader:
            # inputs are the data
            # labels correspond to the ground truth factors ('gts')
            mu, logvar = model.encode(inputs)
            z = model.reparameterize(mu, logvar)  # Get the latent representation
            latent_representations.append(z.cpu().numpy())
            ground_truths.append(labels.cpu().numpy())

    latent_representations = np.concatenate(latent_representations, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)
    
    # Compute DCI (Disentanglement, Completeness, Informativeness) metrics
    # D_i = (1 - H_K(P_i.))
    # C_i = (1 - H_D(P_.j))
    # I = E(z_j, hat_z_j)
    # Calculate disentanglement scores
    R = calculate_R_ij(latent_representations, ground_truths)
    P = calculate_P_ij(R)
    D = disentanglement_score(P)
    print("Disentanglement", D)
    
    return D


# Data loading and training setup
npz_path = './data/teapots.npz'  # Update this path
dataset = TeapotsDatasetNPZ(npz_path)
model = BetaVAE(latent_dim=20, beta=4.0)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
optimizer = Adam(model.parameters(), lr=1e-4)
epochs = 2
beta = 6.0

# Traing the model
for epoch in range(1, epochs + 1):
    model = train(epoch, model, train_loader, optimizer)
    torch.save(model.state_dict(), './beta_vae.pth')

# Evaluate the model
D = evaluate(model, device, train_loader)



import argparse
import numpy as np
from scipy.stats import entropy
from sklearn.linear_model import Lasso
from sklearn.preprocessing import normalize
from sklearn.metrics import mutual_info_score
from sklearn.metrics import mutual_info_score

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from utils import TeapotsDatasetNPZ

class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNetEncoder, self).__init__()
        resnet = resnet18()
        self.latent_dim = latent_dim
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, latent_dim * 2)

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
        self.fc = nn.Linear(latent_dim, 512)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 52, stride=2, padding=1),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1,3,64,64))

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 512, 1, 1)
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
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

def vae_loss(x, recon_x, mu_z, logvar_z, beta):
    # Reconstruction loss (gaussian)
    log_likelihood = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

    # KL divergence between the approximate posterior and the prior
    KLD = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

    # Total loss with the β scaling factor applied to KL divergence
    total_loss = log_likelihood + beta * KLD

    return total_loss


def train(epoch, model, device, data_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(data_loader):
        optimizer.zero_grad()
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(data, recon_batch, mu, logvar, model.beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 1000 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(data_loader.dataset):.4f}')
    
    return model


def evaluate(model, device, data_loader):
    model.eval()
    latents = []
    factors = []
    
    with torch.no_grad():
        for data, gts in data_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            latents.append(mu.cpu().numpy())
            factors.append(gts.cpu().numpy())

    latents = np.concatenate(latents, axis=0)
    factors = np.concatenate(factors, axis=0)

    R = np.zeros((latents.shape[1], factors.shape[1]))  # Latent dimensions x Generative factors

    # Train a regressor for each latent dimension and each factor
    for i in range(latents.shape[1]):  # For each latent dimension
        for j in range(factors.shape[1]):  # For each generative factor
            regressor = Lasso(alpha=0.1)
            regressor.fit(latents[:, i:i+1], factors[:, j])
            R[i, j] = np.abs(regressor.coef_)

    # Normalize R across columns (latent dimensions) for C
    R_norm_columns = normalize(R, norm='l1', axis=0)
    C = 1 - entropy(R_norm_columns, base=2, axis=0)
    C = np.mean(C)

    # Normalize R across rows (generative factors) for D
    R_norm_rows = normalize(R, norm='l1', axis=1)
    D = 1 - entropy(R_norm_rows, base=2, axis=1)
    D = np.mean(D)

    # Calculate Mutual Information for Informativeness (I)
    mutual_infos = []
    for i in range(latents.shape[1]):
        mi = mutual_info_score(None, None, contingency=np.histogram2d(latents[:, i], factors.ravel())[0])
        mutual_infos.append(mi)
    I = np.mean(mutual_infos)

    return D, C, I


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:6", help="Device to train the model")
    parser.add_argument("--npz-path", type=str, default='./data/teapots.npz', help="Path to the NPZ file")
    parser.add_argument("--beta", type=float, default=6.0, help="Beta value for the VAE loss")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    args = parser.parse_args()

    # Data loading and training setup
    dataset = TeapotsDatasetNPZ(args.npz_path)
    model = BetaVAE(latent_dim=20, beta=args.beta)
    device = torch.device(args.device)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    epochs = args.epochs

    # Traing the model
    model.to(device)
    for epoch in range(1, epochs + 1):
        model = train(epoch, model, device, train_loader, optimizer)
        torch.save(model.state_dict(), './beta_vae.pth')

    # Evaluate the model
    model.eval()
    D, C, I = evaluate(model, device, train_loader)

    print(f"Disentanglement: {D:.4f}, Completeness: {C:.4f}, Informativeness: {I:.4f}")



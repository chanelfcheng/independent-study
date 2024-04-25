import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import normalize
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy


class TeapotsDatasetNPZ(Dataset):
    def __init__(self, npz_file, transform=None):
        data = np.load(npz_file)
        self.images = data['images']
        self.gts = data['gts']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1) / 255.0
        gts = self.gts[idx]
        if self.transform:
            image = self.transform(image)
        return image, gts

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
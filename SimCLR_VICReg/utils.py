import torch
from torch.utils.data import Dataset
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        def cosine_similarity(z_i, z_j):
            # L2 normalize the vectors along the last dimension
            z_i_norm = z_i / (z_i.norm(dim=1, keepdim=True) + 1e-6)
            z_j_norm = z_j / (z_j.norm(dim=1, keepdim=True) + 1e-6)

            # Compute the cosine similarity
            cosine_sim = torch.mm(z_i_norm, z_j_norm.T)
            return cosine_sim

        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = cosine_similarity(z, z)

        # Construct the positive similarity vector
        sim_ij = torch.diag(sim_matrix, z_i.size(0))
        sim_ji = torch.diag(sim_matrix, -z_i.size(0))
        log_numerator = torch.cat((sim_ij, sim_ji), dim=0) / self.temperature

        # Construct the negative similarity vector
        tensor = torch.eye(z_i.size(0), device=z.device, dtype=bool)
        mask = torch.vstack([
            torch.hstack([tensor, tensor]), torch.hstack([tensor, tensor])
        ])
        sim_matrix = sim_matrix.masked_fill(mask, 0)

        log_denominator = torch.logsumexp(sim_matrix / self.temperature, dim=1)
        quantity_to_maximize = log_numerator - log_denominator
        loss = -quantity_to_maximize
        return loss.mean()


class VICRegLoss(nn.Module):
    def __init__(self, lamb=25, mu=25, nu=1):
        super(VICRegLoss, self).__init__()
        self.lamb = lamb
        self.mu = mu
        self.nu = nu
    
    def forward(self, z_i, z_j):
        # Invariance loss
        sim_loss = F.mse_loss(z_i, z_j)

        # Variance loss
        std_z_i = torch.sqrt(z_i.var(dim=0) + 1e-04)
        std_z_j = torch.sqrt(z_j.var(dim=0) + 1e-04)
        std_loss = torch.mean(torch.relu(1 - std_z_i)) + torch.mean(torch.relu(1 - std_z_j))

        # Covariance loss
        N, D = z_i.size()
        z_i = z_i - z_i.mean(dim=0)
        z_j = z_j - z_j.mean(dim=0)
        cov_z_i = (z_i.T @ z_i) / (N - 1)
        cov_z_j = (z_j.T @ z_j) / (N - 1)
        # cov_loss -= (cov_z_i.diagonal() ** 2).sum() / D + (cov_z_j.diagonal()
        # ** 2).sum() / D
        cov_loss = ((torch.triu(cov_z_i, 1) ** 2).sum() / D + (torch.triu(cov_z_j, 1) ** 2).sum() / D) * 4

        # Combine losses
        # Look at each of the 3 terms separately -- log them
        loss = self.lamb * sim_loss + self.mu * std_loss + self.nu * cov_loss
        return loss


class AugmentedPairDataset(Dataset):
    def __init__(self, images, transform=None, subset_size=None):
        self.images = images
        self.transform1 = transforms.Compose([
            transforms.ToTensor()
        ])
        self.transform2 = transform
        if subset_size is not None:
            # Ensure subset_size does not exceed the total number of images
            subset_size = min(subset_size, len(images))
            indices = np.random.choice(len(images), subset_size, replace=False)
            self.images = images[sorted(indices)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.transform2:
            img1 = self.transform1(img)
            img2 = self.transform2(img)
        else:
            img1 = self.transform1(img)
            img2 = self.transform1(img)
        
        return img1, img2
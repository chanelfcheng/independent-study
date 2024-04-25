import torch
import torch.nn as nn
from torchvision.models import resnet18


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
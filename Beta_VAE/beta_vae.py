import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils import *
from models import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:6", help="Device to train the model")
    parser.add_argument("--npz-path", type=str, default='./data/teapots.npz', help="Path to the NPZ file")
    parser.add_argument("--beta", type=float, default=6.0, help="Beta value for the VAE loss")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    args = parser.parse_args()

    # Data loading and training setup
    dataset = TeapotsDatasetNPZ(args.npz_path)
    device = torch.device(args.device)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Train the model if no saved model is found
    if not os.path.exists('./beta_vae.pth'):
        print("Training the model...")
        model = BetaVAE(latent_dim=20, beta=args.beta)
        model.to(device)
        optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        epochs = args.epochs
        for epoch in range(1, epochs + 1):
            model = train(epoch, model, device, dataloader, optimizer)
            torch.save(model.state_dict(), './beta_vae.pth')
    else:
        print("Loading the model...")
        model = BetaVAE(latent_dim=20, beta=args.beta)
        model.load_state_dict(torch.load('./beta_vae.pth'))
        model.to(device)

    # Evaluate the model
    print("Evaluating the model...")
    model.eval()
    D, C, I = evaluate(model, device, dataloader)

    print(f"Disentanglement: {D:.4f}, Completeness: {C:.4f}, Informativeness: {I:.4f}")



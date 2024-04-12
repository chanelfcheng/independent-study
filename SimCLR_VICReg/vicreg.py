import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import *

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
        print(loss)
        return loss


if __name__ == "__main__":
    # Define data transforms
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=32, scale=(0.8,1)),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomChoice([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), transforms.Grayscale(num_output_channels=3)])
        # transforms.GaussianBlur(kernel_size=9)
    ])
    batch_size = 128

    if not os.path.exists('vicreg_model.pth'):
        # Create dataset and dataloader
        cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        images = cifar10_dataset.data
        dataset = AugmentedPairDataset(images, transform=data_transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Setup device, model, and optimizer
        device = torch.device('cuda:8' if torch.cuda.is_available() else 'cpu')
        model = BaseModel().to(device)
        vicreg_loss = VICRegLoss()
        # optimizer = LARSOptimizer(model.parameters(), lr=0.3*batch_size/256,
        # weight_decay=1e-6)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
        scheduler = CosineLearningRateScheduler(optimizer, base_learning_rate=0.001, num_examples=len(dataset), train_batch_size=batch_size, warmup_epochs=10, total_epochs=100)

        # Train the model
        epochs = 10
        model.train()
        print("Start training...")
        for epoch in range(epochs):
            for img1, img2 in dataloader:
                img1, img2 = img1.to(device), img2.to(device)
                optimizer.zero_grad()
                z_i, z_j = model(img1, img2)
                loss = vicreg_loss(z_i, z_j)
                loss.backward()
                optimizer.step()
            scheduler.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        print("Training complete.")

        # Save the model
        torch.save(model.state_dict(), 'vicreg_model.pth')
    else:
        # Load the model
        device = torch.device('cuda:8' if torch.cuda.is_available() else 'cpu')
        model = BaseModel().to(device)
        model.load_state_dict(torch.load('vicreg_model.pth'))
        model.eval()

    # Use representations learned by model to train linear classifier
    cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', transform=data_transforms, train=True, download=True)
    dataloader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    linear = LinearClassifier(input_dim=128, num_classes=10).to(device)
    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(linear.parameters(), lr=0.001, weight_decay=1e-6) 

    epochs = 10
    linear.train()
    print("Start training linear classifier...")
    for epoch in range(epochs):
        for img, label in dataloader:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            z, _ = model(img, img)
            output = linear(z)
            loss = cross_entropy_loss(output, label)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    print("Training complete.")

    # Save the linear classifier
    torch.save(linear.state_dict(), 'linear_model.pth')

    # Evaluate the linear classifier
    linear.eval()
    correct = 0
    total = 0
    print("Start evaluating linear classifier...")
    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.to(device), label.to(device)
            z, _ = model(img, img)
            output = linear(z)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    print("Accuracy: ", correct / total)
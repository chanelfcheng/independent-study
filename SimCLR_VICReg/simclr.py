import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import *

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
        print(loss.mean())
        return loss.mean()
    

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
    
    # Check if simclr_model.pth exists
    if not os.path.exists('simclr_model.pth'):
        # Create dataset and dataloader
        cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        images = cifar10_dataset.data
        dataset = AugmentedPairDataset(images, transform=data_transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Setup device, model, and optimizer
        device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')
        model = BaseModel().to(device)
        nt_xent_loss = NTXentLoss(temperature=0.1)
        # optimizer = LARSOptimizer(model.parameters(), lr=0.3*batch_size/256,
        # weight_decay=1e-6)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader), eta_min=0, last_epoch=-1)

        # Train the model
        epochs = 10
        model.train()
        print("Start training SimCLR...")
        for epoch in range(epochs):
            for img1, img2 in dataloader:
                img1, img2 = img1.to(device), img2.to(device)
                optimizer.zero_grad()
                z_i, z_j = model(img1, img2)
                loss = nt_xent_loss(z_i, z_j)
                loss.backward()
                optimizer.step()
            scheduler.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        print("Training complete.")

        # Save the model
        torch.save(model.state_dict(), 'simclr_model.pth')
        model.eval()
    else:
        # Load the model
        device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')
        model = BaseModel().to(device)
        model.load_state_dict(torch.load('simclr_model.pth'))
        model.eval()

    # Use representations learned by model to train linear classifier
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
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
            # z = img.view(img.size(0), -1)
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
            # z = img.view(img.size(0), -1)
            output = linear(z)
            _, predicted = torch.max(output.data, 1)
            print("Predicted: ", predicted)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    print("Accuracy: ", correct / total)
    
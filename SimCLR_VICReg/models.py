import torch.nn as nn
from torchvision.models import resnet50


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.encoder = resnet50(pretrained=False)
        self.encoder.fc = nn.Identity()  # Remove the final layer
        self.projector = Projector()
    
    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return z_i, z_j


class Projector(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=128):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        x = self.fc(x)
        return x
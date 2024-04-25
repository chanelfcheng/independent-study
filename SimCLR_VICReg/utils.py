import torch
from torch.utils.data import Dataset
import numpy as np
from simclr import *

import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50

import re
import math


class LARSOptimizer(optim.Optimizer):
    def __init__(self, params, lr, momentum=0.9, use_nesterov=False, weight_decay=0.0, exclude_from_weight_decay=None, exclude_from_layer_adaptation=None, classic_momentum=True, eeta=0.001):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        
        defaults = dict(lr=lr, momentum=momentum, use_nesterov=use_nesterov, weight_decay=weight_decay, classic_momentum=classic_momentum, eeta=eeta)
        super(LARSOptimizer, self).__init__(params, defaults)
        
        self.exclude_from_weight_decay = exclude_from_weight_decay
        if exclude_from_layer_adaptation is not None:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            learning_rate = group['lr']
            eeta = group['eeta']
            use_nesterov = group['use_nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if self._use_weight_decay(p.name):
                    grad = grad.add(p.data, alpha=weight_decay)
                
                param_state = self.state[p]
                
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(grad, alpha=1)
                
                if self._do_layer_adaptation(p.name):
                    w_norm = torch.norm(p.data, p=2)
                    g_norm = torch.norm(buf, p=2)
                    trust_ratio = eeta * w_norm / g_norm
                    scaled_lr = learning_rate * trust_ratio
                else:
                    scaled_lr = learning_rate
                
                if use_nesterov:
                    p.add_(buf, alpha=-momentum * scaled_lr).add_(grad, alpha=-scaled_lr)
                else:
                    p.add_(buf, alpha=-scaled_lr)
                
        return loss
    
    def _use_weight_decay(self, param_name):
        if not self.defaults['weight_decay']:
            return False
        if self.exclude_from_weight_decay is None:
            return True
        return all(re.search(r, param_name) is None for r in self.exclude_from_weight_decay)
    
    def _do_layer_adaptation(self, param_name):
        if self.exclude_from_layer_adaptation is None:
            return True
        return all(re.search(r, param_name) is None for r in self.exclude_from_layer_adaptation)


class CosineLearningRateScheduler:
    def __init__(self, optimizer, base_learning_rate, num_examples, train_batch_size, warmup_epochs, total_epochs, learning_rate_scaling='linear'):
        self.optimizer = optimizer
        self.base_learning_rate = base_learning_rate
        self.num_examples = num_examples
        self.train_batch_size = train_batch_size
        self.warmup_steps = int(round(warmup_epochs * num_examples / train_batch_size))
        self.total_steps = total_epochs * num_examples / train_batch_size
        self.global_step = 0
        self.learning_rate_scaling = learning_rate_scaling

        if learning_rate_scaling == 'linear':
            self.scaled_lr = base_learning_rate * train_batch_size / 256.
        elif learning_rate_scaling == 'sqrt':
            self.scaled_lr = base_learning_rate * math.sqrt(train_batch_size)
        else:
            raise ValueError(f'Unknown learning rate scaling {learning_rate_scaling}')

    def step(self):
        self.global_step += 1
        if self.warmup_steps:
            warmup_lr = self.global_step / self.warmup_steps * self.scaled_lr
        else:
            warmup_lr = self.scaled_lr

        if self.global_step < self.warmup_steps:
            learning_rate = warmup_lr
        else:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (self.global_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
            decayed_lr = self.scaled_lr * cosine_decay
            learning_rate = decayed_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def get_lr(self):
        # Optionally implement this method if you need to retrieve the current learning rate
        if self.global_step < self.warmup_steps:
            return self.global_step / self.warmup_steps * self.scaled_lr
        else:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (self.global_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
            return self.scaled_lr * cosine_decay


class AugmentedPairDataset(Dataset):
    def __init__(self, images, transform=None, subset_size=None):
        self.images = images
        self.transform = transform
        if subset_size is not None:
            # Ensure subset_size does not exceed the total number of images
            subset_size = min(subset_size, len(images))
            indices = np.random.choice(len(images), subset_size, replace=False)
            self.images = images[sorted(indices)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)
        else:
            img1, img2 = img, img
        
        return img1, img2


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
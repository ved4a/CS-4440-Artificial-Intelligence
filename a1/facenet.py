import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceNet, self).__init__()
        self.embedding_size = embedding_size

        # lightweight CNN to reduce computation on large datasets
        self.conv_layers = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4,4))
        )
        self.fc = nn.Linear(128*4*4, embedding_size)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)

        # L2-normalize embeddings for similarity computation
        x = F.normalize(self.fc(x), p=2, dim=1)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
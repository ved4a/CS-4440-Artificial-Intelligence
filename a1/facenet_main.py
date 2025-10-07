import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from facenet import FaceNet
from preprocessing import loadImages

class FaceDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.astype(np.float32)/255.0
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = np.expand_dims(self.images[idx], axis=0)
        label = self.labels[idx]
        return torch.tensor(img, dtype=torch.float32), label

# load dataset

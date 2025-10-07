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
root_dirs = ['a1/indian-face-dataset/train', 'a1/indian-face-dataset/val']
images, labels, label_map = loadImages(root_dirs, size=(100, 100))
dataset = FaceDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# initialize model & optimizer
device = 'cpu'
model = FaceNet(embedding_size=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# trainin loop
model.train()
for epoch in range(5):
    for batch_imgs, batch_labels in dataloader:
        batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        embeddings = model(batch_imgs)
        logits = embeddings @ embeddings.T
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")

# save
model.save('results/facenet_model.pth')
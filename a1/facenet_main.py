import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

from facenet import FaceNet
from preprocessing import createSplit
from metrics import classification_summary, rank_k_identification 


class FaceDataset(Dataset):
    def __init__(self, images, image_size=(100, 100)):
        self.images = images.astype(np.float32) / 255.0
        self.h, self.w = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].reshape(self.h, self.w)
        img = np.expand_dims(img, axis=0)  # add channel dimension
        return torch.tensor(img, dtype=torch.float32)


def embed_all(model, loader, device='cpu'):
    model.eval()
    all_emb = []
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device)
            all_emb.append(model(imgs).cpu().numpy())
    return np.vstack(all_emb)


def main():
    # data split (consistent with Eigenfaces pipeline)
    root_dirs = [
        os.path.join('A1', 'indian-face-dataset', 'train'),
        os.path.join('A1', 'indian-face-dataset', 'val'),
    ]
    img_size = (100, 100)
    X_train, X_test, y_train, y_test, _ = createSplit(root_dirs, testSize=0.3, size=img_size)

    # dataloaders
    train_loader = DataLoader(FaceDataset(X_train, img_size), batch_size=64, shuffle=True)
    full_train_loader = DataLoader(FaceDataset(X_train, img_size), batch_size=256, shuffle=False)
    test_loader = DataLoader(FaceDataset(X_test, img_size), batch_size=256, shuffle=False)

    # model and optimizer
    device = 'cpu'
    model = FaceNet(embedding_size=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # lightweight embedding training (unsupervised proxy)
    model.train()
    epochs = 5
    for epoch in range(epochs):
        running = 0.0
        for batch_idx, batch_imgs in enumerate(train_loader):
            batch_imgs = batch_imgs.to(device)
            optimizer.zero_grad()
            emb = model(batch_imgs)
            # encourage spread-out embeddings via variance maximization
            loss = -torch.var(emb, dim=0).mean()
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"Epoch {epoch+1}/{epochs} loss: {running/(batch_idx+1):.4f}")

    # ensure results dir
    results_dir = os.path.join('A1', 'results')
    os.makedirs(results_dir, exist_ok=True)
    model.save(os.path.join(results_dir, 'facenet_model.pth'))

    # embeddings and evaluation
    train_emb = embed_all(model, full_train_loader, device)
    test_emb = embed_all(model, test_loader, device)

    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf.fit(train_emb, y_train)
    preds = clf.predict(test_emb)

    acc = accuracy_score(y_test, preds)
    print(f"FaceNet embedding kNN accuracy: {acc*100:.2f}%")

    # classification summary (precision/recall/F1 + confusion matrix)
    summary = classification_summary(y_test, preds, labels=np.unique(y_train), print_report=True)
    cm = summary["confusion_matrix"]
    with open(os.path.join(results_dir, 'facenet_classification_report.txt'), 'w') as f:
        f.write(summary["report_text"])
    np.savetxt(os.path.join(results_dir, 'facenet_confusion_matrix.csv'), cm, fmt='%d', delimiter=',')

    # identification rates (rank-1 == accuracy, rank-5)
    ir = rank_k_identification(train_emb, y_train, test_emb, y_test, ks=(1, 5))
    rank1 = ir["rank_1"]
    rank5 = ir["rank_5"]
    print(f"Identification rate (Rank-1): {rank1*100:.2f}%")
    print(f"Identification rate (Rank-5): {rank5*100:.2f}%")

     # save embeddings and metrics
    np.save(os.path.join(results_dir, 'facenet_train_embeddings.npy'), train_emb)
    np.save(os.path.join(results_dir, 'facenet_test_embeddings.npy'), test_emb)

     # write a compact metrics CSV
    header = "accuracy,balanced_accuracy,precision_macro,recall_macro,f1_macro,precision_weighted,recall_weighted,f1_weighted,rank1,rank5"
    values = [
        summary["accuracy"],
        summary["balanced_accuracy"],
        summary["precision_macro"],
        summary["recall_macro"],
        summary["f1_macro"],
        summary["precision_weighted"],
        summary["recall_weighted"],
        summary["f1_weighted"],
        rank1,
        rank5,
    ]

    np.savetxt(os.path.join(results_dir, 'facenet_metrics.csv'), np.array([values]), delimiter=',', header=header, comments='')


if __name__ == '__main__':
    main()
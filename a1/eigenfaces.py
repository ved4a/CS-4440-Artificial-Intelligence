import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
import matplotlib.pyplot as plt

class Eigenfaces:
    def __init__(self, n_components=0.95):
        self.n_components = n_components
        self.pca = None
        self.knn = None
        self.label_map = None

    def fit(self, X_train, y_train, n_neighbors=1):
        print("Step 1: Fit PCA")
        self.pca = PCA(n_components=self.n_components, svd_solver='full') # keep components that explain 95% variance
        X_train_pca = self.pca.fit_transform(X_train)
        print(f"Original dim: {X_train.shape[1]}, Reduced dim: {X_train_pca.shape[1]}")

        print("Step 2: Train K-Nearest Neighbors Classifier")
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean') # classify based on SINGLE closest training example
        self.knn.fit(X_train_pca, y_train)
        print("Training complete")

    def predict(self, X_test):
        X_test_pca = self.pca.transform(X_test)
        return self.knn.predict(X_test_pca)

    def evaluate(self, X_test, y_test, label_map):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        print("Recognition Identification Rate (Accuracy):", accuracy)
        print("Precision:", precision)
        print("\nDetailed classification report:\n")
        print(classification_report(y_test, y_pred, target_names=list(label_map.keys())))
        return accuracy, precision

    def plot_eigenfaces(self, h, w, n_faces=10):
        eigenfaces = self.pca.components_.reshape((-1, h, w))
        plt.figure(figsize=(12, 6))
        for i in range(n_faces):
            plt.subplot(2, n_faces//2, i+1)
            plt.imshow(eigenfaces[i], cmap='gray')
            plt.title(f"Eigenface {i+1}")
            plt.axis('off')
        plt.show()
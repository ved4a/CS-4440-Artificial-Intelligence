import numpy as np

class Eigenfaces:
    def __init__(self, numComponents=None):
        self.numComponents = numComponents
        self.meanFace = None
        self.eigenfaces = None
        self.eigenvalues = None

    def fit(self, X):
        print(">> Begin Eigenfaces training")
        print("Step 1: Compute mean face")
        self.meanFace = np.mean(X, axis=0)
        print("Mean face computed.")

        print("Step 2: Subtract mean")
        A = (X- self.meanFace).T
        m = X.shape[0]
        print("Mean subtracted.")

        print("Step 3: Compute small covariance matrix L = (1/m) * A^T * A")
        L = (A.T @ A) / m
        print("Covariance matrix L computed (size: {}x{}).".format(*L.shape))

        print("Step 4: Eigen decomposition of L")
        eigenvalues, eigenvectorsSmall = np.linalg.eig(L)
        print("Eigen decomposition complete.")

        print("Step 5: Compute Eigenfaces from small Eigenvectors")
        eigenfaces = A @ eigenvectorsSmall
        print("Eigenfaces computed")

        print("Step 6: Normalize Eigenfaces")
        eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0, keepdims=True)
        print("Eigenfaces normalized.")

        print("Step 7: Sort Eigenfaces by descending Eigenvalue")
        idx = np.argsort(-eigenvalues.real)
        eigenvalues = eigenvalues[idx].real
        eigenfaces = eigenfaces[:, idx].real
        print("Eigenfaces sorted.")

        print("Step 8: Keep top 'k' Eigenfaces")
        if self.numComponents is not None:
            eigenfaces = eigenfaces[:, :self.numComponents]
            eigenvalues = eigenvalues[:self.numComponents]
        self.eigenfaces = eigenfaces
        self.eigenvalues = eigenvalues
        print("Retained {self.eigenfaces.shape[1]} Eigenfaces.")

    def project(self, X):
        if self.meanFace is None or self.eigenfaces is None:
            raise ValueError("Model not fitted yet. Call fit(X) first.")
        
        A = (X - self.meanFace)
        coeffs = A @ self.eigenfaces
        return coeffs
    
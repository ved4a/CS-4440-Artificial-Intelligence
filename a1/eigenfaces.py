import numpy as np

class Eigenfaces:
    def __init__(self, numComponents=None):
        self.numComponents = numComponents
        self.meanFace = None
        self.eigenfaces = None
        self.eigenvalues = None

    def fit(self, X):
        print("Step 1: Compute mean face")
        self.meanFace = np.mean(X, axis=0)
        print("Mean face computed.")

        print("Step 2: Subtract mean")
        A = (X- self.meanFace).T
        m, nPixels = X.shape
        print("Mean subtracted.")

        if m < nPixels:
            print("Step 3: Compute small covariance matrix L = (1/m) * A^T * A")
            L = (A.T @ A) / m
            print(f"Covariance matrix L computed (size: {L.shape[0]}x{L.shape[1]}).")

            print("Step 4: Eigen decomposition of L")            
            eigenvalues, eigenvectorsSmall = np.linalg.eigh(L)
            print("Eigen decomposition complete.")

            print("Step 5: Compute Eigenfaces from small Eigenvectors")
            eigenfaces = A @ eigenvectorsSmall
        else:
            print("Step 3: Using C = (1/m) * A * A^T (size N^2 x N^2)")
            C = (A @ A.T) / m
            print(f"Covariance matrix C computed (size: {C.shape[0]}x{C.shape[1]}).")

            print("Step 4 & 5: Eigen decomposition of C and compute Eigenfaces")
            eigenvalues, eigenfaces = np.linalg.eigh(C)
            print("Eigen decomposition and computation complete.")

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
        print(f"Retained {self.eigenfaces.shape[1]} Eigenfaces.")

    def project(self, X):
        if self.meanFace is None or self.eigenfaces is None:
            raise ValueError("Model not fitted yet. Call fit(X) first.")
        
        A = (X - self.meanFace)
        coeffs = A @ self.eigenfaces
        return coeffs
    
    def reconstruct(self, coeffs):
        return coeffs @ self.eigenfaces.T + self.meanFace

    def recognize(self, X_train, y_train, X_test):
        print(">> Projecting training and test faces")
        trainProjection = self.project(X_train)
        testProjection = self.project(X_test)

        predictions = []
        for testVector in testProjection:
            distances = np.linalg.norm(trainProjection - testVector, axis=1)
            predictions.append(y_train[np.argmin(distances)])
        return np.array(predictions)
import numpy as np
from preprocessing import createSplit
from eigenfaces import Eigenfaces
from sklearn.metrics import accuracy_score

def main():
    print("Step 1: Loading and preprocessing dataset")
    root_dirs = [
        'a1/indian-face-dataset/train',
        'a1/indian-face-dataset/val'
    ]
    
    X_train, X_test, y_train, y_test, label_map = createSplit(root_dirs, testSize=0.3)

    print("\nSTEP 2: Training Eigenfaces model")
    numComponents = min(100, X_train.shape[0])  # cap at 100 for stability
    model = Eigenfaces(numComponents=numComponents)
    model.fit(X_train)

    # save the model for later
    model.save('a1/results/eigenfaces_model.npz')

    print("\nSTEP 3: Recognizing test images...")
    preds = model.recognize(X_train, y_train, X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\nRESULT: Recognition accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
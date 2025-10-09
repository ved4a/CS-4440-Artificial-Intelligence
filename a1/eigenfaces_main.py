import numpy as np
from preprocessing import createSplit
from eigenfaces import Eigenfaces
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import cv2

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

    # ensure results directory
    results_dir = 'a1/results'
    os.makedirs(results_dir, exist_ok=True)

    # save the model for later
    model.save(os.path.join(results_dir, 'eigenfaces_model.npz'))

    # save mean face and first K eigenfaces for visualization
    side = int(np.sqrt(X_train.shape[1]))
    def to_img(x):
        x = x.reshape(side, side)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return (x * 255).astype(np.uint8)

    cv2.imwrite(os.path.join(results_dir, 'mean_face.png'), to_img(model.meanFace))
    k_vis = min(10, model.eigenfaces.shape[1])
    for i in range(k_vis):
        cv2.imwrite(os.path.join(results_dir, f'eigenface_{i+1:02d}.png'), to_img(model.eigenfaces[:, i]))

    print("\nSTEP 3: Recognizing test images...")
    preds = model.recognize(X_train, y_train, X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\nRESULT: Recognition accuracy: {acc * 100:.2f}%")

    # log confusion matrix and classification report
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)
    np.savetxt(os.path.join(results_dir, 'eigenfaces_confusion_matrix.csv'), cm, fmt='%d', delimiter=',')
    with open(os.path.join(results_dir, 'eigenfaces_classification_report.txt'), 'w') as f:
        f.write(report)

if __name__ == "__main__":
    main()
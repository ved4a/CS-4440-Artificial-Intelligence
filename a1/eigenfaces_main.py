import numpy as np
from preprocessing import createSplit
from eigenfaces import Eigenfaces
from sklearn.metrics import accuracy_score
import os
import cv2
from metrics import classification_summary, rank_k_identification

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

    # detailed metrics via shared helper
    summary = classification_summary(y_test, preds, labels=np.unique(y_train), print_report=True)
    cm = summary["confusion_matrix"]
    np.savetxt(os.path.join(results_dir, 'eigenfaces_confusion_matrix.csv'), cm, fmt='%d', delimiter=',')
    with open(os.path.join(results_dir, 'eigenfaces_classification_report.txt'), 'w') as f:
        f.write(summary["report_text"])

    # compute Rank-1/Rank-5 identification rates using PCA projections
    train_feats = model.project(X_train)
    test_feats = model.project(X_test)
    ir = rank_k_identification(train_feats, y_train, test_feats, y_test, ks=(1, 5))
    rank1 = ir["rank_1"]
    rank5 = ir["rank_5"]
    print(f"Identification rate (Rank-1): {rank1*100:.2f}%")
    print(f"Identification rate (Rank-5): {rank5*100:.2f}%")

    # persist a compact metrics CSV for easy comparison
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
    np.savetxt(os.path.join(results_dir, 'eigenfaces_metrics.csv'), np.array([values]), delimiter=',', header=header, comments='')

if __name__ == "__main__":
    main()
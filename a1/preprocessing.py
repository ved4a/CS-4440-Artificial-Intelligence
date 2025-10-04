import cv2
import os
import numpy as np
import re
import glob
from sklearn.model_selection import train_test_split

def canonicalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r'[^a-z]', '', name)
    return name

def loadImages(rootDirectories, size=(100, 100)):
    images, labels = [],[]
    labelMap = {}
    counter  = 0

    for rootDirectory in rootDirectories:
        for person in os.listdir(rootDirectory):
            personPath = os.path.join(rootDirectory, person)
            if not os.path.isdir(personPath):
                continue

            canonical = canonicalize_name(person)
            if canonical not in labelMap:
                labelMap[canonical] = counter
                counter += 1

            for ext in ("*.jpg", "*.png", "*.jpeg"):
                for imagePath in glob.glob(os.path.join(personPath, ext)):
                    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue
                    imageResized = cv2.resize(image, size)
                    images.append(imageResized.flatten())
                    labels.append(labelMap[canonical])
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, labelMap

def createSplit(rootDirectories, testSize=0.3, size=(100, 100), randomState=42):
    X, y, labelMap = loadImages(rootDirectories, size=size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState, stratify=y)
    print(f"Total classes: {len(labelMap)}")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, labelMap

if __name__ == "__main__":
    root_dirs = ['indian-face-dataset/train', 'indian-face-dataset/val']
    X_train, X_test, y_train, y_test, label_map = createSplit(root_dirs)
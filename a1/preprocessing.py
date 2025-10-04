import cv2
import os
import numpy as np
import re
import glob

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
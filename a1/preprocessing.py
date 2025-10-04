import cv2
import os
import numpy as np
import re

import re

def canonicalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r'[^a-z]', '', name)
    return name



def loadImages(dataDirectory, size=(100, 100), allowedIDs=None):
    images, labels = [],[]
    labelMap = {}
    counter  = 0

    for person in os.listdir(dataDirectory):
        personPath = os.path.join(dataDirectory, person)
        if not os.path.isdir(personPath):
            continue

        canonical = canonicalize_name(person)
        if allowedIDs and canonical not in allowedIDs:
            continue
        if canonical not in lavelMap:
            labelMap[canonical] = counter
            counter += 1
        
        for imageName in os.listdir(personPath):
            imagePath = os.path.join(personPath, imageName)
            if not imagePath.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            imageResized = cv2.resize(image, size)
            images.append(imageResized.flatten())
            labels.append(labelMap[canonical])
    
    return np.array(images), np.array(labels), labelMap
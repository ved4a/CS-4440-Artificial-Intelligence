import cv2
import os
import numpy as np

def loadImages(dataDirectory, size=(100, 100)):
    images, labels = [],[]
    labelMap = {}
    counter  = 0

    for person in os.listdir(dataDirectory):
        personPath = os.path.join(dataDirectory, person)
        if not os.path.isdir(personPath):
            continue

        if person not in labelMap:
            labelMap[person] = counter
            counter += 1
        
        for imageName in os.listdir(personPath):
            imagePath = os.path.join(personPath, imageName)
            image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            imageResized = cv2.resize(image, size)
            images.append(imageResized.flatten())
            labels.append(labelMap[person])
    
    return np.array(images), np.array(labels), labelMap
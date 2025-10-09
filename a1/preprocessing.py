import cv2
import os
import numpy as np
import re
import glob
from sklearn.model_selection import train_test_split

# if the same person has 2 folders in diff formats -> standardize
def canonicalizeName(name: str) -> str:
    name = name.lower()
    name = re.sub(r'[^a-z]', '', name)
    return name

# for the instances that:
#   i) train & val have images (& thus folders) of the same person
#   ii) train has 1+ folders of the same person
def mergeFolders(rootDirectories, tempMergedDirectory="merged_faces"):
    if not os.path.exists(tempMergedDirectory):
        os.makedirs(tempMergedDirectory)
    
    for rootDirectory in rootDirectories:
        for person in os.listdir(rootDirectory):
            personPath = os.path.join(rootDirectory, person)
            if not os.path.isdir(personPath):
                continue
            
            canonical = canonicalizeName(person)
            mergedPersonPath = os.path.join(tempMergedDirectory, canonical)
            os.makedirs(mergedPersonPath, exist_ok=True)

            for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"):
                for imagePath in glob.glob(os.path.join(personPath, ext)):
                    imageName = os.path.basename(imagePath)
                    targetPath = os.path.join(mergedPersonPath, imageName)

                    if not os.path.exists(targetPath):
                        try:
                            os.link(imagePath, targetPath)
                        except:
                            import shutil
                            shutil.copy(imagePath, targetPath)
    return tempMergedDirectory

# when counting avg # of images per folder, remove outliers to prevent skew
def removeOutliers(counts):
    counts = np.array(counts)
    q1, q3 = np.percentile(counts, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered = counts[(counts >= lower_bound) & (counts <= upper_bound)]
    return filtered

def loadImages(rootDirectories, size=(100, 100)):
    # process images from merged folder(s)
    mergedDir = mergeFolders(rootDirectories)

    # remove folders (aka people) w/ only a few photos
    # "few" being defined as < avg (removing outliers)
    imageCounts = {}
    for person in os.listdir(mergedDir):
        personPath = os.path.join(mergedDir, person)
        if not os.path.isdir(personPath):
            continue
        count = sum(len(glob.glob(os.path.join(personPath, ext))) for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"))
        imageCounts[person] = count
    
    filteredCounts = removeOutliers(list(imageCounts.values()))
    avgCount = np.mean(filteredCounts)
    print(f"Average # images per person (excluding outliers): {avgCount:.2f}")

    validPeople = [p for p, c in imageCounts.items() if c >= avgCount]
    print(f"Selected {len(validPeople)} people with ≥ average images")

    # load the images
    images, labels = [],[]
    labelMap = {}
    counter  = 0

    for person in validPeople:
        personPath = os.path.join(mergedDir, person)
        labelMap[person] = counter
        counter += 1

        for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"):
            for imagePath in glob.glob(os.path.join(personPath, ext)):
                image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                imageResized = cv2.resize(image, size)
                images.append(imageResized.flatten())
                labels.append(labelMap[person])
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, labelMap

def createSplit(rootDirectories, testSize=0.3, size=(100, 100), randomState=42):
    X, y, labelMap = loadImages(rootDirectories, size=size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState, stratify=y)
    print(f"Total classes post-filtering: {len(labelMap)}")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, labelMap
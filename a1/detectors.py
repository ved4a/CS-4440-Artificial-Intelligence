import cv2

_haar = None


def _get_haar():
    global _haar
    if _haar is None:
        try:
            path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            _haar = cv2.CascadeClassifier(path)
        except Exception:
            _haar = None
    return _haar


def detect_faces_haar(gray_img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)):
    haar = _get_haar()
    if haar is None:
        return []
    rects = haar.detectMultiScale(gray_img, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    boxes = []
    for (x, y, w, h) in rects:
        boxes.append([int(x), int(y), int(w), int(h)])
    return boxes

def detect_faces(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    boxes = detect_faces_haar(gray)
    return boxes

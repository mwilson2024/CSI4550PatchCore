import cv2
import numpy as np

def applyMorphology(binaryMask: np.ndarray, operation: str = 'opening', kernel = 3) -> np.ndarray:

    if kernel < 1 or kernel %2 == 0:
        raise ValueError(f'Kernal Size is: {kernel} needs to be and odd int')

    if binaryMask.ndim != 2:
        raise ValueError(f'Ivalid Dimensions for Array Current: {binaryMask.ndim}')

    mask = (binaryMask > 0).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))

    operations ={'erosion': cv2.MORPH_ERODE, 'dilation': cv2.MORPH_DILATE, 'opening': cv2.MORPH_OPEN, 'closing': cv2.MORPH_CLOSE}

    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")

    return cv2.morphologyEx(mask, operations[operation], kernel)
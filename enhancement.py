import cv2
import numpy as np

def claheEnhancement(img:np.ndarray, clipLimit: float = 2.0, grid: int = 8) -> np.ndarray:

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lChannel, aChannel, bChannel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid, grid))

    enhanceL = clahe.apply(lChannel)

    enhancedLab = cv2.merge((enhanceL, aChannel, bChannel))

    return cv2.cvtColor(enhancedLab, cv2.COLOR_LAB2BGR)

def gammaCorect(img:np.ndarray, gamma: float = 0.5) -> np.ndarray:

    if gamma <= 0:
        raise ValueError(f'Gamma is: {gamma} must be greater then 0')

    table = np.array([((i / 255.0) ** gamma) * 255
    for i in range(256)]).astype(np.uint8)

    return cv2.LUT(img, table)

def denoising(img: np.ndarray, kernel: int = 3) -> np.ndarray:

    if kernel <= 0 or kernel %2 == 0:
            raise ValueError(f'Kernal Size is: {kernel} needs to be and odd int')
    
    return cv2.medianBlur(img, kernel)
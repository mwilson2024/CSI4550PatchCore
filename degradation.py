import cv2
import numpy as np

def darken(img: np.ndarray, factor: float = .5) -> np.ndarray:
    
    if factor <= 0 or factor >= 1:
        raise ValueError(f'Factor is currently {factor} must be between 0 and 1')

    return np.clip(img.astype(np.float32) * factor, 0 , 255).astype(np.uint8)

def gaussianBlur(img: np.ndarray, kernel=5, sig: float = 1.5)-> np.ndarray:
    if kernel <= 0 or kernel %2 == 0:
        raise ValueError(f'Kernal Size is: {kernel} needs to be and odd int')

    return cv2.GaussianBlur(img,(kernel, kernel), sig)

def gaussianNoise(img: np.ndarray, sig: float = 15.0, seed=42)-> np.ndarray:

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sig, img.shape)

    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
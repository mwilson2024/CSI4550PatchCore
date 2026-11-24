from pathlib import Path
import hashlib
import cv2
from config import dataRoot, conditionsRoot, selectedCategories
from degradation import darken, gaussianBlur, gaussianNoise
from enhancement import claheEnhancement, gammaCorect, denoising

def prepareCategory(category: str) -> None:
    src = dataRoot / category / 'test'
    paths = sorted(src.rglob('*.png'))
    if not paths:
        raise FileNotFoundError(f'No test images found: {src.resolve()}')
    count = 0
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'Unable to read {path}')
        rel = path.relative_to(src)
        seed = int.from_bytes(hashlib.sha256(rel.as_posix().encode()).digest()[:4], 'little')
        dark = darken(image, 0.5)
        noisy = gaussianNoise(image, sig=15.0, seed=seed)
        transformed = {'original': image, "original_gamma": gammaCorect(image, gamma=0.5),'dark': dark,'dark_clahe': claheEnhancement(dark),'dark_gamma': gammaCorect(dark),
            'blurred': gaussianBlur(image, kernel=5, sig=1.5),'noisy': noisy,'noisy_median': denoising(noisy)}
        
        for condition, result in transformed.items():
            target = conditionsRoot / condition / category / 'test' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), result):
                raise RuntimeError(f'Unable to write {target}')
            
        count += 1
    print(f'{category}: prepared {count} images across {len(transformed)} conditions')


def main(categories=None):
    for category in selectedCategories(categories):
        prepareCategory(category)


if __name__ == '__main__':
    main()

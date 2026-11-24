from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from degradation import darken, gaussianBlur, gaussianNoise
from enhancement import claheEnhancement, gammaCorect, denoising

def previewCategory(category):
    ds = Path('datasets/MVTecAD') / category / 'test'

    imgPaths = sorted(ds.rglob('*.png'))

    if not imgPaths:
        raise FileNotFoundError('No file found')

    img = cv2.imread(str(imgPaths[0]))

    if img is None:
        raise RuntimeError(f'Could no read file {imgPaths[0]}')

    dark = darken(img, factor=.5)
    noisy = gaussianNoise(img, 15)
    blurred = gaussianBlur(img)

    imgs = {'Orignal': img, 'Dark': dark, 'Blurry': blurred, 'Noisy': noisy,
            'darkCLAHE': claheEnhancement(dark), 'darkGamma': gammaCorect(dark),
            'noisyMed': denoising(noisy)}

    print('Starting')
    output = Path('testing/results') / category

    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for ax, (name, img) in zip(axes.flat, imgs.items()):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax.imshow(rgb)
        ax.set_title(name)
        ax.axis("off")

    for ax in list(axes.flat)[len(imgs):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output / "preprocessing_comparison.png", dpi=150)
    plt.close(fig)


def main(categories=None):
    from config import selectedCategories
    for category in selectedCategories(categories):
        previewCategory(category)


if __name__ == "__main__":
    main()

import csv
import gc
from pathlib import Path
from multiprocessing import freeze_support
import cv2
import numpy as np
from anomalib.engine import Engine
from anomalib.models import Patchcore

from config import (conditionsRoot, resultsRoot, modelConfig, selectedCategories)
from modeltools import getCategoryCheckpoint
from eval import (collect, getAnomalibSplit, chooseThreshold, imgAuc, pixelAuc, maskMetrics)


fusionRoot = resultsRoot / "fusion"

# Fixed brightness cutoffs and gamma weight.
darkLevel = 0.25
brightLevel = 0.55
maxGammaWeight = 0.75


def brightness(imagePath: Path) -> float:
    image = cv2.imread(str(imagePath))

    if image is None:
        raise FileNotFoundError(imagePath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return float(gray.mean() / 255.0)


def adaptiveWeight(imagePath: Path) -> float:
    value = brightness(imagePath)

    amount = (brightLevel - value) / (brightLevel - darkLevel)

    return float(maxGammaWeight * np.clip(amount, 0.0, 1.0))


def calibration(rows: list[dict]) -> tuple[float, float]:
    """
    Determine a common score range for one view
    using only original-condition validation images.

    Subsample pixels for memory efficiency.
    """
    pixels = np.concatenate([r["map"].ravel()[::4096] for r in rows])

    low, high = np.percentile(pixels, [1.0, 99.5])

    if high <= low:
        raise RuntimeError("Invalid map calibration range")

    return float(low), float(high)



def normalizeMap(anomalyMap: np.ndarray,limits: tuple[float, float]) -> np.ndarray:

    low, high = limits

    # Leave scores above the range unclipped.
    return ((anomalyMap - low) / (high - low)).astype(np.float32)



def createFusion(baseRows: list[dict], gammaRows: list[dict], baseLimits: tuple[float, float],
                 gammaLimits: tuple[float, float], method: str) -> list[dict]:

    gammaByPath = {r["relative"]: r for r in gammaRows}

    fused = []

    for row in baseRows:
        key = row["relative"]

        if key not in gammaByPath:
            raise RuntimeError(f"Missing gamma prediction: {key}")

        gammaRow = gammaByPath[key]

        originalMap = normalizeMap(row["map"], baseLimits)

        gammaMap = normalizeMap(gammaRow["map"], gammaLimits)

        if originalMap.shape != gammaMap.shape:
            raise RuntimeError(f"Map size mismatch: {key}")

        if method == "original":
            weight = 0.0

        elif method == "gamma":
            weight = 1.0

        elif method == "fixed_50":
            weight = 0.5

        elif method == "adaptive":
            weight = adaptiveWeight(row["image_path"])

        else:
            raise ValueError(f"Unknown method: {method}")

        anomalyMap = ((1.0 - weight) * originalMap + weight * gammaMap).astype(np.float32)

        result = dict(row)

        result["map"] = anomalyMap

        # Use the map maximum for each method.
        result["score"] = float(anomalyMap.max())

        result["gamma_weight"] = weight

        fused.append(result)

    return fused


def savePreview(rows: list[dict], method: str, condition: str, threshold: float, output: Path) -> None:

    example = next((r for r in rows if r["label"] == 1), None)

    if example is None:
        return

    image = cv2.imread(str(example["image_path"]))

    h, w = example["map"].shape
    image = cv2.resize(image, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(np.clip(example["map"], 0, 1) * 255),cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    truth = np.uint8(example["mask"]) * 255

    prediction = np.uint8(example["map"] >= threshold) * 255

    truthBgr = cv2.cvtColor(truth, cv2.COLOR_GRAY2BGR)

    predBgr = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)

    panels = np.hstack([image, overlay, truthBgr, predBgr])

    target = output / (f"preview_{condition}_{method}.png")

    cv2.imwrite(str(target), panels)


def saveCSV(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluateCategory(category: str) -> list[dict]:
    print(f"\nFusion: {category}")

    checkpoint = getCategoryCheckpoint(category)

    model = Patchcore(**modelConfig)

    engine = Engine(accelerator="cpu", devices=1, default_root_dir=str(Path("results/fusion_engine") / category))

    validationKeys, testKeys = getAnomalibSplit(category)

    # Get both views from the saved model.
    original = collect(category, "original", engine, model, checkpoint)

    originalGamma = collect(category, "original_gamma", engine, model, checkpoint)

    dark = collect(category, "dark", engine, model, checkpoint)

    darkGamma = collect(category, "dark_gamma", engine, model, checkpoint)

    originalKeys = {r["relative"] for r in original}

    for other in (originalGamma, dark, darkGamma):
        if {r["relative"] for r in other} != originalKeys:
            raise RuntimeError(f"Image pairs differ in {category}")

    if validationKeys | testKeys != originalKeys:
        raise RuntimeError(f"Split does not match images in {category}")

    # Get the scales from clean validation images.
    baseValidation = [r for r in original if r["relative"] in validationKeys]

    gammaValidation = [r for r in originalGamma if r["relative"] in validationKeys]

    baseLimits = calibration(baseValidation)
    gammaLimits = calibration(gammaValidation)

    methods = ["original", "gamma", "fixed_50", "adaptive"]

    conditions = {"original": (original, originalGamma), "dark": (dark, darkGamma)}

    output = fusionRoot / category
    output.mkdir(parents=True, exist_ok=True)

    results = []

    for method in methods:

        # Pick the threshold on clean validation images.
        cleanFused = createFusion(original, originalGamma, baseLimits, gammaLimits, method)

        validation = [r for r in cleanFused if r["relative"] in validationKeys]

        threshold = chooseThreshold(validation)

        # Use the same threshold for both test conditions.
        for condition, (base, enhanced) in conditions.items():

            if condition == "original":
                fused = cleanFused
            else:
                fused = createFusion(base, enhanced, baseLimits, gammaLimits, method)

            test = [r for r in fused if r["relative"] in testKeys]

            f1, iou = maskMetrics(test, threshold, "none")

            erosionF1, erosionIou = maskMetrics(test, threshold, "erosion")

            result = {"category": category, "condition": condition, "method": method,
                      "n_images": len(test), "image_auroc": imgAuc(test), "pixel_auroc": pixelAuc(test),
                      "pixel_f1": f1, "pixel_iou": iou, "erosion_f1": erosionF1, "erosion_iou": erosionIou,
                      "threshold": threshold, "mean_gamma_weight": float(np.mean([r["gamma_weight"] for r in test]))}

            results.append(result)

            savePreview(test, method, condition, threshold, output)

            print(f"{category:12} " f"{condition:8} " f"{method:10} "
                  f"AUROC={result['image_auroc']:.4f} " f"F1={f1:.4f}")

    saveCSV(output / "fusion_scores.csv", results)

    del model, engine
    gc.collect()

    return results


def main(categories=None):
    from result_files import combineCategories
    for category in selectedCategories(categories):
        evaluateCategory(category)
        combineCategories(fusionRoot, 'fusion_scores.csv', 'fusion_scores_all.csv')
    print('Fusion results saved:', fusionRoot.resolve())


if __name__ == "__main__":
    freeze_support()
    main()

from multiprocessing import freeze_support
from pathlib import Path
import csv
import cv2
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.data import MVTecAD
from modeltools import lastestCheckPoint, getCategoryCheckpoint
from morphology import applyMorphology
from config import (dataRoot, conditionsRoot, resultsRoot, conditionOptions,
                    morphologyOperations, modelConfig, selectedCategories)



def getAnomalibSplit(category: str) -> tuple[set[Path], set[Path]]:
    dm = MVTecAD(root=str(dataRoot), category=category, train_batch_size=3, eval_batch_size=3,
        num_workers=0,val_split_mode="from_test", val_split_ratio=0.30, seed=42)

    dm.setup()

    testRoot = (dataRoot / category / "test").resolve()

    def collectPaths(loader) -> set[Path]:
        paths = set()

        for batch in loader:
            batchPaths = (batch.image_path 
            if hasattr(batch, "image_path")
            else batch["image_path"])

            if isinstance(batchPaths, (str, Path)):
                batchPaths = [batchPaths]

            for imagePath in batchPaths:
                relative = (Path(imagePath).resolve().relative_to(testRoot))
                paths.add(relative)

        return paths

    validation = collectPaths(dm.val_dataloader())
    testing = collectPaths(dm.test_dataloader())

    if not validation or not testing:
        raise RuntimeError(f"Empty validation or test split for {category}")

    overlap = validation & testing

    if overlap:
        raise RuntimeError(f"Validation/test overlap: {len(overlap)} images")

    print(f"{category}: \n{len(validation)} validation images, \n{len(testing)} test images")

    return validation, testing


def groundTruthFor(category: str, rel: Path, size: tuple[int, int]) -> np.ndarray:
    
    if rel.parts[0] == 'good':
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    source = dataRoot / category / 'ground_truth' / rel.parent / (rel.stem + '_mask.png')
    mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f'Ground turth mask missing {source.resolve()}')

    return (cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)

def collect(category: str, condition: str, engine: Engine, model: Patchcore, ckpt: Path) -> list[dict]:

    folder = conditionsRoot/condition/category /'test'

    if not folder.is_dir():
        raise FileNotFoundError(f'Missing folder: {folder}')

    batches = engine.predict(model=model, ckpt_path=str(ckpt), data_path=(folder), return_predictions=True)

    if not batches:
        raise RuntimeError(f'No predictions for {category}/{condition}')

    records = []

    for batch in batches:
        paths = batch.image_path

        if isinstance(paths, (str, Path)):
            paths = [paths]

        if not paths:
            raise RuntimeError(f'Prediction batch contains no image paths')

        scores = batch.pred_score.detach().cpu().numpy().reshape(-1)
        maps = batch.anomaly_map.detach().cpu().numpy()

        if len(scores) != len(paths) or len(maps) != len(paths):
                    raise RuntimeError('Prediction batch shape/path mismatch')
        for i, p in enumerate(paths):
            imagePath = Path(p)
            relative = imagePath.resolve().relative_to(folder.resolve())
            anomaly = np.squeeze(maps[i]).astype(np.float32)
            if anomaly.ndim != 2:
                raise ValueError(f'Expected 2D anomaly map, got {anomaly.shape}')
            h, w = anomaly.shape
            truth = groundTruthFor(category, relative, (w, h))
            records.append({'relative': relative, 
                            'label': int(relative.parts[0] != 'good'),
                            'score': float(scores[i]), 'map': anomaly,
                            'mask': truth, 'image_path': imagePath})
    expected = {p.relative_to(folder) for p in folder.rglob('*.png')}
    observed = [r['relative'] for r in records]
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError(f'Missing or duplicate predictions for {category}/{condition}')
    return sorted(records, key=lambda r: r['relative'].as_posix()) 

def imgAuc(rows: list[dict]) -> float:
    labels = [r['label'] for r in rows]
    if len(set(labels)) < 2:
        return float('nan')
    return float(roc_auc_score(labels, [r['score'] for r in rows]))

def pixelAuc(rows: list[dict]) -> float:
    labels = np.concatenate([r['mask'].ravel() for r in rows])
    scores = np.concatenate([r['map'].ravel() for r in rows])
    if np.unique(labels).size < 2:
        return float('nan')
    return float(roc_auc_score(labels, scores))

def maskMetrics(rows: list[dict], threshold: float, operation: str) -> tuple[float, float]:
    tp = fp = fn = 0
    for row in rows:
        predicted = (row['map'] >= threshold).astype(np.uint8)
        if operation != 'none':
            predicted = (applyMorphology(predicted, operation=operation, kernel=3) > 0).astype(np.uint8)
        actual = row['mask'].astype(bool)
        predicted = predicted.astype(bool)
        tp += int(np.count_nonzero(predicted & actual))
        fp += int(np.count_nonzero(predicted & ~actual))
        fn += int(np.count_nonzero(~predicted & actual))
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return float(f1), float(iou)

def chooseThreshold(validationRows: list[dict]) -> float:
    allValues = np.concatenate([r['map'].ravel() for r in validationRows])
    candidates = np.unique(np.quantile(allValues, np.linspace(0.50, 0.999, 70)))
    if candidates.size == 0:
        raise RuntimeError('No candidate thresholds')
    return max((float(t) for t in candidates),
               key=lambda t: maskMetrics(validationRows, t, 'none')[0])

def saveExample(rows: list[dict], condition: str, threshold: float, out: Path):

    record = next((r for r in rows if r['label'] == 1), None)
    if record is None:
        return
    img = cv2.imread(str(record['image_path']))
    if img is None:
        raise RuntimeError(f"Failed to load preview {record['image_path']}")
    h, w = record['map'].shape
    img = cv2.resize(img, (w, h))
    anomaly = cv2.normalize(record['map'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    overlay = cv2.addWeighted(img, 0.6, cv2.applyColorMap(anomaly, cv2.COLORMAP_JET), 0.4, 0)
    binary = (record['map'] >= threshold).astype(np.uint8) * 255
    opening = applyMorphology(binary, 'opening', 3)
    closing = applyMorphology(binary, 'closing', 3)
    toBgr = lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    panels = [img, overlay, toBgr(record['mask'] * 255), toBgr(binary), toBgr(opening), toBgr(closing)]
    if not cv2.imwrite(str(out / f'preview_{condition}.png'), np.hstack(panels)):
        raise RuntimeError('Unable to save preview')
    
def saveCsv(path: Path, rows: list[dict]):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def evaluateCategory(category: str):
    import json

    checkpoint = getCategoryCheckpoint(category)
    model = Patchcore(**modelConfig)
    engine = Engine(accelerator='cpu', devices=1, default_root_dir=str(Path('results/evaluation') / category))
    out = resultsRoot / category
    out.mkdir(parents=True, exist_ok=True)
    print(f'\nEvaluating {category} with {checkpoint}')

    validationKeys, testKeys = getAnomalibSplit(category)
    clean = collect(category, 'original', engine, model, checkpoint)

    validation = [r for r in clean if r["relative"] in validationKeys]

    if not validation:
        raise RuntimeError("No validation records found for {category}")

    allKeys = {r["relative"] for r in clean}
    if validationKeys | testKeys != allKeys:
        raise RuntimeError(f"Anomalib split does not match prepared images: {category}")
    
    threshold = chooseThreshold(validation)
    splitManifest = {"category": category, "seed": 42,
    "validation_ratio": 0.30, "validation": sorted(p.as_posix() for p in validationKeys),
    "test": sorted(p.as_posix() for p in testKeys)}

    (out / "split_manifest.json").write_text(json.dumps(splitManifest, indent=2), encoding="utf-8")

    scoreRows = []
    morphologyRows = []
    
    expected = {r['relative'] for r in clean}
    for condition in conditionOptions:
        rows = clean if condition == 'original' else collect(category, condition, engine, model, checkpoint)
        if {r['relative'] for r in rows} != expected:
            raise RuntimeError(f'Condition sets differ for {category}/{condition}')
        test = [r for r in rows if r['relative'] in testKeys]
        if not test:
            raise RuntimeError('Empty test split')
        scoreRows.append({'category': category, 'condition': condition,
                           'n_images': len(test), 'image_auroc': imgAuc(test),
                           'pixel_auroc': pixelAuc(test)})
        for operation in morphologyOperations:
            f1, iou = maskMetrics(test, threshold, operation)
            morphologyRows.append({'category': category, 'condition': condition,
                                    'operation': operation, 'pixel_f1': f1, 'iou': iou})
        saveExample(test, condition, threshold, out)
        print(f'[{category}] {condition} image AUROC: {scoreRows[-1]["image_auroc"]:.4f}')
    saveCsv(out / 'scores.csv', scoreRows)
    saveCsv(out / 'morphology.csv', morphologyRows)
    (out / 'threshold.txt').write_text(f'{threshold:.10f}\n')
    (out / 'split_warning.txt')
    return scoreRows, morphologyRows


def main(categories=None):
    from result_files import combineCategories
    for category in selectedCategories(categories):
        evaluateCategory(category)
        combineCategories(resultsRoot, 'scores.csv', 'scores_all.csv')
        combineCategories(resultsRoot, 'morphology.csv', 'morphology_all.csv')
    print(f'Results saved in {resultsRoot.resolve()}')


if __name__ == '__main__':
    freeze_support()
    main()
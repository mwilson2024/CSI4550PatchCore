"""Compare clean and condition-matched threshold calibration.

Results go in results/calibration. The model and split stay fixed.
"""

import csv
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from anomalib.data.utils import read_image
from anomalib.models import Patchcore
from sklearn.metrics import roc_auc_score

from config import (conditionOptions, conditionsRoot, resultsRoot,
                    calibrationRoot, defaultBatchSize, selectedCategories)
from eval import groundTruthFor
from modeltools import getCategoryCheckpoint
from thresholding import exactF1Threshold, pixelCounts, metricsFromCounts


def writeCsv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fileHash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def loadSplit(category):
    path = resultsRoot / category / 'split_manifest.json'
    split = json.loads(path.read_text(encoding='utf-8'))
    validation, test = set(split['validation']), set(split['test'])
    if (split['category'] != category or not validation or not test
            or validation & test or len(validation) != len(split['validation'])
            or len(test) != len(split['test'])):
        raise ValueError(f'Invalid or overlapping split: {path}')
    return split, validation, test


def predictCondition(model, category, condition, expected, batchSize):
    folder = conditionsRoot / condition / category / 'test'
    paths = sorted(folder.rglob('*.png'))
    relative = [p.relative_to(folder).as_posix() for p in paths]
    if set(relative) != expected:
        raise ValueError(f'Prepared images differ from saved split: {folder}')
    rawMaps, normalizedMaps, rawScores, normalizedScores = [], [], [], []
    # Match Engine preprocessing. export_transform uses different resizing.
    with torch.inference_mode():
        for start in range(0, len(paths), batchSize):
            images = [model.pre_processor.transform(read_image(p, as_tensor=True))
                      for p in paths[start:start + batchSize]]
            raw = model.model(torch.stack(images))
            normalized = model.post_processor(raw)
            rawMaps.append(raw.anomaly_map.cpu().numpy()[:, 0].copy())
            normalizedMaps.append(normalized.anomaly_map.cpu().numpy()[:, 0].copy())
            rawScores.extend(raw.pred_score.cpu().numpy().reshape(-1).tolist())
            normalizedScores.extend(normalized.pred_score.cpu().numpy().reshape(-1).tolist())
    rawMaps = np.concatenate(rawMaps)
    h, w = rawMaps.shape[-2:]
    masks = np.stack([groundTruthFor(category, Path(p), (w, h)) for p in relative])
    return {
        'relative': relative, 'paths': paths, 'masks': masks,
        'labels': np.array([Path(p).parts[0] != 'good' for p in relative]),
        'raw_maps': rawMaps, 'normalized_maps': np.concatenate(normalizedMaps),
        'raw_scores': np.array(rawScores), 'normalized_scores': np.array(normalizedScores)
    }


def savePreview(data, testIndices, baselineThreshold, calibratedThreshold, path):
    # Preview the first defective test image.
    i = next(i for i in testIndices if data['labels'][i])
    h, w = data['masks'][i].shape
    original = cv2.cvtColor(cv2.resize(cv2.imread(str(data['paths'][i])), (w, h)), cv2.COLOR_BGR2RGB)
    panels = [original, data['masks'][i],
              data['normalized_maps'][i] >= baselineThreshold,
              data['normalized_maps'][i] >= calibratedThreshold]
    titles = ['Input', 'Ground truth', 'Clean-only threshold', 'Condition calibration']
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, cmap='gray', vmin=0, vmax=1) if panel.ndim == 2 else ax.imshow(panel)
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(f'{path.parent.name} / {path.stem.removeprefix("preview_")} / {data["relative"][i]}')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def runCategory(category, conditions, output, batchSize):
    out = output / category
    out.mkdir(parents=True, exist_ok=True)
    split, validationKeys, testKeys = loadSplit(category)
    checkpoint = getCategoryCheckpoint(category)
    oldThresholdPath = resultsRoot / category / 'threshold.txt'
    oldThreshold = float(oldThresholdPath.read_text())
    with (resultsRoot / category / 'morphology.csv').open(newline='') as stream:
        existingMasks = {r['condition']: r for r in csv.DictReader(stream) if r['operation'] == 'none'}
    with (resultsRoot / category / 'scores.csv').open(newline='') as stream:
        existingScores = {r['condition']: r for r in csv.DictReader(stream)}
    model = Patchcore.load_from_checkpoint(str(checkpoint), map_location='cpu', weights_only=False)
    model.eval()
    rows, imageRows, aurocRows, choices = [], [], [], {}
    cleanExact = None
    for condition in conditions:
        start = perf_counter()
        print(f'[{category}/{condition}] predicting {len(validationKeys | testKeys)} images', flush=True)
        data = predictCondition(model, category, condition, validationKeys | testKeys, batchSize)
        validation = np.array([i for i, p in enumerate(data['relative']) if p in validationKeys])
        test = np.array([i for i, p in enumerate(data['relative']) if p in testKeys])
        validationMasks = data['masks'][validation]
        calibrated = exactF1Threshold(validationMasks, data['normalized_maps'][validation])
        rawCalibrated = exactF1Threshold(validationMasks, data['raw_maps'][validation])
        if condition == 'original':
            cleanExact = calibrated
        methods = [
            ('legacy_clean', 'normalized', oldThreshold, 'original'),
            ('exact_clean', 'normalized', cleanExact, 'original'),
            ('condition_calibrated', 'normalized', calibrated, condition),
            ('condition_raw', 'raw', rawCalibrated, condition)
        ]
        conditionRows = []
        for method, domain, threshold, calibrationCondition in methods:
            maps = data[f'{domain}_maps']
            counts = pixelCounts(data['masks'][test], maps[test], threshold)
            metrics = metricsFromCounts(counts)
            row = {
                'category': category, 'condition': condition, 'method': method,
                'score_domain': domain, 'calibration_condition': calibrationCondition,
                'threshold': threshold, 'n_validation': len(validation), 'n_test': len(test),
                **metrics
            }
            conditionRows.append(row)
            for i, (tp, fp, fn) in zip(test, counts):
                imageRows.append({'category': category, 'condition': condition, 'method': method,
                                   'relative': data['relative'][i], 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)})
        # Check that inference still matches the saved baseline.
        legacy = conditionRows[0]
        baselineError = max(abs(legacy[key] - float(existingMasks[condition][key]))
                             for key in ['pixel_f1', 'iou'])
        normalizedAuc = float(roc_auc_score(data['labels'][test], data['normalized_scores'][test]))
        if baselineError > 2e-4 or abs(normalizedAuc - float(existingScores[condition]['image_auroc'])) > 2e-4:
            raise RuntimeError(f'{category}/{condition}: baseline reproduction failed; '
                               f'mask error={baselineError}, image AUROC={normalizedAuc}')
        for domain in ['normalized', 'raw']:
            maps = data[f'{domain}_maps'][test]
            aurocRows.append({
                'category': category, 'condition': condition, 'score_domain': domain,
                'image_auroc': float(roc_auc_score(data['labels'][test], data[f'{domain}_scores'][test])),
                'pixel_auroc': float(roc_auc_score(data['masks'][test].ravel(), maps.ravel()))
            })
        if abs(aurocRows[-2]['pixel_auroc'] - float(existingScores[condition]['pixel_auroc'])) > 2e-4:
            raise RuntimeError(f'{category}/{condition}: baseline pixel AUROC reproduction failed')
        choices[condition] = {
            'condition_threshold': calibrated, 'raw_condition_threshold': rawCalibrated,
            'validation_f1': metricsFromCounts(pixelCounts(validationMasks, data['normalized_maps'][validation], calibrated))['pixel_f1'],
            'test_pixels_clipped_high_fraction': float(np.mean(data['normalized_maps'][test] == 1)),
            'baseline_max_mask_metric_error': baselineError
        }
        rows.extend(conditionRows)
        writeCsv(out / 'metrics.csv', rows)
        writeCsv(out / 'per_image_counts.csv', imageRows)
        writeCsv(out / 'auroc.csv', aurocRows)
        savePreview(data, test, oldThreshold, calibrated, out / f'preview_{condition}.png')
        print(f'[{category}/{condition}] F1 {legacy["pixel_f1"]:.4f} -> '
              f'{conditionRows[2]["pixel_f1"]:.4f}; raw {conditionRows[3]["pixel_f1"]:.4f}; '
              f'{perf_counter() - start:.1f}s', flush=True)
    metadata = {
        'category': category, 'conditions': conditions, 'split': split,
        'checkpoint_sha256': fileHash(checkpoint), 'split_sha256': fileHash(resultsRoot / category / 'split_manifest.json'),
        'legacy_threshold': oldThreshold, 'exact_clean_threshold': cleanExact, 'calibration': choices,
        'selection': 'Maximize pooled validation pixel F1; ties choose highest threshold; no morphology.',
        'limitation': 'Condition calibration requires known condition and labeled validation masks from that condition. '
                      'This is an exploratory follow-up on an already inspected test split.',
        'python': platform.python_version(),
        'packages': {name: version(name) for name in ['torch', 'torchvision', 'anomalib', 'numpy', 'scikit-learn']},
        'source_sha256': {name: fileHash(Path(name)) for name in ['calibrate_thresholds.py', 'thresholding.py', 'eval.py']}
    }
    (out / 'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return rows


def main(categories=None, conditions=conditionOptions, output=calibrationRoot,
         batchSize=defaultBatchSize):
    if batchSize < 1:
        raise ValueError('Batch size must be positive')
    output = Path(output)
    if output.resolve() == resultsRoot.resolve() or resultsRoot.resolve() in output.resolve().parents:
        raise ValueError('Use a separate output directory to preserve the baseline')
    conditions = ['original'] + [c for c in dict.fromkeys(conditions) if c != 'original']
    for category in selectedCategories(categories):
        runCategory(category, conditions, output, batchSize)
    print(f'Results saved in {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()

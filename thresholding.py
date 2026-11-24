import numpy as np
from sklearn.metrics import precision_recall_curve


def exactF1Threshold(masks, maps) -> float:
    labels = np.asarray(masks).ravel()
    scores = np.asarray(maps).ravel()
    if not labels.size or labels.shape != scores.shape:
        raise ValueError('Masks and maps must be nonempty and have matching shapes')
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected finite scores and binary ground-truth masks')
    if not np.any(labels):
        raise ValueError('F1 calibration requires anomalous validation pixels')
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def pixelCounts(masks, maps, threshold):
    """Count true positives, false positives and misses per image."""
    actual = np.asarray(masks).astype(bool)
    predicted = np.asarray(maps) >= threshold
    axes = tuple(range(1, actual.ndim))
    return np.stack([
        np.sum(predicted & actual, axis=axes),
        np.sum(predicted & ~actual, axis=axes),
        np.sum(~predicted & actual, axis=axes)
    ], axis=-1)


def metricsFromCounts(counts):
    tp, fp, fn = np.asarray(counts).sum(axis=0).tolist()
    return {
        'pixel_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        'iou': tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        'precision': tp / (tp + fp) if tp + fp else 0.0,
        'recall': tp / (tp + fn) if tp + fn else 0.0,
        'tp': tp, 'fp': fp, 'fn': fn
    }

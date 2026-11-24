"""Plots and summaries from saved calibration results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import conditionOptions


def pairedF1Interval(imageCounts, condition, repeats=2000):
    """Paired-image bootstrap for the current model and split."""
    subset = imageCounts[imageCounts.condition.eq(condition)]
    baseline = subset[subset.method.eq('legacy_clean')].set_index('relative').sort_index()
    calibrated = subset[subset.method.eq('condition_calibrated')].set_index('relative').sort_index()
    if not baseline.index.equals(calibrated.index) or not baseline.index.is_unique:
        raise ValueError('Bootstrap requires one paired result per test image')
    counts1 = baseline[['tp', 'fp', 'fn']].to_numpy()
    counts2 = calibrated[['tp', 'fp', 'fn']].to_numpy()
    indices = np.random.default_rng(42).integers(0, len(baseline), (repeats, len(baseline)))

    def f1(counts):
        tp, fp, fn = counts[indices].sum(axis=1).T
        denominator = 2 * tp + fp + fn
        return np.divide(2 * tp, denominator, out=np.ones(repeats), where=denominator > 0)

    return np.quantile(f1(counts2) - f1(counts1), [.025, .975])


def main(categories=None, root=Path('results/calibration')):
    root = Path(root)
    # A config file marks a finished run.
    files = [p.parent / 'metrics.csv' for p in sorted(root.glob('*/config.json'))]
    if categories is not None:
        files = [p for p in files if p.parent.name in categories]
        missing = set(categories) - {p.parent.name for p in files}
        if missing:
            raise FileNotFoundError(f'Missing calibration results for: {sorted(missing)}')
    if not files:
        raise RuntimeError('No completed calibration runs found')
    frames = []
    for path in files:
        config = json.loads((path.parent / 'config.json').read_text(encoding='utf-8'))
        frame = pd.read_csv(path)
        methods = ['legacy_clean', 'exact_clean', 'condition_calibrated', 'condition_raw']
        expected = {(config['category'], condition, method)
                    for condition in config['conditions'] for method in methods}
        observed = set(frame[['category', 'condition', 'method']].itertuples(index=False, name=None))
        if len(frame) != len(expected) or observed != expected:
            raise RuntimeError(f'Incomplete or duplicate calibration results: {path}')
        frames.append(frame)
    output = root if categories is None else root / 'reports' / '_'.join(categories)
    output.mkdir(parents=True, exist_ok=True)
    scores = pd.concat(frames, ignore_index=True)
    scores.to_csv(output / 'metrics_all.csv', index=False)
    f1 = scores.pivot(index=['category', 'condition'], columns='method', values='pixel_f1')
    iou = scores.pivot(index=['category', 'condition'], columns='method', values='iou')
    comparison = pd.DataFrame({
        'baseline_f1': f1['legacy_clean'], 'calibrated_f1': f1['condition_calibrated'],
        'f1_delta': f1['condition_calibrated'] - f1['legacy_clean'],
        'baseline_iou': iou['legacy_clean'], 'calibrated_iou': iou['condition_calibrated'],
        'iou_delta': iou['condition_calibrated'] - iou['legacy_clean'],
        'exact_clean_f1': f1['exact_clean'], 'condition_raw_f1': f1['condition_raw']
    }).reset_index()
    comparison['f1_delta_bootstrap_low'] = np.nan
    comparison['f1_delta_bootstrap_high'] = np.nan
    for category in comparison.category.unique():
        imageCounts = pd.read_csv(root / category / 'per_image_counts.csv')
        for i in comparison.index[comparison.category.eq(category)]:
            low, high = pairedF1Interval(imageCounts, comparison.loc[i, 'condition'])
            comparison.loc[i, 'f1_delta_bootstrap_low'] = low
            comparison.loc[i, 'f1_delta_bootstrap_high'] = high
    comparison.to_csv(output / 'comparison.csv', index=False)
    scores.groupby(['category', 'method'])[['pixel_f1', 'iou']].mean().to_csv(output / 'ablation_means.csv')

    categories = list(comparison['category'].unique())
    fig, axes = plt.subplots(len(categories), 1, figsize=(12, 3.4 * len(categories)), squeeze=False)
    for ax, category in zip(axes.flat, categories):
        subset = comparison[comparison.category.eq(category)].set_index('condition')
        subset = subset.reindex([c for c in conditionOptions if c in subset.index])
        x = np.arange(len(subset))
        bars1 = ax.bar(x - .19, subset.baseline_f1, .38, label='Clean-only threshold', color='#64748b')
        bars2 = ax.bar(x + .19, subset.calibrated_f1, .38, label='Condition validation calibration', color='#087f8c')
        ax.bar_label(bars1, fmt='%.3f', fontsize=8, padding=3)
        ax.bar_label(bars2, fmt='%.3f', fontsize=8, padding=3)
        ax.set_xticks(x, [c.replace('_', ' ') for c in subset.index], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Test pixel F1')
        ax.set_title(category.replace('_', ' ').title(), loc='left', fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.18)
    axes.flat[0].legend(loc='upper right', frameon=False)
    fig.suptitle('Can validation calibration improve defect masks?', fontsize=16, fontweight='bold')
    fig.text(.5, .01, 'Same model, images and split. Thresholds chosen on validation masks only. '
             'Exploratory follow-up; known image condition required.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    fig.savefig(output / 'pixel_f1_comparison.png', dpi=180)
    plt.close(fig)
    print(comparison.round(4).to_string(index=False))
    print('\nAverages across the tested categories and conditions:')
    print(comparison[['baseline_f1', 'calibrated_f1', 'f1_delta', 'baseline_iou', 'calibrated_iou']].mean().round(4).to_string())


if __name__ == '__main__':
    main()

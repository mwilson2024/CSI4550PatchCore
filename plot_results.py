from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from config import resultsRoot


def main(categories=None):

    root = resultsRoot

    scores = pd.read_csv(root / 'scores_all.csv')
    morph = pd.read_csv(root / 'morphology_all.csv')
    if categories is not None:
        scores = scores[scores.category.isin(categories)]
        morph = morph[morph.category.isin(categories)]
        missing = set(categories) - (set(scores.category) & set(morph.category))
        if missing or scores.empty or morph.empty:
            raise ValueError(f'Missing evaluation results for: {sorted(missing)}')
        root = root / 'reports' / '_'.join(categories)
        root.mkdir(parents=True, exist_ok=True)
    
    scores['image_auroc'] = pd.to_numeric(scores['image_auroc'], errors='coerce')
    scores['pixel_auroc'] = pd.to_numeric(scores['pixel_auroc'], errors='coerce')
    
    morph['pixel_f1'] = pd.to_numeric(morph['pixel_f1'], errors='coerce')
    
    summary = scores.groupby('condition', as_index=False)[['image_auroc', 'pixel_auroc']].mean()
    summary.to_csv(root / 'mean_auroc_by_condition.csv', index=False)
   
    ax = summary.plot.bar(x='condition', y=['image_auroc','pixel_auroc'], figsize=(11,5))
    ax.set_ylim(0, 1)
    ax.set_ylabel('Mean AUROC across evaluated categories')
    ax.set_xlabel('Image condition')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(root / 'mean_auroc.png', dpi=170)
    plt.close()
    
    pivot = scores.pivot(index='category', columns='condition', values='image_auroc')
    pivot.to_csv(root / 'image_auroc_by_category.csv')
    ax = pivot.plot.bar(figsize=(max(10, len(pivot)*0.9), 6))
    ax.set_ylim(0, 1)
    ax.set_ylabel('Image AUROC')
    plt.xticks(rotation=40, ha='right')
    plt.tight_layout()
    plt.savefig(root / 'image_auroc_by_category.png', dpi=170)
    plt.close()
    
    selected = morph[(morph['condition'].isin(['original', 'dark', 'dark_clahe']))]
    means = selected.groupby(['operation', 'condition'])['pixel_f1'].mean().unstack()
    ax = means.plot.bar(figsize=(9,5))
    ax.set_ylim(0, 1)
    ax.set_ylabel('Mean pixel F1 across evaluated categories')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(root / 'morphology_comparison.png', dpi=170)
    plt.close()
    
    print('Charts and tables saved in', root.resolve())


if __name__ == '__main__':
    main()
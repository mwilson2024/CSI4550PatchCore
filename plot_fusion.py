
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

fusionResultsRoot = Path("results/experiments/fusion")

def main(categories=None):
    df = pd.read_csv(fusionResultsRoot / "fusion_scores_all.csv")
    output = fusionResultsRoot
    if categories is not None:
        df = df[df.category.isin(categories)]
        missing = set(categories) - set(df.category)
        if missing or df.empty:
            raise ValueError(f'Missing fusion results for: {sorted(missing)}')
        output = fusionResultsRoot / 'reports' / '_'.join(categories)
        output.mkdir(parents=True, exist_ok=True)

    # Compare methods under darkness
    dark = df[df["condition"] == "dark"]

    summary = (
        dark.groupby("method")[
            ["pixel_f1", "pixel_auroc", "erosion_f1"]
        ]
        .mean()
    )

    print(summary.round(4))

    ax = summary[
        ["pixel_f1", "erosion_f1"]
    ].plot.bar(figsize=(10, 5))

    ax.set_title("Fusion Methods Under Dark Conditions")
    ax.set_ylabel("Mean Pixel F1")
    ax.set_xlabel("Method")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output / "fusion_f1_comparison.png", dpi=200)
    plt.close()

    # Compare original and dark conditions
    pivot = df.pivot_table(
        index="method",
        columns="condition",
        values="pixel_f1",
        aggfunc="mean"
    )

    ax = pivot.plot.bar(figsize=(10, 5))
    ax.set_title("Fusion Robustness: Clean vs Dark")
    ax.set_ylabel("Mean Pixel F1")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output / "fusion_robustness.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()

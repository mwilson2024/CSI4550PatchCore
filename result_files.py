"""Merge the saved category CSV files."""

import csv
from config import categoryOptions


def combineCategories(root, sourceName, targetName):
    rows = []
    fields = None
    for category in categoryOptions:
        path = root / category / sourceName
        if not path.is_file():
            continue
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if fields is None:
                fields = reader.fieldnames
            elif reader.fieldnames != fields:
                raise ValueError(f'Incompatible result columns: {path}')
            rows.extend(reader)
    if not rows:
        raise FileNotFoundError(f'No saved {sourceName} category results in {root}')
    target = root / targetName
    temporary = target.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)

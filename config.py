from pathlib import Path

dataRoot = Path('datasets/MVTecAD')
conditionsRoot = Path('experiments/datasets')
checkpointRoot = Path('checkpoints')
resultsRoot = Path('results/experiments')

categoryOptions = [ "bottle", "cable", "metal_nut", "screw", "wood", "zipper"]

conditionOptions = ('original', 'dark', 'dark_clahe', 'dark_gamma', 'blurred', 'noisy', 'noisy_median')

morphologyOperations = ('none', 'erosion', 'dilation', 'opening', 'closing')

modelConfig = dict(backbone='resnet18', layers=['layer2','layer3'],
            pre_trained=True, coreset_sampling_ratio=0.1, num_neighbors=9)

defaultCategories = ('bottle',)
defaultAccelerator = 'cpu'
defaultBatchSize = 3
calibrationRoot = Path('results/calibration')


def selectedCategories(categories=None):
    chosen = list(defaultCategories if categories is None else categories)
    if not chosen or any(category not in categoryOptions for category in chosen):
        raise ValueError(f'Choose from: {", ".join(categoryOptions)}')
    return list(dict.fromkeys(chosen))

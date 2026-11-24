from anomalib.engine import Engine
from anomalib.models import Patchcore
from multiprocessing import freeze_support
from pathlib import Path

from modeltools import getCategoryCheckpoint

def predictCategory(category):
    
    ckPoint = getCategoryCheckpoint(category)
    print(ckPoint)
    
    dataPath = str(Path('experiments/datasets/dark') / category / 'test')

    model = Patchcore(backbone="resnet18", layers=["layer2", "layer3"], pre_trained=True, coreset_sampling_ratio=0.1, num_neighbors=9)

    eng = Engine(accelerator="cpu", devices=1, default_root_dir="./results")


    predictions = eng.predict(model=model, ckpt_path=ckPoint, data_path=dataPath)

    if predictions:
   
        first = predictions[0]
        print(f'Prediction Type: {type(first)}')

        if isinstance(first, (list, tuple)) and first:
                first = first[0]

        print(f"Available fields: {getattr(first, '__dict__', {}) .keys()}")
    else:
        raise RuntimeError('No predictions made')

def main(categories=None):
    from config import selectedCategories
    for category in selectedCategories(categories):
        predictCategory(category)


if __name__ == "__main__":
    freeze_support()
    main()   
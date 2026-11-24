from anomalib.data import MVTecAD
from anomalib.engine import Engine 
from anomalib.models import Patchcore
from multiprocessing import freeze_support
def runCategory(category):


    # Load the dataset
    dm = MVTecAD(root='./datasets/MVTecAD', category=category, train_batch_size= 3, eval_batch_size=3, val_split_ratio=0.30, seed=42, val_split_mode="from_test")

    # PatchCore model
    model = Patchcore(backbone='resnet18', layers=['layer2', 'layer3'], pre_trained=True, coreset_sampling_ratio = .1, num_neighbors=9)

    # Build the memory bank
    eng = Engine(max_epochs=1, accelerator = 'cpu', devices=1, default_root_dir=f'./results/baseline_demo/{category}')
    eng.fit(model=model, datamodule=dm)

    # Test normal and defective images
    res = eng.test(model=model, datamodule=dm)

    print(f'Baseline results :\n{res}')

def main(categories=None):
    from config import selectedCategories
    for category in selectedCategories(categories):
        runCategory(category)


if __name__ == '__main__':
    freeze_support()
    main()
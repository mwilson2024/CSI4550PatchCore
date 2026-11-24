from multiprocessing import freeze_support
import gc
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
from config import (dataRoot, checkpointRoot, modelConfig,
                    defaultAccelerator, selectedCategories)


def packageVersion(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return 'unknown'


def trainCategory(category: str, accelerator: str, overwrite: bool) -> None:
    dataDir = dataRoot / category / 'train' / 'good'
    if not dataDir.is_dir():
        raise FileNotFoundError(f'Missing training directory: {dataDir.resolve()}')
    ckpt = checkpointRoot / category / 'patchcore.ckpt'
    if ckpt.exists() and not overwrite:
        print(f'[{category}] already trained, keeping {ckpt}')
        return
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    dm = MVTecAD(
        root=str(dataRoot), category=category,
        train_batch_size=3, eval_batch_size=3,
        num_workers=0,
        val_split_mode='from_test', val_split_ratio=0.30, seed=42
    )
    model = Patchcore(**modelConfig)
    eng = Engine(max_epochs=1, accelerator=accelerator, devices=1,
                 default_root_dir=str(Path('results/train') / category))
    print(f'[{category}] building memory bank on {accelerator}')
    eng.fit(model=model, datamodule=dm)
    eng.trainer.save_checkpoint(str(ckpt))
    if not ckpt.is_file() or ckpt.stat().st_size == 0:
        raise RuntimeError(f'Checkpoint was not saved correctly: {ckpt}')
    metadata = {
        'category': category, 'checkpoint': str(ckpt.resolve()),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'model': modelConfig, 'accelerator': accelerator,
        'validation': {'mode': 'from_test', 'ratio': 0.30, 'seed': 42},
        'python': platform.python_version(),
        'torch': torch.__version__, 'anomalib': packageVersion('anomalib')
    }
    (ckpt.parent / 'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'[{category}] checkpoint saved: {ckpt.resolve()}')
    del model, eng, dm
    gc.collect()


def main(categories=None):
    for category in selectedCategories(categories):
        trainCategory(category, defaultAccelerator, False)


if __name__ == '__main__':
    freeze_support()
    main()

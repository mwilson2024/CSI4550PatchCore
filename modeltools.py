from pathlib import Path
from datetime import datetime
from config import checkpointRoot

def checkPointStats(res: Path = Path('results')):

    #checkpoint
    cp = sorted(res.rglob('*.ckpt'), key=lambda path: path.stat().st_mtime, reverse=True)

    if not cp:
        print(f'No check points found')
    else:
        for x in cp:
            
            stats = x.stat()
            size = stats.st_size / 1024**2
            lastSaved = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

            print(f'{x.resolve()} | {size:.1f} MB | Time: {lastSaved}')

def lastestCheckPoint(res: Path = Path('results')) -> Path:

    cp = max(res.rglob('*.ckpt'), key=lambda path: path.stat().st_mtime, default=None)

    if cp is None:
        raise FileNotFoundError(f'No checkpoints found {res.resolve()}')

    return cp.resolve()

def getCategoryCheckpoint(category: str) -> Path:
    path = checkpointRoot / category / 'patchcore.ckpt'
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path.resolve()}')
    return path.resolve()


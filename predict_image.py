import json
import math
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np
from PIL import Image, ImageOps
from config import categoryOptions, defaultAccelerator, modelConfig, resultsRoot
from modeltools import getCategoryCheckpoint


def loadThreshold(category):
    path = resultsRoot / category / 'threshold.txt'
    if not path.is_file():
        raise FileNotFoundError(f'No validation threshold for {category}. Run baseline evaluation first: {path}')
    threshold = float(path.read_text().strip())
    if not math.isfinite(threshold):
        raise ValueError(f'Invalid threshold in {path}')
    return threshold


def saveOutputs(image, anomaly, score, threshold, output, source, category, checkpoint):
    """Save the prediction at the model output size."""
    anomaly = np.asarray(anomaly, dtype=np.float32)
    if anomaly.ndim != 2 or not np.isfinite(anomaly).all() or not math.isfinite(score):
        raise ValueError('Prediction must contain finite scores and a two-dimensional map')
    height, width = anomaly.shape
    resized = cv2.resize(image, (width, height))
    mask = np.uint8(anomaly >= threshold) * 255
    display = cv2.normalize(anomaly, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(display, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(resized, 0.6, heatmap, 0.4, 0)
    panels = np.hstack([resized, overlay, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
    output.mkdir(parents=True, exist_ok=False)
    for name, array in [('input.png', resized), ('heatmap.png', heatmap),
                        ('overlay.png', overlay), ('defect_mask.png', mask), ('comparison.png', panels)]:
        # Handles Unicode paths on Windows.
        ok, encoded = cv2.imencode('.png', array)
        if not ok:
            raise RuntimeError(f'Unable to save {name}')
        encoded.tofile(output / name)
    np.save(output / 'anomaly_map.npy', anomaly)
    area = int(np.count_nonzero(mask))
    report = {
        'source_image': str(source), 'category': category, 'checkpoint': str(checkpoint),
        'method': 'baseline PatchCore, no enhancement or morphology',
        'image_anomaly_score': float(score), 'pixel_threshold': threshold,
        'threshold_source': str((resultsRoot / category / 'threshold.txt').resolve()),
        'source_width': int(image.shape[1]), 'source_height': int(image.shape[0]),
        'mask_width': width, 'mask_height': height, 'predicted_defect_pixels': area,
        'predicted_image_coverage_percent': 100 * area / mask.size,
        'comparison_panels': ['input', 'heatmap overlay', 'predicted defect mask'],
        'notes': 'White mask pixels have anomaly scores at or above the saved validation threshold. '
                 'Heatmap colors are normalized for display only. Scores are not probabilities. '
                 'No ground truth was supplied, so accuracy, F1 and IoU are not calculated.'
    }
    (output / 'prediction.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def predictImage(imagePath, category, outputRoot=Path('results/user_images')):
    if category not in categoryOptions:
        raise ValueError(f'Unknown model category: {category}')
    source = Path(imagePath).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Image not found: {source}')
    checkpoint = getCategoryCheckpoint(category)
    threshold = loadThreshold(category)
    with Image.open(source) as original:
        rgb = np.array(ImageOps.exif_transpose(original).convert('RGB'))
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Check the files before loading the model.
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    model = Patchcore(**modelConfig)
    engine = Engine(accelerator=defaultAccelerator, devices=1,
                    default_root_dir='results/user_image_engine')
    # Use a temporary PNG after fixing orientation.
    with tempfile.TemporaryDirectory(prefix='patchcore_image_') as temporary:
        prepared = Path(temporary) / 'input.png'
        Image.fromarray(rgb).save(prepared)
        batches = engine.predict(model=model, ckpt_path=str(checkpoint),
                                 data_path=prepared, return_predictions=True)
    if not batches or len(batches) != 1 or len(batches[0].pred_score.reshape(-1)) != 1:
        raise RuntimeError('Expected exactly one image prediction')
    batch = batches[0]
    anomaly = batch.anomaly_map[0].detach().cpu().numpy().squeeze()
    score = float(batch.pred_score.detach().cpu().reshape(-1)[0])
    name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
    output = Path(outputRoot) / category / name
    report = saveOutputs(image, anomaly, score, threshold, output, source, category, checkpoint)
    print(f'\nImage anomaly score: {score:.6f} (not a probability)')
    print(f'Predicted defect pixels: {report["predicted_defect_pixels"]} '
          f'({report["predicted_image_coverage_percent"]:.2f}% of the model image)')
    print(f'Image, heatmap, mask and report saved in: {output.resolve()}')
    return output

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import main
import predict_image


class UserImageTests(unittest.TestCase):
    def testMenuRoutesOneImageWithoutPipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'my image.png'
            path.touch()
            with patch('builtins.input', side_effect=['6', 'zipper', f'"{path}"', '0']), \
                 patch.object(predict_image, 'predictImage') as predict, \
                 patch.object(main, 'runStages') as pipeline, redirect_stdout(io.StringIO()):
                main.main()
                predict.assert_called_once_with(path, 'zipper')
                pipeline.assert_not_called()

    def testCancelDoesNotLoadInference(self):
        with patch('builtins.input', side_effect=['1', '']), \
             patch.object(main.importlib, 'import_module') as load, redirect_stdout(io.StringIO()):
            main.userImageInference()
            load.assert_not_called()

    def testMaskUsesSavedThresholdIncludingEquality(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'output'
            report = predict_image.saveOutputs(np.zeros((8, 10, 3), np.uint8),
                np.array([[0.1, 0.5], [0.7, 0.2]]), 0.8, 0.5, output,
                Path('user.png'), 'bottle', Path('model.ckpt'))
            mask = cv2.imread(str(output / 'defect_mask.png'), cv2.IMREAD_GRAYSCALE)
            np.testing.assert_array_equal(mask, [[0, 255], [255, 0]])
            self.assertEqual(report['predicted_defect_pixels'], 2)
            self.assertEqual(report['predicted_image_coverage_percent'], 50)
            self.assertEqual(json.loads((output / 'prediction.json').read_text())['mask_width'], 2)

    def testMissingThresholdHasClearError(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(predict_image, 'resultsRoot', Path(folder)):
            with self.assertRaisesRegex(FileNotFoundError, 'baseline evaluation'):
                predict_image.loadThreshold('bottle')


if __name__ == '__main__':
    unittest.main()

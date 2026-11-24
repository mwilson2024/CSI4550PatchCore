import unittest

import numpy as np

from thresholding import exactF1Threshold, pixelCounts, metricsFromCounts


class ThresholdTests(unittest.TestCase):
    def testMatchesBruteForceWithTiedScores(self):
        rng = np.random.default_rng(42)
        for _ in range(30):
            masks = rng.integers(0, 2, (4, 3, 5))
            maps = rng.integers(0, 8, masks.shape) / 7
            expected = max(np.unique(maps), key=lambda t: (
                metricsFromCounts(pixelCounts(masks, maps, t))['pixel_f1'], t))
            self.assertEqual(exactF1Threshold(masks, maps), expected)

    def testSmallDefectAboveOldQuantileRange(self):
        masks = np.zeros((1, 100, 100), dtype=np.uint8)
        maps = np.arange(10000, dtype=float).reshape(masks.shape)
        masks[0, -1, -1] = 1
        self.assertEqual(exactF1Threshold(masks, maps), 9999.0)
        self.assertEqual(metricsFromCounts(pixelCounts(masks, maps, 9999))['pixel_f1'], 1)

    def testInvalidValidationData(self):
        for masks, maps in [([], []), ([0, 0], [1, 2]), ([1], [np.nan]), ([1, 0], [1]), ([2], [1])]:
            with self.subTest(masks=masks, maps=maps), self.assertRaises(ValueError):
                exactF1Threshold(masks, maps)

    def testPooledCountsAndThresholdBoundary(self):
        masks = np.array([[[1, 0]], [[1, 0]]])
        maps = np.array([[[0.5, 0.5]], [[0.1, 0.1]]])
        counts = pixelCounts(masks, maps, 0.5)
        np.testing.assert_array_equal(counts, [[1, 1, 0], [0, 0, 1]])
        metrics = metricsFromCounts(counts)
        self.assertEqual(metrics['pixel_f1'], 0.5)
        self.assertEqual(metrics['iou'], 1 / 3)


if __name__ == '__main__':
    unittest.main()

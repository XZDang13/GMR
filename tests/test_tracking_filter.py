import unittest

import numpy as np

from general_motion_retargeting.tracking_filter import evaluate_tracking_errors


class TrackingFilterTest(unittest.TestCase):
    def test_accepts_when_percentile_error_is_below_threshold(self):
        result = evaluate_tracking_errors(
            [[0.03, 0.05], [0.04, 0.06], [0.02, 0.05]],
            threshold=0.2,
            percentile=95,
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "ok")

    def test_rejects_when_percentile_error_exceeds_threshold(self):
        result = evaluate_tracking_errors(
            [[0.05, 0.10], [0.12, 0.25], [0.18, 0.30]],
            threshold=0.2,
            percentile=95,
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "tracking error above threshold")

    def test_rejects_non_finite_tracking_errors(self):
        result = evaluate_tracking_errors(
            [[0.05, 0.10], [np.nan, 0.12]],
            threshold=0.2,
            percentile=95,
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "non-finite tracking error")

    def test_uses_per_frame_max_error_before_percentile(self):
        result = evaluate_tracking_errors(
            [[0.01, 0.50], [0.20, 0.10]],
            threshold=0.49,
            percentile=100,
        )

        self.assertFalse(result.accepted)
        self.assertTrue(np.isclose(result.max_error, 0.50))
        self.assertTrue(np.isclose(result.percentile_error, 0.50))


if __name__ == "__main__":
    unittest.main()

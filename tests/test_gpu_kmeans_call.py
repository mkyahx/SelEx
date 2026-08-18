import os
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_SCRIPT = os.path.join(
    REPO_ROOT,
    "methods",
    "contrastive_training",
    "contrastive_training.py",
)


class GpuKMeansCallTest(unittest.TestCase):

    def test_gpu_kmeans_filters_unsupported_kwargs_and_falls_back_when_unbounded(self):
        with open(TRAINING_SCRIPT, "r", encoding="utf-8") as handle:
            contents = handle.read()

        self.assertIn("parser.add_argument('--max_kmeans_iter'", contents)
        self.assertIn("if not np.isfinite(all_feats).all():", contents)
        self.assertIn("inspect.signature(kmeans)", contents)
        self.assertIn('"iter_limit" not in kmeans_params', contents)
        self.assertIn("KMeans(n_clusters=", contents)
        self.assertIn("random_state=args.seed", contents)


if __name__ == "__main__":
    unittest.main()

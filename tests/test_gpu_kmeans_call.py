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

    def test_gpu_kmeans_has_iteration_limit_seed_and_finite_feature_guard(self):
        with open(TRAINING_SCRIPT, "r", encoding="utf-8") as handle:
            contents = handle.read()

        self.assertIn("parser.add_argument('--max_kmeans_iter'", contents)
        self.assertIn("if not np.isfinite(all_feats).all():", contents)
        self.assertIn("iter_limit=args.max_kmeans_iter", contents)
        self.assertIn("seed=args.seed", contents)


if __name__ == "__main__":
    unittest.main()

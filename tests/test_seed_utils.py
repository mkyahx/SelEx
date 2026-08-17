import os
import importlib
import random
import sys
import types
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeGenerator:

    def __init__(self, device=None):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


def make_fake_torch():
    cudnn = types.SimpleNamespace(benchmark=True, deterministic=False, allow_tf32=True)
    matmul = types.SimpleNamespace(allow_tf32=True)
    cuda_backend = types.SimpleNamespace(matmul=matmul)
    backends = types.SimpleNamespace(cudnn=cudnn, cuda=cuda_backend)
    cuda = types.SimpleNamespace(manual_seed=Mock(), manual_seed_all=Mock())
    return types.SimpleNamespace(
        manual_seed=Mock(),
        initial_seed=Mock(return_value=0),
        use_deterministic_algorithms=Mock(),
        Generator=FakeGenerator,
        cuda=cuda,
        backends=backends,
    )


class SeedUtilsTest(unittest.TestCase):

    def setUp(self):
        self.fake_torch = make_fake_torch()
        self.previous_torch = sys.modules.get("torch")
        sys.modules["torch"] = self.fake_torch
        sys.modules.pop("project_utils.seed_utils", None)
        project_utils = sys.modules.get("project_utils")
        if project_utils is not None and hasattr(project_utils, "seed_utils"):
            delattr(project_utils, "seed_utils")
        self.seed_utils = importlib.import_module("project_utils.seed_utils")

    def tearDown(self):
        sys.modules.pop("project_utils.seed_utils", None)
        if self.previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self.previous_torch

    def test_seed_everything_enables_strict_cuda_determinism(self):
        self.seed_utils.seed_everything(123, deterministic=True)

        self.assertEqual(os.environ["PYTHONHASHSEED"], "123")
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.fake_torch.manual_seed.assert_called_once_with(123)
        self.fake_torch.cuda.manual_seed.assert_called_once_with(123)
        self.fake_torch.cuda.manual_seed_all.assert_called_once_with(123)
        self.fake_torch.use_deterministic_algorithms.assert_called_once_with(True)
        self.assertFalse(self.fake_torch.backends.cudnn.benchmark)
        self.assertTrue(self.fake_torch.backends.cudnn.deterministic)
        self.assertFalse(self.fake_torch.backends.cudnn.allow_tf32)
        self.assertFalse(self.fake_torch.backends.cuda.matmul.allow_tf32)

    def test_seed_worker_derives_numpy_and_python_seed_from_torch_initial_seed(self):
        self.fake_torch.initial_seed.return_value = 2**32 + 77
        self.seed_utils.seed_worker(worker_id=3)

        self.assertEqual(np.random.randint(0, 1000), 727)
        self.assertEqual(random.randint(0, 1000), 818)

    def test_make_torch_generator_is_reproducible(self):
        generator = self.seed_utils.make_torch_generator(9)

        self.assertEqual(generator.seed, 9)
        self.assertIsNone(generator.device)


if __name__ == "__main__":
    unittest.main()

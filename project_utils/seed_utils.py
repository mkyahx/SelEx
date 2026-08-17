import os
import random

import numpy as np
import torch


CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def seed_everything(seed=1029, deterministic=True):
    """Seed Python, NumPy, PyTorch, CUDA, and deterministic GPU backends."""

    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic)
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(bool(deterministic))

    return seed


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed, device=None):
    if device is None:
        generator = torch.Generator()
    else:
        generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def make_dataloader_seed_kwargs(seed):
    return {
        "worker_init_fn": seed_worker,
        "generator": make_torch_generator(seed),
    }


def seed_subprocess_env(seed, deterministic=True):
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(int(seed))
    if deterministic:
        env["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    return env

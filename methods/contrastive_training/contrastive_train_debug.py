"""Debug entry point for SelEx contrastive training.

Run this module in place of ``contrastive_training`` when training produces a
NaN ``center_shift`` or K-Means does not terminate.  It keeps the original
training implementation intact and instruments the tensors flowing through it:

* dataloader images, model features, projection-head outputs and InfoNCE logits;
* contrastive/supervised losses, gradients and parameters around ``SGD.step``;
* every K-Means input, distance matrix, centre update and centre shift.

The replacement K-Means routine intentionally caps its iterations and raises at
the first non-finite tensor, so a corrupted evaluation cannot spin forever.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from methods.contrastive_training import contrastive_training as training


@dataclass
class DebugState:
    """Mutable context included in every diagnostic line."""

    log_interval: int
    kmeans_iter_limit: int
    phase: str = "setup"
    epoch: int = -1
    batch_idx: int = -1
    parameter_names: Optional[dict] = None

    def prefix(self) -> str:
        return (
            f"[debug phase={self.phase} epoch={self.epoch} "
            f"batch={self.batch_idx}]"
        )

    def should_log(self, step: Optional[int] = None) -> bool:
        step = self.batch_idx if step is None else step
        return step < 0 or step % self.log_interval == 0


def check_tensor(
    name: str,
    value: torch.Tensor,
    state: DebugState,
    *,
    log: bool = True,
) -> None:
    """Fail at the first NaN/Inf while retaining a compact numeric summary."""

    if not isinstance(value, torch.Tensor):
        return

    tensor = value.detach()
    finite = torch.isfinite(tensor)
    if not bool(finite.all()):
        bad_indices = (~finite).nonzero(as_tuple=False)[:5].cpu().tolist()
        num_bad = int((~finite).sum().item())
        raise FloatingPointError(
            f"{state.prefix()} {name}: found {num_bad} NaN/Inf values; "
            f"first indices={bad_indices}, shape={tuple(tensor.shape)}"
        )

    if log and state.should_log():
        stats = tensor.float()
        print(
            f"{state.prefix()} {name}: shape={tuple(tensor.shape)} "
            f"min={stats.min().item():.3e} max={stats.max().item():.3e} "
            f"mean={stats.mean().item():.3e} norm={stats.norm().item():.3e}",
            flush=True,
        )


def _check_batch_images(batch: Any, state: DebugState, loader_name: str) -> None:
    """Check image tensors before the original training code moves them to CUDA."""

    if not isinstance(batch, (tuple, list)) or not batch:
        return

    images = batch[0]
    if isinstance(images, (tuple, list)):
        for view_idx, image_view in enumerate(images):
            check_tensor(f"{loader_name}.images.view_{view_idx}", image_view, state)
    elif isinstance(images, torch.Tensor):
        check_tensor(f"{loader_name}.images", images, state)


class DebugLoader:
    """Delegating dataloader that records the current phase and batch index."""

    def __init__(
        self,
        loader: Iterable,
        state: DebugState,
        name: str,
        *,
        advances_epoch: bool = False,
    ) -> None:
        self.loader = loader
        self.state = state
        self.name = name
        self.advances_epoch = advances_epoch

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]

    def __iter__(self):
        if self.advances_epoch:
            self.state.epoch += 1
        self.state.phase = self.name
        for batch_idx, batch in enumerate(self.loader):
            self.state.batch_idx = batch_idx
            _check_batch_images(batch, self.state, self.name)
            yield batch


def _module_output_hook(name: str, state: DebugState) -> Callable:
    def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        if isinstance(output, torch.Tensor):
            check_tensor(name, output, state)
        elif isinstance(output, (tuple, list)):
            for output_idx, tensor in enumerate(output):
                if isinstance(tensor, torch.Tensor):
                    check_tensor(f"{name}[{output_idx}]", tensor, state)

    return hook


def _check_named_parameters(
    named_parameters: Sequence[Tuple[str, torch.nn.Parameter]],
    state: DebugState,
    location: str,
) -> None:
    for name, parameter in named_parameters:
        check_tensor(f"{location}.parameter.{name}", parameter, state, log=False)


class DebugSGD(SGD):
    """Original SGD plus finite-gradient/finite-parameter checks at every step."""

    debug_state: DebugState

    def step(self, closure: Optional[Callable] = None):  # type: ignore[override]
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter_name = self.debug_state.parameter_names.get(
                    id(parameter), "unnamed"
                )
                check_tensor(
                    f"gradient.{parameter_name}",
                    parameter.grad,
                    self.debug_state,
                    log=False,
                )

        grads = [
            parameter.grad.detach().float().norm()
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if grads and self.debug_state.should_log():
            grad_norm = torch.linalg.vector_norm(torch.stack(grads))
            check_tensor("gradient.global_norm", grad_norm, self.debug_state)

        result = super().step(closure)

        for group in self.param_groups:
            for parameter in group["params"]:
                parameter_name = self.debug_state.parameter_names.get(
                    id(parameter), "unnamed"
                )
                check_tensor(
                    f"after_optimizer_step.parameter.{parameter_name}",
                    parameter,
                    self.debug_state,
                    log=False,
                )
        return result


def make_debug_kmeans(state: DebugState) -> Callable:
    """Build a finite-checked Euclidean K-Means compatible with kmeans_pytorch."""

    from kmeans_pytorch import pairwise_distance

    def debug_kmeans(
        X: torch.Tensor,
        num_clusters: int,
        distance: str = "euclidean",
        cluster_centers: Any = None,
        tol: float = 1e-4,
        tqdm_flag: bool = True,
        iter_limit: int = 0,
        device: Optional[torch.device] = None,
        **_unused: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if distance != "euclidean":
            raise NotImplementedError("The debug K-Means supports SelEx's euclidean mode only.")

        state.phase = "kmeans"
        state.batch_idx = -1
        device = device or torch.device("cpu")
        X = X.float().to(device)
        check_tensor("kmeans.input_features", X, state)

        if X.ndim != 2:
            raise ValueError(f"K-Means expects [samples, features], got {tuple(X.shape)}")
        if not 0 < num_clusters <= len(X):
            raise ValueError(
                f"num_clusters={num_clusters} must be in [1, {len(X)}] for K-Means"
            )

        if cluster_centers is None or isinstance(cluster_centers, list):
            indices = torch.randperm(len(X), device=device)[:num_clusters]
            centers = X[indices].clone()
            print(
                f"{state.prefix()} kmeans: initialized {num_clusters} centres from samples",
                flush=True,
            )
        else:
            supplied_centers = cluster_centers.float().to(device)
            check_tensor("kmeans.supplied_centres", supplied_centers, state)
            initial_distance = pairwise_distance(X, supplied_centers, device=device, tqdm_flag=False)
            check_tensor("kmeans.initial_distance", initial_distance, state)
            centers = X[torch.argmin(initial_distance, dim=0)].clone()

        check_tensor("kmeans.initial_centres", centers, state)
        max_iterations = state.kmeans_iter_limit

        for iteration in range(1, max_iterations + 1):
            distances = pairwise_distance(X, centers, device=device, tqdm_flag=False)
            check_tensor("kmeans.distances", distances, state, log=False)
            assignments = torch.argmin(distances, dim=1)
            previous_centers = centers.clone()

            for cluster_idx in range(num_clusters):
                selected_indices = torch.nonzero(
                    assignments == cluster_idx, as_tuple=False
                ).flatten()
                if selected_indices.numel() == 0:
                    replacement_idx = torch.randint(len(X), (1,), device=device)
                    centers[cluster_idx] = X[replacement_idx]
                    print(
                        f"{state.prefix()} kmeans iteration={iteration}: "
                        f"empty cluster={cluster_idx}; reinitialized it",
                        flush=True,
                    )
                else:
                    centers[cluster_idx] = X.index_select(0, selected_indices).mean(dim=0)

            check_tensor("kmeans.updated_centres", centers, state, log=False)
            per_center_shift_sq = (centers - previous_centers).square().sum(dim=1)
            check_tensor("kmeans.per_center_shift_sq", per_center_shift_sq, state, log=False)
            center_shift_sq = per_center_shift_sq.sqrt().sum().square()
            check_tensor("kmeans.center_shift_sq", center_shift_sq, state, log=False)

            if iteration == 1 or iteration % state.log_interval == 0:
                print(
                    f"{state.prefix()} kmeans iteration={iteration}: "
                    f"center_shift_sq={center_shift_sq.item():.6e}, tol={tol:.6e}",
                    flush=True,
                )

            if center_shift_sq.item() < tol:
                print(
                    f"{state.prefix()} kmeans converged in {iteration} iterations",
                    flush=True,
                )
                return assignments.cpu(), centers.cpu()

        raise RuntimeError(
            f"{state.prefix()} K-Means did not converge within {max_iterations} iterations. "
            "The finite checks passed; inspect the center_shift output above."
        )

    return debug_kmeans


def install_debug_instrumentation(
    model: torch.nn.Module,
    projection_head: torch.nn.Module,
    state: DebugState,
) -> List[Any]:
    """Patch the imported training module and return removable forward hooks."""

    original_info_nce_logits = training.info_nce_logits
    original_label_smoothing_loss = training.LabelSmoothingLoss
    original_supcon_loss = training.SupConLoss
    original_average_meter = training.AverageMeter

    def debug_info_nce_logits(features, confusion_factor, args, is_code=False):
        check_tensor("infonce.input_features", features, state)
        check_tensor("infonce.confusion_factor", confusion_factor, state)
        logits, labels, similarity = original_info_nce_logits(
            features, confusion_factor, args, is_code=is_code
        )
        check_tensor("infonce.logits", logits, state)
        check_tensor("infonce.similarity", similarity, state)
        return logits, labels, similarity

    class DebugLabelSmoothingLoss(original_label_smoothing_loss):
        def forward(self, input, target, similarity, smoothing=0.5):
            check_tensor("label_smoothing.input", input, state)
            check_tensor("label_smoothing.similarity", similarity, state)
            loss = super().forward(input, target, similarity, smoothing)
            check_tensor("contrastive_loss", loss, state)
            return loss

    class DebugSupConLoss(original_supcon_loss):
        def forward(self, features, labels=None, mask=None, is_code=False):
            check_tensor("supcon.features", features, state)
            loss = super().forward(features, labels=labels, mask=mask, is_code=is_code)
            check_tensor("supcon_loss", loss, state)
            return loss

    class DebugAverageMeter(original_average_meter):
        """Catch an invalid total loss when the upstream loop records it."""

        def update(self, value, n=1):
            if isinstance(value, torch.Tensor):
                check_tensor("average_meter.value", value, state)
            elif not np.isfinite(value):
                raise FloatingPointError(
                    f"{state.prefix()} average_meter.value: got non-finite scalar {value}"
                )
            return super().update(value, n)

    state.parameter_names = {
        id(parameter): f"model.{name}" for name, parameter in model.named_parameters()
    }
    state.parameter_names.update(
        {
            id(parameter): f"projection_head.{name}"
            for name, parameter in projection_head.named_parameters()
        }
    )
    DebugSGD.debug_state = state

    training.info_nce_logits = debug_info_nce_logits
    training.LabelSmoothingLoss = DebugLabelSmoothingLoss
    training.SupConLoss = DebugSupConLoss
    training.AverageMeter = DebugAverageMeter
    training.SGD = DebugSGD
    training.kmeans = make_debug_kmeans(state)

    return [
        model.register_forward_hook(_module_output_hook("base_model.features", state)),
        projection_head.register_forward_hook(
            _module_output_hook("projection_head.features", state)
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SelEx contrastive training with NaN/K-Means diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--eval_funcs", nargs="+", default=["v1", "v2"])
    parser.add_argument("--warmup_model_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="vit_dino")
    parser.add_argument("--dataset_name", type=str, default="cub")
    parser.add_argument("--prop_train_labels", type=float, default=0.5)
    parser.add_argument("--use_ssb_splits", type=training.str2bool, default=True)
    parser.add_argument("--grad_from_block", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--save_best_thresh", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--exp_root", type=str, default=training.exp_root)
    parser.add_argument("--transform", type=str, default="imagenet")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--base_model", type=str, default="vit_dino")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sup_con_weight", type=float, default=0.35)
    parser.add_argument("--n_views", default=2, type=int)
    parser.add_argument("--contrast_unlabel_only", type=training.str2bool, default=False)
    parser.add_argument("--strategy", type=str, default="zero_one")
    parser.add_argument("--cluster_momentum", type=float, default=1)
    parser.add_argument("--unsupervised_smoothing", type=float, default=1)
    parser.add_argument("--distance", type=str, default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--train_report_interval", default=200, type=int)
    parser.add_argument("--prototype_extraction_interval", default=1, type=int)
    parser.add_argument("--gpu_clustering", type=training.str2bool, default=True)
    parser.add_argument("--unbalanced", type=training.str2bool, default=False)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--report", type=training.str2bool, default=False)
    parser.add_argument(
        "--debug_log_interval",
        type=int,
        default=1,
        help="Print tensor/K-Means statistics every N batches or K-Means iterations.",
    )
    parser.add_argument(
        "--debug_kmeans_iter_limit",
        type=int,
        default=100,
        help="Hard cap for the debug K-Means routine; prevents infinite iteration.",
    )
    parser.add_argument(
        "--debug_detect_anomaly",
        type=training.str2bool,
        default=False,
        help="Enable PyTorch autograd anomaly detection (slow, but reports bad backward ops).",
    )
    return parser


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.base_model != "vit_dino":
        raise NotImplementedError("The upstream training entry point supports vit_dino only.")

    args.interpolation = 3
    args.crop_pct = 0.875
    model = training.vits.__dict__["vit_base"]()
    torch.cuda.empty_cache()

    if training.dino_v1:
        state_dict = torch.load(training.dino_pretrain_path, map_location="cpu")["teacher"]
        for key in list(state_dict.keys()):
            state_dict[key.replace("backbone.", "")] = state_dict.pop(key)
    else:
        state_dict = torch.load(training.dino_pretrain_path2, map_location="cpu")
    model.load_state_dict(state_dict)

    if args.warmup_model_dir is not None:
        checkpoint_path = os.path.join(args.warmup_model_dir, "model_best.pt")
        print(f"Loading weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)

    model.to(device)
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = 65536

    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if "block" in name and int(name.split(".")[1]) >= args.grad_from_block:
            parameter.requires_grad = True
    return model


def main() -> None:
    args = build_parser().parse_args()
    if args.debug_log_interval < 1 or args.debug_kmeans_iter_limit < 1:
        raise ValueError("--debug_log_interval and --debug_kmeans_iter_limit must both be positive")

    device = torch.device("cuda:0")
    training.device = device
    args = training.get_class_splits(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    training.init_experiment(args, runner_name=["metric_learn_gcd_debug"])

    state = DebugState(
        log_interval=args.debug_log_interval,
        kmeans_iter_limit=args.debug_kmeans_iter_limit,
    )
    model = load_model(args, device)
    projection_head = training.vits.__dict__["DINOHead"](
        in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers
    ).to(device)
    if args.warmup_model_dir is not None:
        checkpoint_path = os.path.join(args.warmup_model_dir, "model_proj_head_best.pt")
        print(f"Loading projection head from {checkpoint_path}")
        projection_head.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)

    _check_named_parameters(list(model.named_parameters()), state, "startup")
    _check_named_parameters(list(projection_head.named_parameters()), state, "startup")

    train_transform, test_transform = training.get_transform(
        args.transform, image_size=args.image_size, args=args
    )
    train_transform = training.ContrastiveLearningViewGenerator(
        base_transform=train_transform, n_views=args.n_views
    )
    train_dataset, test_dataset, unlabelled_train_examples_test, _ = training.get_datasets(
        args.dataset_name, train_transform, test_transform, args
    )

    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = torch.DoubleTensor(
        [
            1 if index < label_len else label_len / (unlabelled_len + label_len)
            for index in range(len(train_dataset))
        ]
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(train_dataset)
    )
    merge_train_loader = DataLoader(
        train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
    )
    unlabelled_train_loader = DataLoader(
        unlabelled_train_examples_test,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False
    )

    hooks = install_debug_instrumentation(model, projection_head, state)
    previous_anomaly_state = torch.is_anomaly_enabled()
    torch.autograd.set_detect_anomaly(args.debug_detect_anomaly)
    if not os.path.exists("Plots"):
        os.mkdir("Plots")

    try:
        training.train(
            projection_head,
            model,
            DebugLoader(train_loader, state, "train", advances_epoch=True),
            DebugLoader(test_loader, state, "test"),
            DebugLoader(unlabelled_train_loader, state, "unlabelled_test"),
            DebugLoader(merge_train_loader, state, "prototype_extraction"),
            args,
        )
    finally:
        torch.autograd.set_detect_anomaly(previous_anomaly_state)
        for hook in hooks:
            hook.remove()


if __name__ == "__main__":
    main()

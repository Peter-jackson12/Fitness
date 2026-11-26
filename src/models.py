"""ResNet18 네 가지 학습 전략(D-013)과 옵티마이저 파라미터 그룹(D-012).

| 전략 | 초기 가중치 | 학습하는 부분 |
|---|---|---|
| scratch | 없음(`weights=None`) | 전체 |
| fe      | ImageNet | `fc`만(백본 고정, 백본 BN도 `eval()`) |
| partial | ImageNet | `layer4` + `fc` |
| full    | ImageNet | 전체 |
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

import common  # noqa: F401  (TORCH_HOME을 fitness/.cache/torch로 잡는다)

STRATEGIES = ("scratch", "fe", "partial", "full")
NUM_CLASSES = 12


def build_model(strategy: str, num_classes: int = NUM_CLASSES, seed: int = 42, pretrained: bool = True) -> nn.Module:
    """pretrained=False는 저장한 가중치를 곧바로 덮어쓸 때(평가) 쓴다. ImageNet 가중치를 읽지 않는다."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy는 {STRATEGIES} 중 하나: {strategy}")
    torch.manual_seed(seed)
    weights = None if strategy == "scratch" or not pretrained else ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights, progress=False)
    # 새 분류기. 직전에 seed를 다시 고정해 네 전략의 fc 초기값을 똑같이 만든다(공정 비교).
    torch.manual_seed(seed)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if strategy in ("fe", "partial"):
        trainable_prefixes = ("fc.",) if strategy == "fe" else ("layer4.", "fc.")
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(trainable_prefixes)
    return model


def set_train_mode(model: nn.Module) -> nn.Module:
    """`model.train()` 뒤, 가중치가 고정된 BatchNorm은 `eval()`로 되돌린다.

    BN은 `train()` 상태에서 순전파만 해도 running_mean/var가 바뀐다. 고정한 백본(FE의 전체 백본,
    Partial의 layer1~3)의 특징이 몰래 바뀌지 않도록 고정 BN은 ImageNet 통계를 그대로 쓴다.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.weight is not None and not m.weight.requires_grad:
            m.eval()
    return model


def param_groups(model: nn.Module, strategy: str, lr_head: float = 1e-3, lr_backbone: float = 1e-4) -> list[dict]:
    """옵티마이저 파라미터 그룹. Scratch는 전체를 lr_head로, TL은 사전학습 백본만 lr_backbone로 학습한다."""
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if (strategy == "scratch" or name.startswith("fc.")) else backbone).append(p)
    groups = [{"params": head, "lr": lr_head, "name": "head" if strategy != "scratch" else "all"}]
    if backbone:
        groups.append({"params": backbone, "lr": lr_backbone, "name": "backbone"})
    return groups


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}

"""ResNet18 마지막 합성곱 묶음(layer4)의 Grad-CAM(새 패키지 없이 hook으로 구현).

Grad-CAM: 어떤 클래스 점수를 layer4 특징 지도로 미분한 값의 공간 평균을 채널 가중치로 쓰고,
특징 지도를 가중합한 뒤 ReLU를 씌운다. 그 클래스 판단에 "어디가 기여했는가"의 거친(7×7) 지도다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gradcam(model: torch.nn.Module, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
    """x: (1,3,H,W) 정규화된 입력. (H×W로 키운 0~1 CAM, 사용한 클래스)를 돌려준다."""
    acts, grads = {}, {}
    layer = model.layer4
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)  # 고정된 층이 있어도 기울기가 layer4까지 흐르게 한다
        logits = model(x)
        c = int(logits.argmax(1)) if class_idx is None else int(class_idx)
        logits[0, c].backward()
        w = grads["g"].mean(dim=(2, 3), keepdim=True)                  # 채널 가중치
        cam = F.relu((w * acts["a"]).sum(1, keepdim=True))             # (1,1,7,7)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam.detach().cpu().numpy(), c
    finally:
        h1.remove()
        h2.remove()

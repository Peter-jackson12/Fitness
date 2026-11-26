"""평가: Macro F1(핵심 지표, D-009), Accuracy, 클래스별 P/R/F1, Confusion Matrix."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn

from models import NUM_CLASSES

LABELS = list(range(NUM_CLASSES))


def summarize(y_true, y_pred) -> dict:
    """정답·예측 배열 → 지표. 예측에 한 번도 안 나온 클래스도 F1=0으로 평균에 넣는다(labels 고정)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "per_class": {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": s.tolist()},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, criterion: nn.Module | None = None, amp: bool = True,
             return_outputs: bool = False) -> dict:
    """`eval()` + `no_grad`로 한 분할을 평가한다. 증강이 없는 loader를 넘겨야 한다."""
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    ys, preds, probs = [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x)
        logits = logits.float()
        total_loss += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
        ys.append(y.cpu())
        preds.append(logits.argmax(1).cpu())
        if return_outputs:
            probs.append(logits.softmax(1).cpu())
    y_true, y_pred = torch.cat(ys).numpy(), torch.cat(preds).numpy()
    out = {"loss": total_loss / n, **summarize(y_true, y_pred)}
    if return_outputs:
        out.update(y_true=y_true, y_pred=y_pred, probs=torch.cat(probs).numpy())
    return out

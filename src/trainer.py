"""학습 루프: AMP(fp16), Val Macro F1 기준 Early Stopping, 최고 시점 가중치 복원(D-012)."""
from __future__ import annotations

import time

import torch
from torch import nn

from metrics import evaluate, summarize
from models import param_groups, set_train_mode


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, amp: bool = True) -> dict:
    set_train_mode(model)
    total_loss, n = 0.0, 0
    ys, preds = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        ys.append(y.detach().cpu())
        preds.append(logits.detach().argmax(1).cpu())
    m = summarize(torch.cat(ys).numpy(), torch.cat(preds).numpy())
    return {"loss": total_loss / n, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}


def fit(model: nn.Module, strategy: str, loaders: dict, device, max_epochs: int, patience: int = 5,
        lr_head: float = 1e-3, lr_backbone: float = 1e-4, weight_decay: float = 1e-4, amp: bool = True,
        log=print, class_weights: torch.Tensor | None = None) -> dict:
    """학습하고 최고 Val Macro F1 시점의 가중치로 되돌린 모델과 기록을 돌려준다.

    Train 지표는 학습 중(가중치가 바뀌는 동안, train 모드)의 값이라 Val 지표와 조건이 다르다. 과적합 경향을 보는 용도다.
    """
    model.to(device)
    # 학습 손실에만 클래스 가중치를 쓴다. Val 손실은 가중치 없이 계산해 다른 실험과 같은 기준으로 비교한다.
    criterion = nn.CrossEntropyLoss()
    train_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) if class_weights is not None else criterion
    optimizer = torch.optim.AdamW(param_groups(model, strategy, lr_head, lr_backbone), weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")

    history, best = [], {"macro_f1": -1.0, "epoch": 0, "state": None}
    wait, stopped_early = 0, False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        _sync(device)
        t0 = time.perf_counter()
        tr = train_one_epoch(model, loaders["train"], optimizer, scaler, train_criterion, device, amp)
        _sync(device)
        t1 = time.perf_counter()
        va = evaluate(model, loaders["val"], device, criterion, amp)
        _sync(device)
        t2 = time.perf_counter()

        improved = va["macro_f1"] > best["macro_f1"]
        if improved:
            best = {"macro_f1": va["macro_f1"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            wait = 0
        else:
            wait += 1
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["accuracy"], "train_macro_f1": tr["macro_f1"],
               "val_loss": va["loss"], "val_acc": va["accuracy"], "val_macro_f1": va["macro_f1"],
               "train_time_s": t1 - t0, "val_time_s": t2 - t1, "best": improved}
        history.append(row)
        log(f"[{strategy}] epoch {epoch:>2}/{max_epochs} | train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} "
            f"| val loss {va['loss']:.4f} acc {va['accuracy']:.4f} F1 {va['macro_f1']:.4f}"
            f"{' *' if improved else ''} | {t1 - t0:.1f}s + {t2 - t1:.1f}s")
        if wait >= patience:
            stopped_early = True
            log(f"[{strategy}] Early stopping: {patience} epoch 연속 개선 없음. 최고 epoch {best['epoch']}")
            break
    total_time = time.perf_counter() - t_start

    model.load_state_dict(best["state"])
    final_val = evaluate(model, loaders["val"], device, criterion, amp)
    return {
        "model": model,
        "history": history,
        "best_epoch": best["epoch"],
        "best_val": final_val,
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "total_time_s": total_time,
        "peak_gpu_mem_mb": (torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None,
        "best_state": best["state"],
    }

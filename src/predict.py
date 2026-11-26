"""저장한 가중치로 한 분할을 예측해 이미지별 표를 만든다(오류 분석용, E06).

Test는 `allow_test=True`를 명시해야만 예측한다(최종 1회 평가 전까지 쓰지 않는다).
"""
from __future__ import annotations

import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import ROOT, class_names
from dataset import FitnessDataset, build_transforms, load_split
from metrics import evaluate
from models import build_model

META_COLS = ["image_id", "image_path", "target_id", "person_id", "clip_id", "state_code", "camera", "frame_no"]


def load_trained(run_id: str, strategy: str, device) -> torch.nn.Module:
    model = build_model(strategy, pretrained=False)  # 저장한 전체 가중치로 덮어쓰므로 ImageNet 가중치가 필요 없다
    state = torch.load(ROOT / "models" / f"{run_id}_{strategy}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def predict_split(run_id: str, strategy: str, split: str = "val", device=None, batch_size: int = 64,
                  allow_test: bool = False) -> pd.DataFrame:
    """이미지별 정답·예측·확률 표. 순서는 메타데이터 CSV와 같다(shuffle 없음)."""
    if split == "test" and not allow_test:
        raise PermissionError("Test 예측은 최종 평가 때만 한다(allow_test=True 필요).")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_split(split)
    cfg = json.loads((ROOT / "experiments" / run_id / strategy / "config.json").read_text(encoding="utf-8"))
    _, eval_tf = build_transforms(cfg.get("input_size", 224))  # 학습 때와 같은 입력 크기
    loader = DataLoader(FitnessDataset(df, eval_tf), batch_size=batch_size, shuffle=False, num_workers=0)
    out = evaluate(load_trained(run_id, strategy, device), loader, device, return_outputs=True)
    names = class_names()
    res = df[META_COLS].copy()
    res["pred"] = out["y_pred"]
    res["correct"] = res["pred"] == res["target_id"]
    res["conf"] = out["probs"].max(1)                                   # 예측한 클래스의 확률
    res["p_true"] = out["probs"][range(len(res)), res["target_id"]]     # 정답 클래스의 확률
    for c in range(out["probs"].shape[1]):
        res[f"p{c}"] = out["probs"][:, c]
    res["true_name"] = res["target_id"].map(names)
    res["pred_name"] = res["pred"].map(names)
    res.attrs["macro_f1"] = out["macro_f1"]
    return res

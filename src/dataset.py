"""Dataset·전처리(letterbox)·DataLoader.

- 분할은 공식 CSV를 그대로 쓴다(O-01). 학습 데이터 비율은 공식 부분집합 CSV(`metadata_train_{25,50,100}pct.csv`)로 고른다.
- Val·Test에는 증강을 쓰지 않는다. Test는 `allow_test=True`일 때만 만든다(최종 1회 평가용, O-05).
- 메타 열(equipment, pose 등)은 입력으로 쓰지 않는다. 라벨은 `target_id`만 쓴다.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from common import DATA_RAW, SPLITS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# 패딩 색: ImageNet 평균색(0~255). 정규화하면 거의 0이 되어 "정보 없음"에 가깝다.
PAD_FILL = tuple(round(m * 255) for m in IMAGENET_MEAN)  # (124, 116, 104)
TRAIN_PCTS = (25, 50, 100)


class Letterbox:
    """가로세로 비율을 유지해 긴 변을 `size`에 맞추고, 남는 부분을 `fill` 색으로 채워 정사각형을 만든다.

    384×216 → 224×126으로 줄인 뒤 위아래에 49픽셀씩 채운다. 사람·기구가 잘리지 않는다(D-011).
    """

    def __init__(self, size: int = 224, fill: tuple[int, int, int] = PAD_FILL,
                 interpolation=Image.Resampling.BILINEAR):
        self.size = size
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = round(w * scale), round(h * scale)
        resized = img.resize((nw, nh), self.interpolation)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas

    def __repr__(self) -> str:
        return f"Letterbox(size={self.size}, fill={self.fill})"


def augment_preset(name: str | None) -> list | None:
    """이름으로 고르는 Train 전용 증강(PIL 원본 384×216에 letterbox 전에 적용).

    - None/"none": 증강 없음(기본 학습, D-011)
    - "basic": 좌우 뒤집기(p=0.5) + 약한 이동(가로·세로 ±5%)·확대축소(0.9~1.1배). 빈 곳은 패딩과 같은 평균색.
      좌우 뒤집기는 운동 종류를 바꾸지 않는다(01_eda H-d). 이동·확대는 카메라 위치·거리의 작은 차이를 흉내 낸다.
    """
    if name in (None, "none"):
        return None
    if name == "basic":
        return [T.RandomHorizontalFlip(0.5),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=PAD_FILL)]
    raise ValueError(f"알 수 없는 증강: {name}")


def balanced_class_weights(train_pct: int = 100, num_classes: int = 12) -> torch.Tensor:
    """클래스 가중치 w_c = N / (K · n_c)(sklearn의 "balanced"와 같은 식). Train 분할의 장수로만 계산한다.

    적은 클래스(바벨 로우 312장)의 손실은 크게(약 2.2), 많은 클래스(로잉머신 1,120장)는 작게(약 0.6) 매긴다.
    Σ n_c · w_c = N이라 손실의 전체 크기는 가중치가 없을 때와 비슷하다.
    """
    counts = load_split("train", train_pct)["target_id"].value_counts().reindex(range(num_classes), fill_value=0)
    n = counts.to_numpy(dtype=float)
    return torch.tensor(n.sum() / (num_classes * n), dtype=torch.float32)


def build_transforms(size: int = 224, train_augment: list | None = None) -> tuple[T.Compose, T.Compose]:
    """(train_tf, eval_tf)를 돌려준다.

    기본 학습(D-011)은 `train_augment=None`이라 두 변환이 같다. 증강은 튜닝·개선 단계에서
    PIL 이미지에 적용할 변환 목록으로 넘기고, letterbox 앞에 붙는다. eval 변환에는 절대 붙지 않는다.
    """
    tail = [Letterbox(size), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    eval_tf = T.Compose(tail)
    train_tf = T.Compose([*(train_augment or []), *tail])
    return train_tf, eval_tf


def load_split(split: str, train_pct: int = 100) -> pd.DataFrame:
    """분할 하나의 메타데이터. train은 공식 부분집합 CSV에서 고른다."""
    if split not in SPLITS:
        raise ValueError(f"split은 {SPLITS} 중 하나: {split}")
    if train_pct not in TRAIN_PCTS:
        raise ValueError(f"train_pct는 {TRAIN_PCTS} 중 하나: {train_pct}")
    csv = DATA_RAW / f"metadata_train_{train_pct}pct.csv"
    md = pd.read_csv(csv, encoding="utf-8-sig")
    return md[md["split"] == split].reset_index(drop=True)


class FitnessDataset(Dataset):
    """메타데이터 행 → (이미지 텐서, target_id)."""

    def __init__(self, df: pd.DataFrame, transform, root=DATA_RAW):
        self.paths = [str(root / p) for p in df["image_path"]]
        self.targets = df["target_id"].astype(int).tolist()
        self.image_ids = df["image_id"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), self.targets[i]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_reproducible(deterministic: bool = True) -> None:
    """GPU 연산을 결정적으로 만든다(기본).

    seed만 고정하면 부족하다. cuDNN benchmark 모드는 프로세스마다 convolution 알고리즘을 다시 골라서,
    같은 seed로도 실행마다 결과가 달랐다(독립검증 ②에서 직접 확인: 2 epoch Val Macro F1 0.371 vs 0.381).
    결정적 모드는 같은 결과를 냈고, 속도는 약 14% 느렸다.
    """
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def _worker_init(worker_id: int) -> None:
    # 각 worker의 numpy·random seed를 torch가 정한 worker seed에 맞춘다(증강 재현성).
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loaders(train_pct: int = 100, batch_size: int = 32, num_workers: int = 2, eval_workers: int = 0,
                 seed: int = 42, size: int = 224, train_augment: list | None = None, allow_test: bool = False) -> dict:
    """train·val(·test) DataLoader를 만든다. train만 섞는다.

    Windows에서는 worker마다 torch·CUDA DLL을 새로 올려서 프로세스당 약 2.5GB의 가상 메모리를 잡는다.
    worker를 train 4 + val 4개 띄웠을 때 페이징 파일 부족(WinError 1455)으로 죽었다(E00 1차 시도).
    그래서 train만 worker 2개, Val·Test는 main 프로세스에서 읽는다(Val 1,440장은 약 2초).
    """
    train_tf, eval_tf = build_transforms(size, train_augment)
    frames = {"train": load_split("train", train_pct), "val": load_split("val", train_pct)}
    if allow_test:
        frames["test"] = load_split("test", train_pct)
    g = torch.Generator()
    g.manual_seed(seed)
    loaders = {}
    for split, df in frames.items():
        is_train = split == "train"
        nw = num_workers if is_train else eval_workers
        ds = FitnessDataset(df, train_tf if is_train else eval_tf)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=is_train, generator=g if is_train else None,
                                    num_workers=nw, pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0,
                                    worker_init_fn=_worker_init if nw > 0 else None)
    return loaders

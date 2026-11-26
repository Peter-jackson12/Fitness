"""노트북과 스크립트가 함께 쓰는 공통 도구(경로, 메타데이터, 클래스 정보, 그림 설정)."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def project_root(start: Path | None = None) -> Path:
    """`pyproject.toml`이 있는 폴더(= fitness/)를 찾는다. 노트북을 어디서 열어도 같은 경로를 쓰기 위함."""
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("pyproject.toml을 찾지 못했습니다. fitness/ 안에서 실행하세요.")


ROOT = project_root(Path(__file__).parent)
DATA_RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "figures"
FIGURES_SAMPLES = FIGURES / "samples"  # 원본 이미지가 들어간 그림(Git 제외)

# 사전학습 가중치 캐시를 프로젝트 안(fitness/.cache/torch, Git 제외)에 둔다. 사용자가 TORCH_HOME을 정했으면 그것을 쓴다.
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
# 결정적 학습(dataset.set_reproducible)에 필요한 cuBLAS 설정. cuBLAS가 처음 쓰이기 전에 정해져 있어야 해서 import 때 둔다.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SPLITS = ("train", "val", "test")

# 촬영 묶음(EDA에서 확인): 운동 묶음마다 촬영 공간과 기구 종류가 다르다.
GROUPS = {
    "맨몸(바닥)": [0, 1, 4, 10],
    "바벨·덤벨(선 자세)": [2, 7, 8, 9],
    "기구": [3, 5, 6, 11],
}
CLASS_TO_GROUP = {c: g for g, cs in GROUPS.items() for c in cs}


def load_metadata(split: str | None = None) -> pd.DataFrame:
    """metadata.csv를 읽는다. split을 주면 그 분할만 돌려준다.

    주의: Test는 최종 평가 전까지 무결성 확인 외의 분석에 쓰지 않는다.
    """
    md = pd.read_csv(DATA_RAW / "metadata.csv", encoding="utf-8-sig")
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"split은 {SPLITS} 중 하나여야 합니다: {split}")
        md = md[md["split"] == split].reset_index(drop=True)
    return md


def class_names() -> dict[int, str]:
    cm = pd.read_csv(DATA_RAW / "class_mapping.csv", encoding="utf-8-sig")
    return dict(zip(cm["target_id"], cm["exercise"]))


def set_korean_font() -> str:
    """matplotlib 한글 폰트 설정. Windows(맑은 고딕), macOS(AppleGothic), 그 외(NanumGothic) 순으로 찾는다."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        name = plt.rcParams["font.family"][0]
    plt.rcParams["axes.unicode_minus"] = False
    return name

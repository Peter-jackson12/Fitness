"""README용 데이터 그림 3장을 저장된 메타데이터·예측 파일만 읽어 그린다(학습·추론 없음).

- figures/clip_structure.png       : 클립 구조(시점 × 카메라). Train 메타데이터만 사용
- figures/e06_error_by_camera.png  : Val 바벨·덤벨 묶음의 운동 × 카메라 오답률(E04 Full 예측, E06과 같은 자료)
- figures/e08_confusion_pairs.png  : Test 1회 평가(E08) 예측에서 덤벨 로우·데드리프트·바벨 로우 사이 혼동 장수

원본 사진은 그리지 않는다. 실행: uv run python src/readme_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGURES, ROOT, class_names, load_metadata, set_korean_font  # noqa: E402

CAMS = list("ABCDE")
CAM_DESC = {"A": "A(뒤)", "B": "B(뒤)", "C": "C(정면)", "D": "D(옆)", "E": "E(옆)"}  # 05_error_analysis 3.1절의 시점 서술
BB = [2, 7, 8, 9]          # 바벨·덤벨(선 자세) 묶음
HARD = [2, 7, 9]           # 덤벨 로우·데드리프트·바벨 로우
CMAP = "Blues"             # 크기(장수·비율)는 한 색 계열로
INK, MUTED = "#1f2328", "#59636e"


def _cell_text(ax, i, j, text, value, vmax):
    color = "white" if value > 0.55 * vmax else INK
    ax.text(j, i, text, ha="center", va="center", fontsize=10, color=color)


def clip_structure(names):
    tr = load_metadata("train")
    stats = tr.groupby("clip_id").apply(
        lambda d: (len(d), d.frame_no.nunique(), d.groupby("frame_no").camera.nunique().min(),
                   d.groupby("frame_no").camera.nunique().max(), d.camera.nunique()),
        include_groups=False,
    )
    n_clips = len(stats)
    n_ok = int((stats == (8, 4, 2, 2, 5)).sum())
    # 묶음마다 한 클립씩 예시(고정 seed)
    examples = [tr[tr.target_id == c].clip_id.drop_duplicates().sample(1, random_state=7).iloc[0] for c in (1, 9, 5)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    for ax, cid in zip(axes, examples):
        d = tr[tr.clip_id == cid]
        frames = sorted(d.frame_no.unique())
        grid = np.zeros((len(frames), len(CAMS)))
        for _, r in d.iterrows():
            grid[frames.index(r.frame_no), CAMS.index(r.camera)] = 1
        ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(["#f3f4f6", "#3b6fb6"]), vmin=0, vmax=1)
        ax.set_xticks(range(5), [CAM_DESC[c] for c in CAMS], fontsize=9)
        ax.set_yticks(range(len(frames)), [f"시점 {k + 1}\n(frame {f})" for k, f in enumerate(frames)], fontsize=8)
        ax.set_xticks(np.arange(-.5, 5), minor=True); ax.set_yticks(np.arange(-.5, len(frames)), minor=True)
        ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
        ax.set_title(f"{names[d.target_id.iloc[0]]} 클립 예시", fontsize=10, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"클립 1개 = 4시점 × 시점마다 다른 카메라 2대 = 8장 (Train {n_clips:,}클립 중 {n_ok:,}클립이 이 구조)",
                 fontsize=11, color=INK)
    fig.text(0.5, 0.015, "파란 칸 = 데이터에 들어 있는 사진. 시점마다 고른 카메라 2대는 클립마다 다르고, 한 클립 안에서 카메라 5대가 모두 쓰입니다.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    out = FIGURES / "clip_structure.png"; fig.savefig(out, dpi=120, facecolor="white"); plt.close(fig)
    return {"clips": n_clips, "clips_8x4x2": n_ok, "examples": examples}


def error_by_camera(names):
    p = pd.read_csv(ROOT / "experiments" / "E06" / "pred_val_E04_full.csv", encoding="utf-8-sig")
    b = p[p.target_id.isin(BB)]
    err = 1 - b.pivot_table(index="target_id", columns="camera", values="correct", aggfunc="mean").reindex(index=BB, columns=CAMS)
    n = b.pivot_table(index="target_id", columns="camera", values="image_id", aggfunc="size").reindex(index=BB, columns=CAMS)
    wrong = (err * n).round().astype(int)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.imshow(err.values, cmap=CMAP, vmin=0, vmax=1)
    for i in range(len(BB)):
        for j in range(5):
            _cell_text(ax, i, j, f"{err.values[i, j]:.0%}\n({wrong.values[i, j]}/{n.values[i, j]})", err.values[i, j], 1)
    ax.set_xticks(range(5), [CAM_DESC[c] for c in CAMS]); ax.set_yticks(range(len(BB)), [names[c] for c in BB])
    ax.set_xlabel("카메라", color=MUTED); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Validation 오답률: 바벨·덤벨 묶음 × 카메라 (E04 Full FT, 사람 Z23)", fontsize=11, color=INK)
    fig.tight_layout()
    out = FIGURES / "e06_error_by_camera.png"; fig.savefig(out, dpi=120, facecolor="white"); plt.close(fig)
    return {"err": err.round(3).to_dict(), "n": n.to_dict()}


def confusion_pairs(names):
    p = pd.read_csv(ROOT / "experiments" / "E08" / "pred_test.csv", encoding="utf-8-sig")
    h = p[p.target_id.isin(HARD)].copy()
    h["pred_g"] = h.pred.where(h.pred.isin(HARD), -1)
    cols = HARD + [-1]
    m = pd.crosstab(h.target_id, h.pred_g).reindex(index=HARD, columns=cols, fill_value=0)
    row_n = m.sum(axis=1)

    labels = [names[c].replace(" ", "\n", 1) for c in HARD] + ["그 밖의\n운동"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    rate = m.div(row_n, axis=0).values
    ax.imshow(rate, cmap=CMAP, vmin=0, vmax=1)
    for i in range(len(HARD)):
        for j in range(len(cols)):
            _cell_text(ax, i, j, f"{m.values[i, j]}장\n({rate[i, j]:.0%})", rate[i, j], 1)
    ax.set_xticks(range(len(cols)), labels); ax.set_yticks(range(len(HARD)), [f"{names[c]} ({row_n[c]}장)" for c in HARD])
    ax.set_xlabel("모델 예측", color=MUTED); ax.set_ylabel("정답", color=MUTED, rotation=0, labelpad=18); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("최종 Test(1회): 세 운동 사이의 혼동 장수 (Full FT, seed 43)", fontsize=11, color=INK)
    fig.tight_layout()
    out = FIGURES / "e08_confusion_pairs.png"; fig.savefig(out, dpi=120, facecolor="white"); plt.close(fig)
    return {"matrix": m.to_dict(), "row_n": row_n.to_dict()}


def main():
    set_korean_font()
    names = class_names()
    print("clip_structure:", clip_structure(names))
    print("error_by_camera:", error_by_camera(names))
    print("confusion_pairs:", confusion_pairs(names))


if __name__ == "__main__":
    main()

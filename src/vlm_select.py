"""VLM 분석 대상 이미지를 규칙대로 고른다(호출 없음). 공식 Val, 선정 모델 SEL-A-s43(D-031) 예측 기준.

규칙(이미지를 보기 전에 정함, experiments/VLM/PLAN.md)
- 오분류: 클래스마다 상한(ERR_CAP)까지. 같은 클립에서는 1장만(8장이 거의 같은 장면). 확신도가 높은 오답부터,
  카메라를 돌아가며 고른다(한 카메라에 몰리지 않게).
- 정답: 클래스마다 1장. 그 클래스 오답이 가장 많은 카메라에서(정답이 없으면 다음 카메라), 정답 확률이 가장 높은 사진(같은 시점의 맞힌 사례와 대조).
  바벨·덤벨 묶음(2·7·9)은 다른 카메라에서 1장 더.
- Test는 쓰지 않는다.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

ERR_CAP = {2: 8, 7: 8, 9: 8, 8: 5, 4: 4, 10: 4, 0: 1, 5: 1, 11: 1}
PAIR_EXTRA = (2, 7, 9)
OUT = ROOT / "experiments" / "VLM" / "candidates.csv"

r = pd.read_csv(ROOT / "experiments" / "E06" / "pred_val_SEL-A-s43.csv")
rows = []
for c, cap in ERR_CAP.items():
    w = r[(r.target_id == c) & ~r.correct].sort_values("conf", ascending=False).drop_duplicates("clip_id")
    picked, cams = [], {}
    for cam, g in w.groupby("camera"):
        cams[cam] = list(g.itertuples())
    while len(picked) < cap and any(cams.values()):
        for cam in sorted(cams, key=lambda k: -(cams[k][0].conf if cams[k] else -1)):
            if cams[cam] and len(picked) < cap:
                picked.append(cams[cam].pop(0))
    rows += [dict(kind="오분류", **x._asdict()) for x in picked]
for c in range(12):
    ok = r[(r.target_id == c) & r.correct]
    w = r[(r.target_id == c) & ~r.correct]
    cam_order = list(w.camera.value_counts().index) + [k for k in "ABCDE" if k not in set(w.camera)]
    n = 2 if c in PAIR_EXTRA else 1
    got = 0
    for cam in cam_order:  # 그 카메라에 정답이 없으면 다음으로 오답이 많은 카메라로 넘어간다
        g = ok[ok.camera == cam].sort_values("p_true", ascending=False)
        if len(g) and got < n:
            rows.append(dict(kind="정답", **next(g.itertuples())._asdict()))
            got += 1
out = pd.DataFrame(rows).drop(columns=["Index"])
keep = ["kind", "image_id", "image_path", "clip_id", "person_id", "camera", "frame_no", "state_code", "target_id", "true_name",
        "pred", "pred_name", "conf", "p_true"]
out = out[keep]
assert out.image_id.is_unique and out.clip_id.groupby(out.kind).apply(lambda s: s.is_unique).all()
out.to_csv(OUT, index=False, encoding="utf-8-sig")
print(out.groupby(["kind", "true_name"]).size().unstack(0).fillna(0).astype(int).to_string())
print("총", len(out), "장 →", OUT)

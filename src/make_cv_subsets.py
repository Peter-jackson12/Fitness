"""사람 단위 CV용 학습 데이터 부분 집합(25%·50%)을 한 번 만들어 `data/splits/cv3_train_subsets.csv`에 고정한다(DS 실험).

공식 `metadata_train_{25,50}pct.csv`의 규칙을 따른다: 사람×클래스마다 이미지 단위로 정확히 그 비율을 무작위로 남기고,
25%는 50% 안에 포함된다.
- 공식 Train 16명: 공식 25·50% 파일의 선택을 그대로 쓴다.
- 공식 Val 3명: 같은 규칙으로 고정 seed(2026)로 뽑는다.
- Test 3명은 넣지 않는다(E08 동결). 사람마다 선택은 fold와 상관없이 같다.
- 이미 파일이 있으면 덮어쓰지 않는다.

사용: python src/make_cv_subsets.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402
from dataset import load_split  # noqa: E402

OUT = ROOT / "data" / "splits" / "cv3_train_subsets.csv"
SEED = 2026
if OUT.exists():
    sys.exit(f"{OUT} 가 이미 있다. 부분 집합은 고정이므로 다시 만들지 않는다.")

tr, va = load_split("train", 100), load_split("val", 100)
off = {p: set(load_split("train", p)["image_id"]) for p in (25, 50)}
tr = tr.assign(in25=tr.image_id.isin(off[25]), in50=tr.image_id.isin(off[50]), source="official")

rng = np.random.default_rng(SEED)
parts = []
for (_, _), g in va.sort_values("image_id").groupby(["person_id", "target_id"], sort=True):
    order = g.image_id.to_numpy()[rng.permutation(len(g))]
    k25, k50 = int(round(len(g) * 0.25)), int(round(len(g) * 0.50))
    parts.append(g.assign(in25=g.image_id.isin(order[:k25]), in50=g.image_id.isin(order[:k50])))
va = pd.concat(parts).assign(source="sampled_seed2026")

out = pd.concat([tr, va], ignore_index=True)[["image_id", "person_id", "target_id", "in25", "in50", "source"]]
out = out.sort_values("image_id").reset_index(drop=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False, encoding="utf-8")
print(out.groupby("source")[["in25", "in50"]].agg(["sum", "size"]).to_string())

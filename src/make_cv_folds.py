"""사람 단위 3-fold 교차검증 분할을 한 번 만들어 `data/splits/cv3_folds.csv`에 고정한다(D-037).

- 대상: 공식 Train + Val의 19명. Test 3명은 넣지 않는다(E08 동결).
- 각 사람은 한 fold에만 들어간다. 운동 묶음마다 fold에 고르게 나눈다.
  - 바벨·덤벨 6명 → 2/2/2
  - Z84는 바닥·기구 두 묶음을 모두 하므로 먼저 한 fold에 넣고, 나머지로 묶음별 인원을 맞춘다.
- 배정 순서는 고정 seed의 무작위 순열이다(사람 특성을 보고 고르지 않음).
- 이미 파일이 있으면 덮어쓰지 않는다.

사용: python src/make_cv_folds.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402
from dataset import load_split  # noqa: E402

OUT = ROOT / "data" / "splits" / "cv3_folds.csv"
SEED = 2026
if OUT.exists():
    sys.exit(f"{OUT} 가 이미 있다. 분할은 고정이므로 다시 만들지 않는다.")

d = pd.concat([load_split("train"), load_split("val")])
classes = d.groupby("person_id")["target_id"].apply(lambda s: frozenset(s.unique()))
BAR, FLOOR, MACH = frozenset({2, 7, 8, 9}), frozenset({0, 1, 4, 10}), frozenset({3, 5, 6, 11})
both = [p for p, c in classes.items() if c >= FLOOR | MACH]
assert both == ["Z84"], both
bar = sorted(p for p, c in classes.items() if c == BAR)
floor = sorted(p for p, c in classes.items() if c == FLOOR)
mach = sorted(p for p, c in classes.items() if c == MACH)
assert (len(bar), len(floor), len(mach)) == (6, 5, 7), (bar, floor, mach)

rng = np.random.default_rng(SEED)
fold = {}
z84 = int(rng.integers(1, 4))
fold["Z84"] = z84
for p, f in zip(rng.permutation(bar), [1, 1, 2, 2, 3, 3]):
    fold[p] = f
# 바닥: Z84 fold에 1명, 나머지 fold에 2명씩(총 2명씩)
slots = [f for f in (1, 2, 3) for _ in range(1 if f == z84 else 2)]
for p, f in zip(rng.permutation(floor), slots):
    fold[p] = f
# 기구 7명: Z84 fold에 2명, 나머지 두 fold에 3명·2명(총 3/3/2)
others = [f for f in (1, 2, 3) if f != z84]
slots = [z84] * 2 + [others[0]] * 3 + [others[1]] * 2
for p, f in zip(rng.permutation(mach), slots):
    fold[p] = f

out = pd.DataFrame({"person_id": list(fold), "fold": list(fold.values())})
out["group"] = out["person_id"].map(lambda p: "바닥+기구" if p == "Z84" else ("바벨·덤벨" if p in bar else ("바닥" if p in floor else "기구")))
out["official_split"] = out["person_id"].map(d.drop_duplicates("person_id").set_index("person_id")["split"])
out = out.sort_values(["fold", "group", "person_id"]).reset_index(drop=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False, encoding="utf-8")
print(out.to_string())

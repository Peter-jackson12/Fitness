"""CV 학습 부분 집합(`data/splits/cv3_train_subsets.csv`) 검사(DS 실험, 학습 전).

실행: .venv/Scripts/python.exe -X utf8 checks/check_cv_subsets.py
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataset import CV_FOLDS, CV_SUBSETS, load_cv, load_split  # noqa: E402

n_ok = 0


def ok(cond, msg):
    global n_ok
    if not cond:
        raise SystemExit(f"FAIL {msg}")
    n_ok += 1
    print("OK ", msg)


sub = pd.read_csv(CV_SUBSETS)
folds = pd.read_csv(CV_FOLDS)
print("cv3_train_subsets.csv sha256:", hashlib.sha256(CV_SUBSETS.read_bytes()).hexdigest())
tr100, va100, te = load_split("train", 100), load_split("val", 100), load_split("test", 100)

ok(set(sub.person_id) == set(folds.person_id) and sub.person_id.nunique() == 19, "사람 19명 = cv3_folds.csv의 사람")
ok(set(sub.image_id) == set(tr100.image_id) | set(va100.image_id) and len(sub) == len(tr100) + len(va100), "이미지 = 공식 Train 100% + Val, 중복 없음")
ok(not (set(sub.image_id) & set(te.image_id)) and not (set(sub.person_id) & set(te.person_id)), "Test 이미지·사람 없음")
ok(not (sub.in25 & ~sub.in50).any(), "25% ⊂ 50%")

g = sub.groupby(["person_id", "target_id"])
for p in (25, 50):
    dev = (g[f"in{p}"].sum() - (g.size() * p / 100).round()).abs()
    ok(dev.max() <= 1, f"{p}%: 사람×클래스마다 비율대로(반올림 1장 이내, 최대 차이 {dev.max():.0f})")
    off = set(load_split("train", p).image_id)
    o = sub[sub.source == "official"]
    ok(set(o.image_id[o[f"in{p}"]]) == off, f"{p}%: 공식 Train 16명은 공식 부분 집합과 같음({len(off)}장)")

for f in (1, 2, 3):
    d = pd.concat([tr100, va100], ignore_index=True).merge(folds[["person_id", "fold"]], on="person_id")
    old_tr, old_va = d[d.fold != f].drop(columns="fold").reset_index(drop=True), d[d.fold == f].drop(columns="fold").reset_index(drop=True)
    t100, v100 = load_cv(f, 100)
    ok(t100.equals(old_tr) and v100.equals(old_va), f"fold {f}: 100%는 기존 load_cv와 같은 데이터·순서")
    for p in (25, 50):
        t, v = load_cv(f, p)
        ok(v.equals(v100), f"fold {f} {p}%: 검증은 100% 그대로({len(v)}장)")
        ok(set(t.image_id) <= set(t100.image_id) and not (set(t.person_id) & set(v.person_id)), f"fold {f} {p}%: 학습 ⊂ 100% 학습, 검증 사람과 겹치지 않음")
        ok(t.target_id.nunique() == 12 and set(t.person_id) == set(t100.person_id),
           f"fold {f} {p}%: 12클래스·학습 사람 {t.person_id.nunique()}명 모두 남음, 학습 {len(t)}장({len(t) / len(t100):.3f})")
    t25, _ = load_cv(f, 25); t50, _ = load_cv(f, 50)
    ok(set(t25.image_id) <= set(t50.image_id), f"fold {f}: 학습 25% ⊂ 50%")

print(f"\n전체 통과: {n_ok}개 항목")

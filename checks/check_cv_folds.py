"""교차검증 fold 파일(data/splits/cv3_folds.csv) 검사. 실행: uv run python checks/check_cv_folds.py"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataset import load_split  # noqa: E402

F = ROOT / "data" / "splits" / "cv3_folds.csv"
folds = pd.read_csv(F)
print("파일 SHA-256:", hashlib.sha256(F.read_bytes()).hexdigest())
d = pd.concat([load_split("train"), load_split("val")])
te = load_split("test")
n = 0
def check(cond, msg):
    global n
    assert cond, msg
    n += 1
    print("OK ", msg)

check(folds["person_id"].is_unique, "사람마다 한 줄(한 fold에만 배정)")
check(set(folds["person_id"]) == set(d["person_id"]), f"Train+Val의 {d.person_id.nunique()}명이 모두 fold에 배정됨")
check(not set(folds["person_id"]) & set(te["person_id"]), "Test 사람은 fold에 없음(E08 동결)")
check(sorted(folds["fold"].unique()) == [1, 2, 3], "fold 1·2·3")
m = d.merge(folds[["person_id", "fold"]], on="person_id", how="left")
check(m["fold"].notna().all() and len(m) == 8144 + 1440, f"이미지 {len(m):,}장 모두 fold가 있음")
check(m.groupby("clip_id")["fold"].nunique().max() == 1, "클립이 fold 사이에 갈라지지 않음")
for f in (1, 2, 3):
    va, tr = m[m.fold == f], m[m.fold != f]
    check(set(va.target_id) == set(range(12)) and set(tr.target_id) == set(range(12)), f"fold {f}: 검증·학습 모두 12클래스")
    check(not set(va.person_id) & set(tr.person_id), f"fold {f}: 검증·학습 사이 사람 겹침 없음")
print()
print(m.pivot_table(index="target_id", columns="fold", values="image_id", aggfunc="count").assign(합계=lambda t: t.sum(axis=1)).to_string())
print("\nfold별 검증 사람:", m.groupby("fold")["person_id"].unique().map(sorted).to_dict())
pp = m.groupby(["target_id", "fold"]).person_id.nunique().unstack()
cc = m.groupby(["target_id", "fold"]).clip_id.nunique().unstack()
check(pp.min().min() >= 2, f"모든 fold·클래스의 검증 사람이 2명 이상(최소 {int(pp.min().min())}명)")
check(cc.min().min() >= 15, f"모든 fold·클래스의 검증 클립이 15개 이상(최소 {int(cc.min().min())}개)")
print("\nfold별 클래스별 검증 사람·클립 수")
print((pp.astype(str) + "명/" + cc.astype(str) + "클립").to_string())
print(f"\n전체 통과: {n}개 항목")

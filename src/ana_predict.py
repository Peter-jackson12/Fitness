"""ANA(남은 진단 분석) 추론: 저장한 가중치로 예측표를 만든다. 학습하지 않는다. Test는 쓰지 않는다.

실행: .venv/Scripts/python.exe -X utf8 src/ana_predict.py
출력: experiments/ANA/pred/<run_id>__train.csv (이미 있으면 건너뛴다. IMP에 같은 예측표가 있으면 복사한다)
"""
from __future__ import annotations

import shutil
import time

from common import ROOT
from predict import predict_split

OUT = ROOT / "experiments" / "ANA" / "pred"
IMP = ROOT / "experiments" / "IMP" / "pred"
PRE = ("letterbox", "stretch", "aspect384", "ccrop")
JOBS = ([((f"CV1-{p}-f{f}" if s == 42 else f"CV2-{p}-s{s}-f{f}"), "full") for p in PRE for s in (42, 43, 44) for f in (1, 2, 3)]
        + [("E01", "scratch"), ("E04", "full")])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for run_id, strat in JOBS:
        path = OUT / f"{run_id}__train.csv"
        if path.exists():
            print("skip", path.name)
            continue
        src = IMP / f"{run_id}__train.csv"
        if src.exists():
            shutil.copyfile(src, path)
            print("copy", path.name)
            continue
        if not (ROOT / "models" / f"{run_id}_{strat}.pt").exists():
            raise FileNotFoundError(f"가중치 없음: {run_id}_{strat}.pt")
        t0 = time.time()
        df = predict_split(run_id, strat, split="val", batch_size=32)  # 학습 때 평가와 같은 배치(fp16 근소 차이 방지)
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"{path.name}: {len(df)}장, Macro F1 {df.attrs['macro_f1']:.4f}, {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

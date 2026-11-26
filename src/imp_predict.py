"""IMP(VLM 기반 개선 가설) 추론: 저장한 가중치로 예측표를 만든다. 학습하지 않는다. Test는 쓰지 않는다.

실행: .venv/Scripts/python.exe -X utf8 src/imp_predict.py
출력: experiments/IMP/pred/<run_id>__<입력>.csv (이미 있으면 건너뛴다)
"""
from __future__ import annotations

import time

from common import ROOT
from predict import predict_split

OUT = ROOT / "experiments" / "IMP" / "pred"
JOBS = ([(f"CV1-letterbox-f{f}", None) for f in (1, 2, 3)]            # H1 주 판정, H3 컬러 기준
        + [(f"CV1-letterbox-f{f}", "gray") for f in (1, 2, 3)]        # H3 방법 1: 컬러 모델에 흑백 입력
        + [(f"CV1-ccrop-f{f}", None) for f in (1, 2, 3)]              # H1 보조
        + [(f"CV1-gray-f{f}", None) for f in (1, 2, 3)]               # H3 방법 2
        + [("E04", None), ("SEL-A-s43", None), ("SEL-A-s44", None)])  # H1 보조(공식 Val)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for run_id, pre in JOBS:
        path = OUT / f"{run_id}__{pre or 'train'}.csv"
        if path.exists():
            print("skip", path.name)
            continue
        if not (ROOT / "models" / f"{run_id}_full.pt").exists():
            print("가중치 없음, 건너뜀:", run_id)
            continue
        t0 = time.time()
        df = predict_split(run_id, "full", split="val", preprocess=pre, batch_size=32)  # 학습 때 평가와 같은 배치
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"{path.name}: {len(df)}장, Macro F1 {df.attrs['macro_f1']:.4f}, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

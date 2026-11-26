"""VLM 오류 분석 실행(D-040, experiments/VLM/PLAN.md). 공식 Validation 이미지만 보낸다.

- 키는 `fitness/.env`의 OPENAI_API_KEY에서만 읽는다(사용자가 직접 넣음). 키를 출력하거나 기록하지 않는다.
- 사진마다 두 번 호출한다.
  1) blind: 정답·예측을 알려 주지 않고 12개 운동 중 무엇으로 보이는지
  2) explain: 정답과 모델 예측을 알려 주고 안내문 항목(특징, 구분 단서, 오류 요인, 판단 어려운 점)
- 호출 수 상한(MAX_CALLS)과 누적 비용 상한(--budget-usd)을 넘기 전에 멈춘다. 학원 키는 초과해도 자동 차단이 없다.
- 응답은 `experiments/VLM/results.jsonl`에 한 줄씩 원문 그대로 남긴다. 다시 실행하면 이미 끝난 (이미지, 단계)는 건너뛴다.

사용
  python src/vlm_analyze.py --list-models                  # 모델 목록(무료)
  python src/vlm_analyze.py --model M --probe              # 최소 호출 1회(이미지 1장, blind)
  python src/vlm_analyze.py --model M --price-in X --price-out Y --budget-usd 3
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_RAW, ROOT, class_names  # noqa: E402

VLM_DIR = ROOT / "experiments" / "VLM"
RESULTS = VLM_DIR / "results.jsonl"
MAX_CALLS = 110
NAMES = class_names()
CLASS_LIST = ", ".join(f"{i} {n}" for i, n in NAMES.items())

CONTEXT = ("헬스장에 고정된 카메라로 찍은 384×216 사진 한 장입니다. 한 사람이 12가지 운동 중 하나를 하고 있습니다. "
           f"운동 목록: {CLASS_LIST}. 사진은 동작 중 한 순간이라 시작·중간·끝 자세 어느 것일 수도 있습니다.")

BLIND = CONTEXT + """
정답을 모른다고 보고, 사진에서 보이는 것만으로 판단하세요. 반드시 아래 JSON 하나만 답하세요.
{"guess_id": 정수, "guess_name": "운동 이름", "confidence": "높음|중간|낮음",
 "second_guess_id": 정수, "evidence": ["판단 근거가 된 시각 단서(자세, 기구, 손 위치, 상체 각도 등)"],
 "ambiguity": "다른 운동과 헷갈릴 수 있는 이유(없으면 빈 문자열)"}"""

EXPLAIN = CONTEXT + """
이 사진의 정답은 "{true}"이고, 분류 모델(ResNet18)의 예측은 "{pred}"입니다({result}).
모델이 왜 그렇게 예측했는지 단정하지 말고, 사진에서 확인되는 사실과 가설을 구분해서 답하세요. 반드시 아래 JSON 하나만 답하세요.
{{"visual_features": ["사진에서 관찰되는 주요 시각적 특징"],
 "distinguishing_cues": ["정답 운동과 예측 운동을 가르는 특징이 이 사진에서 보이는지, 보이지 않는지"],
 "possible_error_factors": ["오분류와 관련 있을 수 있는 이미지 특성(가설)"],
 "image_conditions": {{"background": "", "lighting": "", "resolution_or_size": "", "person_size_or_occlusion": "", "camera_view": ""}},
 "motion_phase": "동작 단계 추정(시작/중간/끝/알 수 없음)과 근거",
 "hard_to_judge": ["사진 한 장만으로 판단하기 어려운 점"],
 "follow_up": ["추가로 확인하거나 실험해 볼 것"]}}"""


def read_key() -> str:
    env = ROOT / ".env"
    if not env.exists():
        sys.exit("fitness/.env가 없다. 사용자가 OPENAI_API_KEY=... 한 줄을 직접 넣어야 한다.")
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("OPENAI_API_KEY="):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            if v:
                return v
    sys.exit("fitness/.env에 OPENAI_API_KEY 값이 없다.")


def data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def done_keys() -> set:
    if not RESULTS.exists():
        return set()
    return {(r["image_id"], r["stage"]) for r in map(json.loads, RESULTS.read_text(encoding="utf-8").splitlines()) if r.get("ok")}


def spent_usd(price_in: float, price_out: float) -> float:
    if not RESULTS.exists():
        return 0.0
    tot = 0.0
    for r in map(json.loads, RESULTS.read_text(encoding="utf-8").splitlines()):
        tot += r.get("input_tokens", 0) / 1e6 * price_in + r.get("output_tokens", 0) / 1e6 * price_out
    return tot


def call(client, model: str, prompt: str, img: Path) -> tuple[dict, object]:
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt},
                                            {"type": "input_image", "image_url": data_url(img), "detail": "high"}]}],
        text={"format": {"type": "json_object"}},
        max_output_tokens=1500,
    )
    return json.loads(resp.output_text), resp.usage


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--probe", action="store_true", help="이미지 1장 blind 1회만 호출(가능 여부·토큰 확인)")
    p.add_argument("--price-in", type=float, help="USD / 1M 입력 토큰(공식 가격표에서 확인한 값)")
    p.add_argument("--price-out", type=float, help="USD / 1M 출력 토큰")
    p.add_argument("--budget-usd", type=float, default=3.0)
    a = p.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=read_key())
    if a.list_models:
        ids = sorted(m.id for m in client.models.list())
        print("\n".join(ids))
        return
    if not a.model:
        sys.exit("--model이 필요하다")

    cand = pd.read_csv(VLM_DIR / "candidates.csv")
    if a.probe:
        row = cand.iloc[0]
        t0 = time.time()
        out, usage = call(client, a.model, BLIND, DATA_RAW / row.image_path)
        print(json.dumps({"model": a.model, "image_id": row.image_id, "seconds": round(time.time() - t0, 1),
                          "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "answer": out},
                         ensure_ascii=False, indent=1))
        return
    if a.price_in is None or a.price_out is None:
        sys.exit("본 실행에는 --price-in, --price-out(공식 단가)이 필요하다(비용 상한 계산)")

    done = done_keys()
    n_calls = sum(1 for _ in open(RESULTS, encoding="utf-8")) if RESULTS.exists() else 0
    VLM_DIR.mkdir(parents=True, exist_ok=True)
    for row in cand.itertuples():
        for stage in ("blind", "explain"):
            if (row.image_id, stage) in done:
                continue
            if n_calls >= MAX_CALLS:
                sys.exit(f"호출 상한 {MAX_CALLS}회에 닿아 멈춤")
            spent = spent_usd(a.price_in, a.price_out)
            if spent >= a.budget_usd * 0.95:
                sys.exit(f"누적 추정 비용 ${spent:.3f}가 상한 ${a.budget_usd}의 95%에 닿아 멈춤")
            prompt = BLIND if stage == "blind" else EXPLAIN.format(
                true=row.true_name, pred=row.pred_name, result="맞힘" if row.kind == "정답" else "틀림")
            rec = {"image_id": row.image_id, "stage": stage, "kind": row.kind, "true": int(row.target_id), "pred": int(row.pred),
                   "camera": row.camera, "model": a.model, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            try:
                out, usage = call(client, a.model, prompt, DATA_RAW / row.image_path)
                rec.update(ok=True, answer=out, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
            except Exception as e:  # 실패도 기록하고(비용 0으로) 다음으로
                rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
            with open(RESULTS, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_calls += 1
            print(f"{n_calls:3d} {row.image_id} {stage} ok={rec['ok']} 누적 ${spent_usd(a.price_in, a.price_out):.3f}", flush=True)
    print("완료")


if __name__ == "__main__":
    main()

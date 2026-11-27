"""VLM 오류 분석 실행(D-040, experiments/VLM/PLAN.md). 공식 Validation 이미지만 보낸다.

- 키는 `fitness/.env`의 OPENAI_API_KEY에서만 읽는다(사용자가 직접 넣음). 키를 출력하거나 기록하지 않는다.
- 사진마다 두 번 호출한다.
  1) blind: 정답·예측을 알려 주지 않고 12개 운동 중 무엇으로 보이는지
  2) explain: 정답과 모델 예측을 알려 주고 안내문 항목(특징, 구분 단서, 오류 요인, 판단 어려운 점)
- 본 실행 기록은 최대 MAX_CALLS건, probe는 별도 1건이다. 누적 추정 비용과 다음 호출 예상 비용으로 중단한다.
- 실제 청구액의 엄격한 상한은 아니다. SDK 자동 재시도를 끄고 사용량을 알 수 없는 실패 뒤에는 중단한다.
- 응답은 `experiments/VLM/results.jsonl`에 한 줄씩 원문 그대로 남긴다. 다시 실행하면 이미 끝난 (이미지, 단계)는 건너뛴다.

사용
  python src/vlm_analyze.py --list-models                  # 모델 목록(무료)
  python src/vlm_analyze.py --model M --probe --price-in X --price-out Y  # 별도 기록 1회
  python src/vlm_analyze.py --model M --price-in X --price-out Y --budget-usd 3
"""
from __future__ import annotations

import argparse
import base64
import json
import math
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
MAX_OUTPUT_TOKENS = 1500
# 입력 토큰 예약은 추정치다. 이미지 토큰·실제 청구액의 상한을 보장하지 않는다.
ESTIMATED_INPUT_RESERVE = 10000
PROBE_RESULTS = VLM_DIR / "probe_results.jsonl"
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
    tot = 0.0
    for path in (RESULTS, PROBE_RESULTS):
        if not path.exists():
            continue
        for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
            if r.get("input_tokens") is None or r.get("output_tokens") is None:
                raise RuntimeError("사용량 미확인 기록이 있습니다. 제공자 청구 내역을 확인하기 전에는 재개하지 않습니다.")
            tot += r["input_tokens"] / 1e6 * price_in + r["output_tokens"] / 1e6 * price_out
    return tot


def check_budget(price_in: float, price_out: float, budget: float) -> None:
    spent = spent_usd(price_in, price_out)
    reserve = (ESTIMATED_INPUT_RESERVE * price_in + MAX_OUTPUT_TOKENS * price_out) / 1e6
    if spent + reserve >= budget * 0.95:
        sys.exit(f"누적 추정 ${spent:.4f} + 다음 호출 예약 ${reserve:.4f}: 비용 중단 기준 도달")


def validate_prices(price_in, price_out, budget):
    if any(v is None or not math.isfinite(v) or v < 0 for v in (price_in, price_out)):
        sys.exit("--price-in/--price-out에 유한한 0 이상의 실제 단가를 입력하세요")
    if not math.isfinite(budget) or budget <= 0:
        sys.exit("--budget-usd는 유한한 양수여야 합니다")


def call(client, model: str, prompt: str, img: Path) -> tuple[str, object]:
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt},
                                            {"type": "input_image", "image_url": data_url(img), "detail": "high"}]}],
        text={"format": {"type": "json_object"}},
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return resp.output_text, resp.usage


def record_call(client, model, prompt, img, rec, path):
    # JSON 파싱 실패 전에 사용량을 먼저 기록한다. 요청 실패의 비용은 알 수 없으므로 0으로 취급하지 않는다.
    try:
        raw, usage = call(client, model, prompt, img)
        if usage is None:
            raise RuntimeError("API 응답의 사용량 없음")
        rec.update(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
        rec.update(ok=True, answer=json.loads(raw))
    except Exception as e:
        rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if rec.get("input_tokens") is None or rec.get("output_tokens") is None:
        sys.exit("사용량을 확인할 수 없어 중단했습니다. 제공자 청구 내역을 확인하세요.")
    return rec


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
    client = OpenAI(api_key=read_key(), max_retries=0)
    if a.list_models:
        ids = sorted(m.id for m in client.models.list())
        print("\n".join(ids))
        return
    if not a.model:
        sys.exit("--model이 필요하다")

    validate_prices(a.price_in, a.price_out, a.budget_usd)
    cand = pd.read_csv(VLM_DIR / "candidates.csv")
    if a.probe:
        if PROBE_RESULTS.exists() and PROBE_RESULTS.stat().st_size:
            sys.exit("이미 probe 기록이 있습니다. 반복 호출을 막기 위해 중단합니다.")
        check_budget(a.price_in, a.price_out, a.budget_usd)
        row = cand.iloc[0]
        rec = {"model": a.model, "image_id": row.image_id, "stage": "probe"}
        print(json.dumps(record_call(client, a.model, BLIND, DATA_RAW / row.image_path, rec, PROBE_RESULTS),
                         ensure_ascii=False, indent=1))
        return

    done = done_keys()
    n_calls = sum(1 for _ in open(RESULTS, encoding="utf-8")) if RESULTS.exists() else 0
    VLM_DIR.mkdir(parents=True, exist_ok=True)
    for row in cand.itertuples():
        for stage in ("blind", "explain"):
            if (row.image_id, stage) in done:
                continue
            if n_calls >= MAX_CALLS:
                sys.exit(f"호출 상한 {MAX_CALLS}회에 닿아 멈춤")
            check_budget(a.price_in, a.price_out, a.budget_usd)
            prompt = BLIND if stage == "blind" else EXPLAIN.format(
                true=row.true_name, pred=row.pred_name, result="맞힘" if row.kind == "정답" else "틀림")
            rec = {"image_id": row.image_id, "stage": stage, "kind": row.kind, "true": int(row.target_id), "pred": int(row.pred),
                   "camera": row.camera, "model": a.model, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            record_call(client, a.model, prompt, DATA_RAW / row.image_path, rec, RESULTS)
            n_calls += 1
            print(f"{n_calls:3d} {row.image_id} {stage} ok={rec['ok']} 누적 ${spent_usd(a.price_in, a.price_out):.3f}", flush=True)
    print("완료")


if __name__ == "__main__":
    main()

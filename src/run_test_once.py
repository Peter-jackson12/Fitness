"""E08: 선정 모델로 Test를 한 번만 평가한다. 결과 파일(experiments/E08/)이 있으면 거부한다.

E08은 작업 폴더의 같은 내용 스크립트로 2026-11-26에 한 번 실행했다. 이 파일은 저장소에 남기는 사본이며,
결과가 이미 있으므로 다시 실행하면 거부된다.

사용: python src/run_test_once.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import class_names  # noqa: E402
from metrics import summarize  # noqa: E402
from predict import predict_split  # noqa: E402

RUN, STRAT = "SEL-A-s43", "full"
SHA = "f21a180db8c362efc029a8cf0a5d8c651fdcef56d69ca41e82da2f12e65a1f63"
OUT = ROOT / "experiments" / "E08"
summ, pred = OUT / "test_summary.json", OUT / "pred_test.csv"
if summ.exists() or pred.exists():
    sys.exit("E08 결과가 이미 있다. Test는 한 번만 평가한다.")
w = ROOT / "models" / f"{RUN}_{STRAT}.pt"
got = hashlib.sha256(w.read_bytes()).hexdigest()
if got != SHA:
    sys.exit(f"가중치 SHA 불일치: {got}")
(OUT / ".started").open("x").write(time.strftime("%Y-%m-%d %H:%M:%S"))  # 두 번째 실행을 막는 표시
t0 = time.time()
res = predict_split(RUN, STRAT, "test", allow_test=True)
m = summarize(res["target_id"], res["pred"])
assert abs(m["macro_f1"] - res.attrs["macro_f1"]) < 1e-12
names = class_names()
m["per_class"] = {k: ({names[c]: v[c] for c in range(12)}) for k, v in m["per_class"].items()}
m.update(run_id=RUN, strategy=STRAT, weights_sha256=got, n_images=len(res), n_persons=int(res["person_id"].nunique()),
         persons=sorted(res["person_id"].unique().tolist()), time_s=round(time.time() - t0, 1),
         evaluated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
res.to_csv(pred, index=False, encoding="utf-8-sig")
summ.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: m[k] for k in ("macro_f1", "accuracy", "n_images", "n_persons", "persons", "time_s")}, ensure_ascii=False))
for c in range(12):
    n = names[c]
    print(f"{c:2d} {n:12s} P {m['per_class']['precision'][n]:.3f} R {m['per_class']['recall'][n]:.3f} F1 {m['per_class']['f1'][n]:.3f} n {m['per_class']['support'][n]}")

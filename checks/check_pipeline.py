"""2일차 파이프라인 자체 점검(학습 전). 실행: uv run python checks/check_pipeline.py

확인 항목
1. Letterbox: 224×224, 내용 영역 224×126(위아래 49px 패딩), 패딩 색, 정규화 후 패딩 ≈ 0
2. 분할: 공식 수치, 부분집합 CSV의 Val이 100%와 같음, 사람·클립 겹침 0, 기본 loader에 Test 없음
3. 변환: 기본 학습에서 train/eval 변환 결과가 같음(증강 없음), 라벨 = target_id
4. 모델: 학습 파라미터 수, 고정 BN이 eval, FE·Partial 한 스텝 뒤 고정 부분(가중치+BN 통계)이 그대로, 옵티마이저 lr 그룹
5. 평가: evaluate가 eval 모드·no_grad로 동작, Macro F1 = sklearn 직접 계산, 예측 안 된 클래스 포함
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import DATA_RAW  # noqa: E402
from dataset import PAD_FILL, FitnessDataset, Letterbox, build_transforms, load_split, make_loaders, set_reproducible, augment_preset, balanced_class_weights  # noqa: E402
import os  # noqa: E402
from metrics import evaluate, summarize  # noqa: E402
from models import build_model, count_params, param_groups, set_train_mode  # noqa: E402

ok_count = 0


def check(cond: bool, msg: str) -> None:
    global ok_count
    if not cond:
        raise AssertionError(msg)
    ok_count += 1
    print("OK ", msg)


# 1. Letterbox
tr = load_split("train")
img = Image.open(DATA_RAW / tr.loc[0, "image_path"]).convert("RGB")
lb = np.asarray(Letterbox(224)(img))
check(lb.shape == (224, 224, 3), f"letterbox 크기 {lb.shape}")
check((lb[:49] == PAD_FILL).all() and (lb[175:] == PAD_FILL).all(), "위 49px·아래 49px가 패딩 색")
check(not (lb[49:175] == PAD_FILL).all(axis=2).all(axis=1).any(), "내용 영역 126줄(49~174)에 패딩만 있는 줄이 없음")
_, eval_tf = build_transforms()
x = eval_tf(img)
check(tuple(x.shape) == (3, 224, 224), "텐서 3×224×224")
check(x[:, :49].abs().max().item() < 0.01, f"정규화 후 패딩 ≈ 0 (최대 {x[:, :49].abs().max().item():.4f})")

lb384 = np.asarray(Letterbox(384)(img))
check(lb384.shape == (384, 384, 3) and (lb384[:84] == PAD_FILL).all() and (lb384[300:] == PAD_FILL).all(),
      "letterbox 384: 원본 크기 그대로(384×216) + 위아래 84px 패딩")
check(np.array_equal(lb384[84:300], np.asarray(img)), "letterbox 384의 내용 영역 = 원본 픽셀 그대로")

# 2. 분할
va = load_split("val")
check((len(tr), len(va)) == (8144, 1440), "Train 8,144 / Val 1,440")
for pct, n in ((25, 2036), (50, 4072)):
    t = load_split("train", pct)
    check(len(t) == n and set(t["image_id"]) <= set(tr["image_id"]), f"{pct}% Train {n}장, 100%의 부분집합")
    check(load_split("val", pct)["image_id"].tolist() == va["image_id"].tolist(), f"{pct}% CSV의 Val = 100% Val")
te = load_split("test")
for col in ("person_id", "clip_id"):
    s_tr, s_va, s_te = set(tr[col]), set(va[col]), set(te[col])
    check(not (s_tr & s_va or s_tr & s_te or s_va & s_te), f"{col} 분할 간 겹침 0")
loaders = make_loaders(num_workers=0)
check(set(loaders) == {"train", "val"}, "기본 loader에 Test 없음")
check(len(make_loaders(num_workers=0, allow_test=True)["test"].dataset) == 1432, "allow_test=True일 때만 Test 1,432")

# 3. 변환·라벨
ds_tr, ds_va = loaders["train"].dataset, loaders["val"].dataset
check(torch.equal(ds_tr[0][0], eval_tf(img)), "기본 학습: train 변환 = eval 변환(증강 없음)")
check(ds_tr.targets == tr["target_id"].tolist() and ds_va.targets == va["target_id"].tolist(), "라벨 = target_id")
xb, yb = next(iter(loaders["val"]))
check(tuple(xb.shape) == (32, 3, 224, 224) and yb.dtype == torch.int64, "배치 32×3×224×224, 라벨 int64")

# 4. 모델
expected = {"scratch": 11_182_668, "fe": 6_156, "partial": 8_399_884, "full": 11_182_668}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fc_init = {}
for strategy, n in expected.items():
    m = build_model(strategy)
    check(count_params(m)["trainable"] == n, f"{strategy}: 학습 파라미터 {n:,}")
    fc_init[strategy] = m.fc.weight.detach().clone()
    groups = param_groups(m, strategy)
    lrs = {g["name"]: g["lr"] for g in groups}
    exp_lrs = {"scratch": {"all": 1e-3}, "fe": {"head": 1e-3}, "partial": {"head": 1e-3, "backbone": 1e-4},
               "full": {"head": 1e-3, "backbone": 1e-4}}[strategy]
    check(lrs == exp_lrs, f"{strategy}: lr 그룹 {lrs}")
    check(sum(p.numel() for g in groups for p in g["params"]) == n, f"{strategy}: 옵티마이저가 학습 파라미터를 모두 받음")

    set_train_mode(m)
    bns = [(name, mod) for name, mod in m.named_modules() if isinstance(mod, torch.nn.BatchNorm2d)]
    frozen_bn = [name for name, mod in bns if not mod.weight.requires_grad]
    check(all(not mod.training for name, mod in bns if name in frozen_bn), f"{strategy}: 고정 BN {len(frozen_bn)}개가 eval")
    check(all(mod.training for name, mod in bns if name not in frozen_bn), f"{strategy}: 학습 BN은 train")

    if strategy in ("fe", "partial"):
        m.to(device)
        before = {k: v.detach().clone() for k, v in m.state_dict().items()}
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        loss = torch.nn.functional.cross_entropy(m(xb.to(device)), yb.to(device))
        loss.backward()
        opt.step()
        after = m.state_dict()
        trainable_prefix = ("fc.",) if strategy == "fe" else ("layer4.", "fc.")
        frozen_keys = [k for k in before if not k.startswith(trainable_prefix)]
        changed = [k for k in frozen_keys if not torch.equal(before[k], after[k])]
        check(not changed, f"{strategy}: 한 스텝 뒤 고정 부분(가중치·BN 통계 {len(frozen_keys)}개) 그대로")
        check(not torch.equal(before["fc.weight"], after["fc.weight"]), f"{strategy}: fc는 바뀜")
        if strategy == "partial":
            check(not torch.equal(before["layer4.1.bn2.running_mean"], after["layer4.1.bn2.running_mean"]),
                  "partial: layer4 BN 통계는 갱신됨")
check(all(torch.equal(fc_init["scratch"], w) for w in fc_init.values()), "네 전략의 fc 초기값이 같음(seed 42)")

# 5. 평가
y_true = np.array([0, 1, 2, 2, 3, 11])
y_pred = np.array([0, 1, 2, 3, 3, 0])
s = summarize(y_true, y_pred)
check(abs(s["macro_f1"] - f1_score(y_true, y_pred, labels=list(range(12)), average="macro", zero_division=0)) < 1e-12,
      f"summarize Macro F1 = sklearn(12클래스 고정) {s['macro_f1']:.4f}")
check(len(s["per_class"]["f1"]) == 12 and np.array(s["confusion_matrix"]).sum() == 6, "클래스별 12개, CM 합 = 샘플 수")

m = build_model("fe").to(device)
m.train()
sub = torch.utils.data.Subset(ds_va, list(range(64)))
r = evaluate(m, torch.utils.data.DataLoader(sub, batch_size=32), device, return_outputs=True)
check(not m.training, "evaluate 뒤 모델이 eval 모드")
check(len(r["y_true"]) == 64 and r["probs"].shape == (64, 12), "evaluate 출력 모양")
check(abs(r["macro_f1"] - f1_score(r["y_true"], r["y_pred"], labels=list(range(12)), average="macro", zero_division=0)) < 1e-12,
      "evaluate Macro F1 = sklearn 직접 계산")
with torch.no_grad():
    m.eval()
    direct = m(torch.stack([sub[i][0] for i in range(64)]).to(device)).argmax(1).cpu().numpy()
check((direct == r["y_pred"]).mean() > 0.98, f"AMP 평가 예측 ≈ fp32 예측(일치율 {(direct == r['y_pred']).mean():.3f})")

# 5b. 증강(E05): Train에만 붙고 eval 변환은 그대로
aug_loaders = make_loaders(num_workers=0, train_augment=augment_preset("basic"))
check(torch.equal(aug_loaders["val"].dataset[0][0], ds_va[0][0]), "증강을 켜도 Val 변환은 증강 없음과 같음")
torch.manual_seed(0)
diffs = sum(not torch.equal(aug_loaders["train"].dataset[0][0], ds_tr[0][0]) for _ in range(10))
check(diffs >= 8, f"증강을 켜면 Train 이미지가 매번 달라짐(10회 중 {diffs}회)")
check(augment_preset("none") is None and augment_preset(None) is None, "augment none = 증강 없음")

# 5c. 클래스 가중치(E07c·d): N/(K·n_c), 학습 손실에만
w = balanced_class_weights()
cnt = tr["target_id"].value_counts().reindex(range(12)).to_numpy()
check(abs(float((w.numpy() * cnt).sum()) - len(tr)) < 1e-2, f"Σ n_c·w_c = N(8,144): 손실 크기 보존")
check(int(w.argmax()) == int(cnt.argmin()) and int(w.argmin()) == int(cnt.argmax()),
      f"가장 적은 클래스 가중치 최대({w.max():.2f}), 가장 많은 클래스 최소({w.min():.2f})")

# 6. 재현성 설정(독립검증 ② 지적 반영)
check(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "common import 때 CUBLAS_WORKSPACE_CONFIG 설정")
set_reproducible(True)
check(torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark and torch.are_deterministic_algorithms_enabled(),
      "set_reproducible(True): cuDNN deterministic, benchmark 끔, 결정적 알고리즘 강제")

print(f"\n전체 통과: {ok_count}개 항목")

"""실험 하나를 실행하고 `experiments/<run_id>/<strategy>/`에 설정·기록을 남긴다.

예) E00 시간 측정:  python src/run_experiment.py --run-id E00 --strategies scratch fe partial full --max-epochs 2 --no-save
예) E01 Scratch:    python src/run_experiment.py --run-id E01 --strategies scratch --max-epochs 40

설정(config.json)은 학습 전에 쓴다(4.3 P-hacking 방지). Test는 이 스크립트에서 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402

from common import ROOT, class_names  # noqa: E402
from dataset import augment_preset, balanced_class_weights, make_loaders, seed_everything, set_reproducible  # noqa: E402
from models import STRATEGIES, build_model, count_params  # noqa: E402
from trainer import fit  # noqa: E402


def environment() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "platform": platform.platform(),
    }


def _set_aside(path: Path, stamp: str) -> None:
    """기존 결과를 지우지 않고 `<이름>_replaced_<시각>`으로 옮긴다(실패·이전 실험도 기록으로 남긴다)."""
    if path.exists():
        path.rename(path.with_name(f"{path.stem}_replaced_{stamp}{path.suffix}"))


def run(args) -> list[dict]:
    """실행 ID 단위 잠금(`experiments/<run_id>/.lock`)을 잡고 실행한다.

    잠금 파일은 배타 생성('x')이라 원자적이다. 같은 ID로 두 실행이 겹치면 뒤의 것이 시작 전에 멈춘다
    (`--overwrite`여도). 그래서 한 실행이 쓰는 중인 폴더를 다른 실행이 옮기는 일이 없다.
    비정상 종료로 잠금이 남으면, 그 PID의 프로세스가 없는지 확인한 뒤 손으로 지운다.
    """
    out_root = ROOT / "experiments" / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)
    lock = out_root / ".lock"
    try:
        with open(lock, "x", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    except FileExistsError:
        raise SystemExit(f"실행 ID {args.run_id}가 다른 실행에 잠겨 있습니다: {lock} ({lock.read_text(encoding='utf-8').strip()}). "
                         "그 실행이 끝났는데 남은 잠금이면 프로세스가 없는지 확인하고 지우세요.")
    try:
        return _run_locked(args, out_root)
    finally:
        lock.unlink(missing_ok=True)


def _run_locked(args, out_root: Path) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_reproducible(not args.nondeterministic)
    ckpt_dir = ROOT / "models"
    # 같은 실행 ID가 있으면 시작 전에 멈춘다. 덮어쓰면 이전 결과가 사라지고, 중간에 실패하면 새 config와 옛 기록이 섞인다.
    existing = [out_root / s for s in args.strategies if (out_root / s).exists()]
    existing += [ckpt_dir / f"{args.run_id}_{s}.pt" for s in args.strategies if (ckpt_dir / f"{args.run_id}_{s}.pt").exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"이미 있는 결과: {[str(p.relative_to(ROOT)) for p in existing]}. "
                         "새 실행 ID를 쓰거나 --overwrite(기존 결과를 *_replaced_<시각>으로 보관)를 주세요.")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for p in existing:
        _set_aside(p, stamp)
    summaries = []
    for strategy in args.strategies:
        out = out_root / strategy
        out.mkdir(parents=True)
        seed_everything(args.seed)
        model = build_model(strategy, seed=args.seed)
        config = {
            "run_id": args.run_id, "strategy": strategy, "train_pct": args.train_pct, "batch_size": args.batch_size,
            "max_epochs": args.max_epochs, "patience": args.patience, "optimizer": "AdamW",
            "lr_head": args.lr_head, "lr_backbone": args.lr_backbone if strategy in ("partial", "full") else None,
            "weight_decay": args.weight_decay, "loss": "CrossEntropy", "class_weight": args.class_weight,
            "class_weights": balanced_class_weights(args.train_pct).round(decimals=4).tolist() if args.class_weight == "balanced" else None,
            "amp_fp16": not args.no_amp,
            "input": f"letterbox {args.input_size} (ImageNet 평균색 패딩) + ImageNet 정규화", "input_size": args.input_size, "augment": args.augment,
            "augment_detail": [repr(t) for t in (augment_preset(args.augment) or [])],
            "seed": args.seed, "num_workers_train": args.num_workers, "num_workers_eval": 0, "params": count_params(model),
            "notes": args.notes, "env": environment(),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        loaders = make_loaders(train_pct=args.train_pct, batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
                               size=args.input_size, train_augment=augment_preset(args.augment))
        res = fit(model, strategy, loaders, device, args.max_epochs, args.patience,
                  args.lr_head, args.lr_backbone, args.weight_decay, amp=not args.no_amp,
                  class_weights=balanced_class_weights(args.train_pct) if args.class_weight == "balanced" else None)

        pd.DataFrame(res["history"]).to_csv(out / "history.csv", index=False, encoding="utf-8")
        names = class_names()
        bv = res["best_val"]
        summary = {
            "run_id": args.run_id, "strategy": strategy, "train_pct": args.train_pct,
            "params": config["params"], "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
            "stopped_early": res["stopped_early"], "total_time_s": res["total_time_s"],
            "mean_train_epoch_s": sum(h["train_time_s"] for h in res["history"]) / res["epochs_run"],
            "peak_gpu_mem_mb": res["peak_gpu_mem_mb"],
            "val": {"loss": bv["loss"], "accuracy": bv["accuracy"], "macro_f1": bv["macro_f1"],
                    "per_class": {k: dict(zip([names[i] for i in range(len(v))], v)) for k, v in bv["per_class"].items()},
                    "confusion_matrix": bv["confusion_matrix"]},
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.no_save:
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(res["best_state"], ckpt_dir / f"{args.run_id}_{strategy}.pt")
        summaries.append(summary)
        del model, loaders, res
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return summaries


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--strategies", nargs="+", choices=STRATEGIES, required=True)
    p.add_argument("--train-pct", type=int, default=100, choices=(25, 50, 100))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, required=True)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2, help="Train loader worker 수(Val은 0)")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--nondeterministic", action="store_true",
                   help="cuDNN benchmark 모드(약 14%% 빠름, 실행마다 결과가 달라짐). 기본은 결정적 모드")
    p.add_argument("--overwrite", action="store_true", help="같은 실행 ID의 기존 결과를 *_replaced_<시각>으로 옮기고 다시 실행")
    p.add_argument("--no-save", action="store_true", help="가중치 파일을 저장하지 않는다")
    p.add_argument("--input-size", type=int, default=224, help="letterbox 정사각형 한 변(기본 224, E07은 384)")
    p.add_argument("--class-weight", default="none", choices=("none", "balanced"),
                   help="학습 손실의 클래스 가중치(balanced: N/(K·n_c)). Val 손실에는 쓰지 않음")
    p.add_argument("--augment", default="none", choices=("none", "basic"), help="Train 전용 증강(dataset.augment_preset)")
    p.add_argument("--notes", default="")
    return p.parse_args(argv)


if __name__ == "__main__":
    for s in run(parse_args()):
        print(json.dumps({k: s[k] for k in ("strategy", "best_epoch", "epochs_run", "total_time_s",
                                            "mean_train_epoch_s", "peak_gpu_mem_mb")}, ensure_ascii=False),
              f"val_macro_f1={s['val']['macro_f1']:.4f}")

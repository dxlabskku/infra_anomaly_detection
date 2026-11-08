#!/usr/bin/env python
"""S2 CARE-YOLO 학습. --exp base_gray3 | G | GCL | GC | GL  (config experiments)

YOLOv8s(COCO 사전학습)에서 시작해 train 으로만 학습한다. best.pt 선택(Ultralytics fitness = val mAP50-95)에는
서브셋 val 4,003장만 쓴다(prepare.detector_val_source). test 는 보지 않는다.

실험마다 바뀌는 것
  mode                 gray3 (흑백 복제) | phys3 (원본·상대 진폭·깊이 보정 진폭)
  context              [] 이면 GPR 단독 YOLO. [clim, loc] 이면 상황 인코더 + FiLM (care_model.py)
  site_balance_power   train 표집 가중치 = (현장 이미지 수)^-power. 0 이면 Ultralytics 기본 무작위 섞기
학습 중 각 보조 모달을 care.modal_dropout 확률로 결측 처리한다(ModDrop). 검증 때는 지우지 않는다.

출력
  detectors/{exp}/                 Ultralytics 런 폴더 (weights/best.pt, results.csv, args.yaml, 그림)
  detectors/{exp}/train_info.json  실험 설정, 사용한 인자, val 지표(best.pt), 소요 시간, 파라미터 수, FiLM 크기
  detectors/{exp}/epoch_timing.csv epoch 마다 학습 초, 검증+저장 초, epoch 초, 누적 초, FiLM 크기
"""
import csv
import os
import shutil
import sys
import time
from collections import Counter
from copy import copy
from pathlib import Path

import common as C


class BatchReduced(RuntimeError):
    """Ultralytics 가 첫 epoch OOM 에서 batch 를 스스로 절반으로 줄였다."""


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, BatchReduced) or "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def _batch_guard(requested: int):
    """Ultralytics 8.4 는 첫 epoch OOM 이면 batch 를 경고만 남기고 절반씩(최대 3번) 줄인다.
    사양(32 -> 16, 기록)대로 하려고 바뀌는 즉시 멈추고, 바깥에서 batch 16 으로 처음부터 다시 학습한다."""
    def _cb(trainer):
        if int(trainer.batch_size) != requested:
            raise BatchReduced(f"Ultralytics 가 batch 를 {requested} -> {trainer.batch_size} 로 줄였습니다 (GPU 메모리 부족)")
    return _cb


class EpochTimer:
    """Ultralytics 콜백으로 epoch 별 시간을 초 단위로 잰다 (time.perf_counter).

    Ultralytics 8.4.166 의 호출 순서: on_pretrain_routine_start -> on_train_start ->
      [on_train_epoch_start -> 배치 학습 -> on_train_epoch_end -> 검증·저장 -> on_fit_epoch_end] x epoch ->
      final_eval (best.pt 로 val, 안에서 on_fit_epoch_end 한 번 더) -> on_train_end
    """

    COLS = ["epoch", "train_sec", "val_save_sec", "epoch_sec", "cumulative_sec", "val_mAP50", "val_mAP50_95",
            "film_mean_abs"]

    def __init__(self, csv_path: Path, log):
        self.csv_path, self.log = Path(csv_path), log
        self.reset()

    def reset(self):
        self.t_routine = self.t_train_start = self.t_ep = self.t_tr_end = self.t_last_fit = None
        self.rows, self.in_epoch, self.setup_sec, self.final_eval_sec, self.total_sec = [], False, None, None, None

    def attach(self, model):
        for ev in ("on_pretrain_routine_start", "on_train_start", "on_train_epoch_start", "on_train_epoch_end",
                   "on_fit_epoch_end", "on_train_end"):
            model.add_callback(ev, getattr(self, ev))

    def on_pretrain_routine_start(self, trainer):
        self.reset()
        self.t_routine = time.perf_counter()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(self.COLS) + "\n")

    def on_train_start(self, trainer):
        self.t_train_start = time.perf_counter()
        self.setup_sec = self.t_train_start - self.t_routine

    def on_train_epoch_start(self, trainer):
        self.t_ep, self.in_epoch = time.perf_counter(), True

    def on_train_epoch_end(self, trainer):
        self.t_tr_end = time.perf_counter()

    def on_fit_epoch_end(self, trainer):
        now = time.perf_counter()
        if not self.in_epoch:          # final_eval 안에서 다시 불린 경우
            return
        self.in_epoch = False
        m = getattr(trainer, "metrics", {}) or {}
        import care_model as CM
        film = CM.film_magnitude(trainer.model)
        row = {"epoch": int(trainer.epoch) + 1, "train_sec": self.t_tr_end - self.t_ep,
               "val_save_sec": now - self.t_tr_end, "epoch_sec": now - self.t_ep,
               "cumulative_sec": now - self.t_train_start,
               "val_mAP50": m.get("metrics/mAP50(B)"), "val_mAP50_95": m.get("metrics/mAP50-95(B)"),
               "film_mean_abs": (sum(film.values()) / len(film)) if film else None}
        self.rows.append(row)
        self.t_last_fit = now
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(",".join("" if row[c] is None else (f"{row[c]:.6g}" if isinstance(row[c], float) else str(row[c]))
                             for c in self.COLS) + "\n")
        self.log.info("epoch %d/%d  학습 %.1fs  검증+저장 %.1fs  epoch %.1fs  누적 %.1fs  val mAP50 %s%s",
                      row["epoch"], trainer.epochs, row["train_sec"], row["val_save_sec"], row["epoch_sec"],
                      row["cumulative_sec"], "—" if row["val_mAP50"] is None else f"{row['val_mAP50']:.4f}",
                      "" if row["film_mean_abs"] is None else f"  FiLM |w| {row['film_mean_abs']:.2e}")

    def on_train_end(self, trainer):
        now = time.perf_counter()
        self.final_eval_sec = now - (self.t_last_fit or now)
        self.total_sec = now - self.t_routine

    def summary(self) -> dict:
        tr = [r["train_sec"] for r in self.rows]
        va = [r["val_save_sec"] for r in self.rows]
        return {"epochs": len(self.rows), "setup_sec": self.setup_sec,
                "train_sec_total": sum(tr), "val_save_sec_total": sum(va),
                "train_sec_per_epoch_mean": sum(tr) / len(tr) if tr else None,
                "val_save_sec_per_epoch_mean": sum(va) / len(va) if va else None,
                "epoch_sec_mean": (sum(tr) + sum(va)) / len(tr) if tr else None,
                "final_eval_sec": self.final_eval_sec, "total_sec": self.total_sec,
                "csv": str(self.csv_path), "clock": "time.perf_counter",
                "note": "val_save_sec = epoch 끝의 검증 + 체크포인트 저장. final_eval_sec = 학습 뒤 best.pt 로 val 재평가"}


def _read_results_csv(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{k.strip(): v.strip() for k, v in r.items()} for r in rows]


def _metric(d: dict, *names):
    for n in names:
        if n in d and d[n] not in ("", None):
            return float(d[n])
    return None


def site_weights(sites, power: float):
    """현장 균형 표집 가중치 (현장 이미지 수)^-power 와 요약."""
    cnt = Counter(sites)
    w = [cnt[s] ** -power for s in sites]
    tot = sum(w)
    share = Counter()
    for s, wi in zip(sites, w):
        share[s] += wi / tot
    return w, {"n_sites": len(cnt), "power": power,
               "largest_site_share_before": max(cnt.values()) / len(sites), "largest_site_share_after": max(share.values())}


def make_trainer(exp: dict, cfg: dict, table, site_of: dict, log):
    """실험 설정을 담은 Ultralytics 학습기 클래스. ultralytics import 뒤에 부른다."""
    import torch
    from torch.utils.data import WeightedRandomSampler
    from ultralytics.data.build import InfiniteDataLoader, seed_worker
    from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

    import care_model as CM

    mods = exp["context"]
    care_cfg = cfg["care"]
    loc_cards = [len(cfg["location"]["vocab"][c]) + 1 for c in cfg["location"]["columns"]]
    drop_p = float(care_cfg["modal_dropout"])
    power = exp["site_balance_power"]
    seed = int(cfg["seed"])
    state = {"sampler": None}

    class CareValidator(DetectionValidator):
        def preprocess(self, batch):
            batch = super().preprocess(batch)
            if mods:
                CM.CTX.set(*CM.context_batch(table, batch["im_file"], mods, batch["img"].device))
            return batch

    class CareTrainer(DetectionTrainer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._care_gen = torch.Generator().manual_seed(seed)

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            return CM.make_care_model(model, mods, care_cfg, loc_cards)

        def get_validator(self):
            return CareValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args),
                                 _callbacks=self.callbacks)

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            if mods:
                CM.CTX.set(*CM.context_batch(table, batch["im_file"], mods, batch["img"].device, drop_p,
                                             self._care_gen))
            return batch

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode != "train" or power <= 0:
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            dataset = self.build_dataset(dataset_path, mode, batch_size)
            w, summ = site_weights([site_of[Path(f).name] for f in dataset.im_files], power)
            state["sampler"] = summ
            log.info("현장 균형 표집: 현장 %d곳, power %.2f, 가장 큰 현장 비중 %.3f -> %.3f", summ["n_sites"], power,
                     summ["largest_site_share_before"], summ["largest_site_share_after"])
            sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(w), replacement=True,
                                            generator=torch.Generator().manual_seed(seed))
            nw = min(os.cpu_count() // max(torch.cuda.device_count(), 1), int(self.args.workers))
            return InfiniteDataLoader(
                dataset=dataset, batch_size=min(batch_size, len(dataset)), shuffle=False, num_workers=nw,
                sampler=sampler, prefetch_factor=4 if nw > 0 else None, pin_memory=True,
                collate_fn=getattr(dataset, "collate_fn", None), worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(seed + 1), drop_last=False)

    def check_context(trainer):
        if not mods:
            return
        if CM.CTX.used == 0:
            raise C.PipelineError("이번 epoch 에 상황 값이 한 번도 쓰이지 않았습니다 (CTX 연결 오류).")
        log.info("상황 값 사용 %d회, 크기 달라 건너뜀 %d회, FiLM |w| %s", CM.CTX.used, CM.CTX.skipped,
                 {k: f"{v:.2e}" for k, v in CM.film_magnitude(trainer.model).items()})
        CM.CTX.used = CM.CTX.skipped = 0

    return CareTrainer, check_context, state


def main():
    ap = C.base_parser(__doc__)
    ap.add_argument("--exp", required=True, help="config experiments 이름 (base_gray3, G, GCL, ...)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--device", help="detector.device 덮어쓰기 (예: 0)")
    args = ap.parse_args()
    cfg = C.load_config(args, [
        ("detector.train_args.epochs", args.epochs), ("detector.train_args.batch", args.batch),
        ("detector.train_args.workers", args.workers), ("detector.device", args.device),
    ])
    paths = C.Paths(cfg)
    exp = C.experiment(cfg, args.exp)
    log, log_path = C.setup_logging(f"s2_train_{args.exp}", paths)
    seed = int(cfg["seed"])
    C.set_seed(seed)
    log.info("실험 %s: %s", args.exp, exp)
    log.info("설정 덮어쓰기: %s", cfg["_overrides"])

    data_yaml = paths.prepared_mode(exp["mode"]) / "data.yaml"
    if not data_yaml.exists():
        raise C.PipelineError(f"{data_yaml} 이 없습니다. s1_prepare.py 를 먼저 실행하세요.")
    run_dir = paths.det_dir(args.exp)
    # Ultralytics 는 exist_ok=False 이면 이름 뒤에 번호를 붙여 새 폴더를 만든다. 그러지 않게 여기서 먼저 멈춘다.
    C.prepare_output(run_dir, args.overwrite, paths, make_dir=False)
    paths.detectors.mkdir(parents=True, exist_ok=True)

    C.setup_ultralytics_env(paths)
    import torch
    import ultralytics
    from ultralytics import YOLO
    C.configure_ultralytics_settings(paths)
    C.attach_ultralytics_log()
    import care_model as CM
    C.set_seed(seed)

    df = C.load_manifest(cfg, paths)
    site_of = dict(zip(df["filename"], df["site_id"]))
    table = CM.ContextTable(paths.climate, paths.location, cfg["data"]["splits"],
                            bool(cfg["context"]["unseen_to_missing"]))
    if exp["context"]:
        log.info("상황 표: %d장. train 에 없던 위치 값 -> 결측 %s", len(table.row), table.unseen)
    Trainer, check_context, tstate = make_trainer(exp, cfg, table, site_of, log)

    weights = Path(cfg["detector"]["weights"])
    # AMP 점검용 소형 모델 등 Ultralytics 가 cwd 에 받는 파일이 작업 폴더 안에 남도록 한다
    C.ensure_inside_work(paths.weights_run, paths)
    paths.weights_run.mkdir(parents=True, exist_ok=True)
    os.chdir(paths.weights_run)

    train_args = dict(cfg["detector"]["train_args"])
    train_args.update(data=str(data_yaml), project=str(paths.detectors), name=args.exp,
                      device=cfg["detector"]["device"])
    log.info("Ultralytics %s, torch %s, 시작 가중치 %s", ultralytics.__version__, torch.__version__, weights)
    log.info("학습 인자: %s", train_args)

    notes = []
    batch_used = int(train_args["batch"])
    t0 = time.time()
    timer = EpochTimer(run_dir / "epoch_timing.csv", log)
    while True:
        model = YOLO(str(weights))
        model.add_callback("on_train_batch_end", _batch_guard(batch_used))
        model.add_callback("on_train_epoch_end", check_context)
        timer.attach(model)
        try:
            results = model.train(trainer=Trainer, **train_args)
            break
        except Exception as e:  # noqa: BLE001
            fb = int(cfg["detector"]["oom_fallback_batch"])
            if _is_oom(e) and batch_used <= fb:
                raise C.PipelineError(f"batch={batch_used} 에서도 GPU 메모리가 부족합니다. 사양상 더 줄이지 않고 멈춥니다: {e}")
            if _is_oom(e) and batch_used > fb:
                log.warning("[보고] batch=%d 에서 GPU 메모리 부족 -> batch=%d 로 다시 학습합니다 (%s)", batch_used, fb, e)
                notes.append(f"batch {batch_used} 메모리 부족으로 {fb} 로 낮춤")
                del model
                torch.cuda.empty_cache()
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                batch_used = fb
                train_args["batch"] = fb
                t0 = time.time()
                continue
            raise
    t_train = time.time() - t0

    trainer = model.trainer
    save_dir = Path(trainer.save_dir)
    if save_dir.resolve() != run_dir.resolve():
        raise C.PipelineError(f"Ultralytics 저장 폴더가 예상과 다릅니다: {save_dir} (예상 {run_dir})")
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise C.PipelineError(f"best.pt 가 없습니다: {best}")

    # val 지표: 학습 끝에 Ultralytics 가 best.pt 로 val(서브셋 4,003장)을 다시 평가한 값
    rd = getattr(results, "results_dict", None) or dict(getattr(trainer, "metrics", {}) or {})
    val_metrics = {
        "mAP50": _metric(rd, "metrics/mAP50(B)"),
        "mAP50-95": _metric(rd, "metrics/mAP50-95(B)"),
        "precision": _metric(rd, "metrics/precision(B)"),
        "recall": _metric(rd, "metrics/recall(B)"),
        "fitness": _metric(rd, "fitness"),
        "source": "Ultralytics 최종 검증 (best.pt, 검출기 선택용 val, Ultralytics 기본 NMS)",
    }
    rows = _read_results_csv(save_dir / "results.csv")
    best_row = max(rows, key=lambda r: _metric(r, "metrics/mAP50-95(B)") or 0) if rows else None

    best_model = YOLO(str(best)).model
    n_params = int(sum(p.numel() for p in best_model.parameters()))
    info = {
        "exp": args.exp, "experiment": exp, "mode": exp["mode"], "data_yaml": str(data_yaml),
        "init_weights": str(weights), "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "train_args": train_args, "batch_used": batch_used, "batch_trainer_final": int(trainer.batch_size),
        "best_pt": str(best), "args_yaml": str(save_dir / "args.yaml"), "val_metrics": val_metrics,
        "epochs_run": len(rows), "best_epoch_by_results_csv": int(float(best_row["epoch"])) if best_row else None,
        "results_csv_best_row": best_row, "n_params": n_params, "nc": int(getattr(best_model, "nc", -1)),
        "care": getattr(best_model, "care_info", None), "film_magnitude_best": CM.film_magnitude(best_model),
        "modal_dropout": float(cfg["care"]["modal_dropout"]) if exp["context"] else None,
        "context_unseen_to_missing": table.unseen if exp["context"] else None,
        "site_sampler": tstate["sampler"], "epoch_timing": timer.summary(),
        "time_sec": {"train_total": t_train, "per_epoch_mean": (t_train / len(rows)) if rows else None},
        "notes": notes, "log": str(log_path), "smoke": cfg["_smoke"],
    }
    C.write_json(save_dir / "train_info.json", info)
    log.info("완료: %s  val mAP50=%s  mAP50-95=%s  파라미터 %d (상황 %s)  학습 %.1fs", args.exp,
             val_metrics["mAP50"], val_metrics["mAP50-95"], n_params,
             (info["care"] or {}).get("n_params_context", 0), t_train)
    es = info["epoch_timing"]
    if es["epochs"]:
        log.info("시간 요약: 준비 %.1fs, epoch %d개 (학습 평균 %.1fs, 검증+저장 평균 %.1fs, epoch 평균 %.1fs), "
                 "학습 합 %.1fs, 검증+저장 합 %.1fs, 최종 평가 %.1fs, 전체 %.1fs  -> %s",
                 es["setup_sec"] or 0, es["epochs"], es["train_sec_per_epoch_mean"], es["val_save_sec_per_epoch_mean"],
                 es["epoch_sec_mean"], es["train_sec_total"], es["val_save_sec_total"], es["final_eval_sec"] or 0,
                 es["total_sec"] or 0, es["csv"])


if __name__ == "__main__":
    sys.exit(main())

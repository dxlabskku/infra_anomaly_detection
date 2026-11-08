#!/usr/bin/env python
"""S3 예측. --exp 실험 이름 (여러 개 가능, 기본: config experiments 전부 중 best.pt 가 있는 것)

학습된 best.pt 로 val·test 를 예측해 박스를 저장한다 (NMS conf 0.001, IoU 0.5, 최대 100개). 지표는 s4 에서 계산한다.
상황(기후·위치)을 쓰는 실험은 같은 모델에 상황만 바꿔 다시 돌린다 (재학습 없음, predict.context_tests).
  normal                    그대로
  missC / missL / missCL    기후 / 위치 / 둘 다 결측
  permC / permL             기후 / 위치를 같은 분할 안에서 다른 이미지의 값으로 섞음 (순열 검정)

출력 preds/{exp}/preds_{split}_{cond}.csv (filename, x1, y1, x2, y2, score, cls), predict_info.json
"""
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

import common as C

COND_MISSING = {"normal": (), "missC": ("clim",), "missL": ("loc",), "missCL": ("clim", "loc"),
                "permC": (), "permL": ()}
COND_PERM = {"permC": "clim", "permL": "loc"}


class PreparedImages(torch.utils.data.Dataset):
    """Ultralytics 학습과 같은 읽기 규칙: cv2.imread(BGR) -> [..., ::-1] (RGB) -> CHW float /255."""

    def __init__(self, files):
        self.files = list(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(self.files[i])
        if img is None or img.shape != (512, 512, 3) or img.dtype != np.uint8:
            raise RuntimeError(f"준비된 이미지가 (512, 512, 3) uint8 이 아닙니다: {self.files[i]}")
        x = torch.from_numpy(img[..., ::-1].transpose(2, 0, 1).copy()).float() / 255.0
        return x, i


def _nms_fn():
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression
    return non_max_suppression


def permutation(names, seed: int) -> dict:
    """filename -> 같은 분할의 다른 filename (자기 자신으로 가는 비율을 줄이려고 한 칸 민 무작위 순열)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    shifted = np.roll(order, 1)
    return {names[a]: names[b] for a, b in zip(order, shifted)}


def main():
    ap = C.base_parser(__doc__)
    ap.add_argument("--exp", nargs="+", help="실험 이름. 주지 않으면 best.pt 가 있는 실험 전부")
    ap.add_argument("--batch", type=int)
    args = ap.parse_args()
    cfg = C.load_config(args, [("predict.batch", args.batch)])
    paths = C.Paths(cfg)
    log, log_path = C.setup_logging("s3_predict", paths)
    C.set_seed(int(cfg["seed"]))
    pc = cfg["predict"]
    names = args.exp or [e for e in cfg["experiments"] if (paths.det_dir(e) / "weights" / "best.pt").exists()]
    if not names:
        raise C.PipelineError("예측할 실험이 없습니다. s2_train.py 를 먼저 실행하세요.")

    C.setup_ultralytics_env(paths)
    from ultralytics import YOLO
    C.configure_ultralytics_settings(paths)
    C.attach_ultralytics_log()
    import care_model as CM
    nms = _nms_fn()
    if not torch.cuda.is_available():
        raise C.PipelineError("CUDA 를 쓸 수 없습니다.")
    nms_kw = dict(conf_thres=float(pc["nms"]["conf_thres"]), iou_thres=float(pc["nms"]["iou_thres"]),
                  max_det=int(pc["nms"]["max_det"]))
    df = C.load_manifest(cfg, paths)
    table = CM.ContextTable(paths.climate, paths.location, cfg["data"]["splits"],
                            bool(cfg["context"]["unseen_to_missing"]))

    for exp_name in names:
        exp = C.experiment(cfg, exp_name)
        best = paths.det_dir(exp_name) / "weights" / "best.pt"
        if not best.exists():
            raise C.PipelineError(f"{best} 가 없습니다.")
        out_dir = C.prepare_output(paths.pred_dir(exp_name), args.overwrite, paths)
        net = YOLO(str(best)).model.float().eval().cuda()
        for p in net.parameters():
            p.requires_grad_(False)
        mods = exp["context"]
        if mods and not isinstance(net, CM.CareDetectionModel):
            raise C.PipelineError(f"{exp_name} 은 상황을 쓰는 실험인데 체크포인트가 CARE-YOLO 가 아닙니다: {type(net)}")
        conds = list(pc["context_tests"]) if mods else ["normal"]
        log.info("%s: 모드 %s, 상황 %s, 조건 %s, 체크포인트 %s", exp_name, exp["mode"], mods or "없음", conds, best)
        info = {"exp": exp_name, "experiment": exp, "best_pt": str(best), "conditions": conds, "nms": nms_kw,
                "splits": {}}
        for split in ("val", "test"):
            part = C.split_frame(df, split)
            fnames = list(part["filename"])
            perms = {c: {COND_PERM[c]: permutation(fnames, int(pc["permutation_seed"]) + k)}
                     for k, c in enumerate(conds) if c in COND_PERM}
            files = [str(paths.prepared_image(exp["mode"], split, f)) for f in fnames]
            loader = torch.utils.data.DataLoader(PreparedImages(files), batch_size=int(pc["batch"]), shuffle=False,
                                                 num_workers=int(pc["num_workers"]), pin_memory=True)
            rows = {c: [] for c in conds}
            CM.CTX.used = CM.CTX.skipped = 0
            t0 = time.time()
            with torch.no_grad():
                for x, idx in loader:
                    x = x.cuda(non_blocking=True)
                    bn = [fnames[i] for i in idx.numpy()]
                    for c in conds:
                        if mods:
                            CM.CTX.set(*CM.context_batch(table, bn, mods, x.device, force_missing=COND_MISSING[c],
                                                         perm=perms.get(c)))
                        out = net(x)
                        CM.CTX.clear()
                        dets = nms(out[0] if isinstance(out, (list, tuple)) else out, **nms_kw, max_time_img=1e6)
                        for b, d in enumerate(dets):
                            for r in d.float().cpu().numpy():
                                rows[c].append((bn[b], *(float(v) for v in r[:5]), int(round(float(r[5])))))
            if mods and (CM.CTX.skipped or CM.CTX.used != len(loader) * len(conds)):
                raise C.PipelineError(f"상황 값 연결 이상: 사용 {CM.CTX.used}, 건너뜀 {CM.CTX.skipped}")
            for c in conds:
                pd.DataFrame(rows[c], columns=["filename", "x1", "y1", "x2", "y2", "score", "cls"]).to_csv(
                    out_dir / f"preds_{split}_{c}.csv", index=False, encoding="utf-8")
            info["splits"][split] = {"n_images": len(fnames), "sec": time.time() - t0,
                                     "n_boxes": {c: len(rows[c]) for c in conds}}
            log.info("  %s: %d장, 조건 %d개, %.1fs, 박스 %s", split, len(fnames), len(conds), time.time() - t0,
                     info["splits"][split]["n_boxes"])
        info["log"] = str(log_path)
        C.write_json(out_dir / "predict_info.json", info)
        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())

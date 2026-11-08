"""평가 지표 (example_03/code/s5_evaluate.py 에서 그대로 가져옴): GT 읽기, pycocotools AP, 박스 F1, 이미지 이진 지표, 현장 부트스트랩."""
import contextlib
import io
import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import common as C
from s1_prepare import center_to_xyxy, clip_xyxy, json_boxes


# --------------------------------------------------------------------------- GT

def load_gt(paths: C.Paths, part: pd.DataFrame, split: str, size: int = 512, resized: dict | None = None) -> dict:
    """filename -> (k, 5) [cls, x1, y1, x2, y2]. 중심 -> xyxy -> [0, 512] 자르기 (작은 박스도 버리지 않음)."""
    resized = resized or {}
    gt = {}
    for fn in part["filename"]:
        lab = C.read_json(paths.src_label(split, fn))
        h0, w0 = resized.get(fn, (size, size))
        sx, sy = size / w0, size / h0
        rows = []
        for cls, x, y, w, h in json_boxes(lab):
            b = clip_xyxy(center_to_xyxy(x * sx, y * sy, w * sx, h * sy), size)
            rows.append((cls, *b))
        gt[fn] = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
    return gt


def resized_map(paths: C.Paths, split: str) -> dict:
    p = paths.prepared / "prepare_info.json"
    if not p.exists():
        return {}
    return {r["filename"]: (r["h"], r["w"]) for r in C.read_json(p).get("resized", []) if r["split"] == split}


def preds_by_image(preds: pd.DataFrame, filenames) -> dict:
    """filename -> (n, 6) [x1, y1, x2, y2, score, cls]."""
    out = {fn: np.zeros((0, 6)) for fn in filenames}
    if len(preds):
        for fn, g in preds.groupby("filename", sort=False):
            out[fn] = g[["x1", "y1", "x2", "y2", "score", "cls"]].to_numpy(dtype=np.float64)
    return out


# --------------------------------------------------------------------------- COCO 지표

CATEGORIES = [{"id": 1, "name": "rebar"}, {"id": 2, "name": "rebar_abnormal"}, {"id": 3, "name": "crack"}]


def build_coco(gt: dict, preds: dict, filenames, min_score: float = 0.001):
    from pycocotools.coco import COCO
    img_id = {fn: i + 1 for i, fn in enumerate(filenames)}
    images = [{"id": img_id[fn], "file_name": fn, "width": 512, "height": 512} for fn in filenames]
    anns, dets = [], []
    for fn in filenames:
        for cls, x1, y1, x2, y2 in gt[fn]:
            w, h = x2 - x1, y2 - y1
            anns.append({"id": len(anns) + 1, "image_id": img_id[fn], "category_id": int(cls) + 1,
                         "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0})
        for x1, y1, x2, y2, s, c in preds[fn]:
            if s >= min_score:
                dets.append({"image_id": img_id[fn], "category_id": int(round(c)) + 1,
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns, "categories": CATEGORIES}
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco, dets, img_id


def coco_metrics(coco, dets: list, img_ids: list, cat_ids: list) -> dict:
    """precision [T, R, K, A, M] 에서 area=all, maxDets=100. 클래스별 AP50·AP75·AP50:95 와 평균."""
    from pycocotools.cocoeval import COCOeval
    if not img_ids:  # pycocotools 는 빈 imgIds 를 '전체'로 해석하므로 따로 처리한다
        nan = {"AP50": math.nan, "AP75": math.nan, "AP50_95": math.nan, "n_gt": 0}
        return {"mAP50": math.nan, "mAP75": math.nan, "mAP50_95": math.nan, "n_images": 0, "n_dets": 0,
                "per_class": {CATEGORIES[c - 1]["name"]: dict(nan) for c in cat_ids}}
    sel = set(img_ids)
    sub = [d for d in dets if d["image_id"] in sel and d["category_id"] in cat_ids]
    n_gt = {c: len(coco.getAnnIds(imgIds=img_ids, catIds=[c], iscrowd=None)) for c in cat_ids}
    per = {}
    if not sub:  # 예측이 하나도 없으면 GT 가 있는 클래스의 AP 는 0
        for c in cat_ids:
            v = 0.0 if n_gt[c] > 0 else math.nan
            per[c] = {"AP50": v, "AP75": v, "AP50_95": v, "n_gt": n_gt[c]}
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            dt = coco.loadRes(sub)
            E = COCOeval(coco, dt, "bbox")
            E.params.imgIds = list(img_ids)
            E.params.catIds = list(cat_ids)
            E.evaluate()
            E.accumulate()
        P = E.eval["precision"]
        a = E.params.areaRngLbl.index("all")
        m = list(E.params.maxDets).index(100)
        t50 = int(np.argmin(np.abs(E.params.iouThrs - 0.5)))
        t75 = int(np.argmin(np.abs(E.params.iouThrs - 0.75)))
        for k, c in enumerate(E.params.catIds):
            def ap(t):
                p = P[t, :, k, a, m] if t is not None else P[:, :, k, a, m]
                return float(p[p > -1].mean()) if (p > -1).any() else math.nan
            per[c] = {"AP50": ap(t50), "AP75": ap(t75), "AP50_95": ap(None), "n_gt": n_gt[c]}

    def mean(key):
        v = [per[c][key] for c in cat_ids if not math.isnan(per[c][key])]
        return float(np.mean(v)) if v else math.nan

    return {"mAP50": mean("AP50"), "mAP75": mean("AP75"), "mAP50_95": mean("AP50_95"),
            "per_class": {CATEGORIES[c - 1]["name"]: per[c] for c in cat_ids}, "n_images": len(img_ids),
            "n_dets": len(sub)}


# --------------------------------------------------------------------------- 박스 F1

def iou_one_to_many(b, B):
    if len(B) == 0:
        return np.zeros(0)
    x1 = np.maximum(b[0], B[:, 0])
    y1 = np.maximum(b[1], B[:, 1])
    x2 = np.minimum(b[2], B[:, 2])
    y2 = np.minimum(b[3], B[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ua = (b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter
    return inter / np.maximum(ua, 1e-12)


def box_f1(gt: dict, preds: dict, filenames, score_thr: float = 0.5, iou_thr: float = 0.5, n_cls: int = 3) -> dict:
    """클래스별로 score 내림차순 그리디 매칭. 예측마다 아직 짝이 없는 같은 클래스 GT 중 IoU 최대(>= iou_thr)와 짝짓는다."""
    out = {}
    for c in range(n_cls):
        tp = fp = fn = 0
        for f in filenames:
            g = gt[f][gt[f][:, 0] == c][:, 1:5]
            p = preds[f]
            p = p[(np.round(p[:, 5]) == c) & (p[:, 4] >= score_thr)]
            p = p[np.argsort(-p[:, 4], kind="stable")]
            used = np.zeros(len(g), dtype=bool)
            for row in p:
                ious = iou_one_to_many(row[:4], g)
                if len(ious):
                    ious = np.where(used, -1.0, ious)
                    j = int(np.argmax(ious))
                    if ious[j] >= iou_thr:
                        used[j] = True
                        tp += 1
                        continue
                fp += 1
            fn += int((~used).sum())
        out[CATEGORIES[c]["name"]] = _prf(tp, fp, fn)
    tot = {k: sum(out[CATEGORIES[c]["name"]][k] for c in range(n_cls)) for k in ("tp", "fp", "fn")}
    out["all_micro"] = _prf(tot["tp"], tot["fp"], tot["fn"])
    return out


def _prf(tp, fp, fn):
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": tp / (tp + fp) if tp + fp else math.nan,
            "recall": tp / (tp + fn) if tp + fn else math.nan,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else math.nan}


def evaluate_boxes(gt: dict, preds: dict, idx: pd.DataFrame, cfg: dict) -> dict:
    fns = list(idx["filename"])
    coco, dets, img_id = build_coco(gt, preds, fns, float(cfg["predict"]["nms"]["conf_thres"]))
    all_ids = [img_id[f] for f in fns]
    res = {"overall": coco_metrics(coco, dets, all_ids, [1, 2, 3])}
    excl = set(cfg["evaluate"]["crack_excluded_facilities"])
    non_excl = [img_id[f] for f, fac in zip(idx["filename"], idx["facility"]) if fac not in excl]
    res["crack_excluding_" + "_".join(cfg["data"]["facility_en"][f] for f in excl)] = coco_metrics(
        coco, dets, non_excl, [3])
    res["per_facility"] = {}
    for fac in cfg["data"]["facilities"]:
        ids = [img_id[f] for f, ff in zip(idx["filename"], idx["facility"]) if ff == fac]
        cats = [1, 2] if fac in excl else [1, 2, 3]   # 크랙 AP 는 교량을 빼고 계산
        r = coco_metrics(coco, dets, ids, cats)
        r["crack_excluded"] = fac in excl
        res["per_facility"][fac] = r
    bf = cfg["evaluate"]["box_f1"]
    res["box_f1"] = box_f1(gt, preds, fns, bf["score_thr"], bf["iou_thr"])
    res["box_f1_per_facility"] = {
        fac: box_f1(gt, preds, list(idx.loc[idx["facility"] == fac, "filename"]), bf["score_thr"], bf["iou_thr"])
        for fac in cfg["data"]["facilities"]}
    n_small = sum(int(((g[:, 3] - g[:, 1] < 2) | (g[:, 4] - g[:, 2] < 2)).sum()) for g in gt.values())
    res["gt_boxes"] = int(sum(len(g) for g in gt.values()))
    res["gt_boxes_smaller_than_2px_kept"] = n_small
    res["n_dets_total"] = len(dets)
    return res


# --------------------------------------------------------------------------- 이미지 지표

def binary_metrics(y, score=None, pred=None) -> dict:
    y = np.asarray(y).astype(bool)
    n, npos = int(len(y)), int(y.sum())
    out = {"n": n, "n_pos": npos, "n_neg": n - npos,
           "majority_accuracy_all_good": (n - npos) / n if n else math.nan}
    if score is not None:
        s = np.asarray(score, dtype=np.float64)
        ok = 0 < npos < n
        out["auroc"] = float(roc_auc_score(y, s)) if ok else math.nan
        out["ap"] = float(average_precision_score(y, s)) if ok else math.nan
    if pred is not None:
        p = np.asarray(pred).astype(bool)
        tp = int((p & y).sum())
        fp = int((p & ~y).sum())
        fn = int((~p & y).sum())
        tn = int((~p & ~y).sum())
        rec = tp / (tp + fn) if tp + fn else math.nan
        spec = tn / (tn + fp) if tn + fp else math.nan
        out.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": tp / (tp + fp) if tp + fp else math.nan, "recall": rec, "specificity": spec,
                    "balanced_accuracy": (rec + spec) / 2 if not (math.isnan(rec) or math.isnan(spec)) else math.nan,
                    "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else math.nan,
                    "accuracy": (tp + tn) / n if n else math.nan})
    return out

# --------------------------------------------------------------------------- 부트스트랩 재표집

def site_bootstrap_indices(site_ids, n_boot: int, seed: int):
    """현장을 복원추출하고, 뽑힌 현장의 이미지를 모두 모은다(중복 추출이면 중복 포함)."""
    site_ids = np.asarray(site_ids)
    sites = np.unique(site_ids)
    members = {s: np.flatnonzero(site_ids == s) for s in sites}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        chosen = rng.choice(sites, size=len(sites), replace=True)
        yield np.concatenate([members[s] for s in chosen])


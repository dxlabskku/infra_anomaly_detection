#!/usr/bin/env python
"""S4 평가. 기본은 val 만 계산한다. --test 를 줄 때만 test 를 계산한다 (test 는 최종 구성을 고른 뒤 한 번).

이미지 점수 = rebar(c)·crack 박스 최고 점수 (박스가 없으면 0). 불량 = 점수 >= tau  (MIL max 규칙, 학습 없음)
tau        실험마다 val(normal 조건)에서 F1 최대 (0.01~0.99, 같으면 가장 작은 값). 결측·순열 조건에도 같은 tau 를 쓴다.
R0         점수 >= 0.5 (KPI-2 규칙, tau 를 쓰지 않음)
선택       val F1(normal)이 가장 높은 실험. 주안은 config experiments_main (KPI 판정 대상)
부트스트랩 현장 단위 복원추출 (95% CI), 실험 쌍의 짝지은 차이 (evaluate.paired_diffs)

출력 eval/metrics.json, eval/tables/*.csv, eval/report.md
"""
import math
import sys
import time

import numpy as np
import pandas as pd

import common as C
import metrics as M


def image_scores(preds: pd.DataFrame, filenames, classes) -> np.ndarray:
    d = preds[np.round(preds["cls"]).astype(int).isin(classes)]
    best = d.groupby("filename")["score"].max()
    return pd.Series(list(filenames)).map(best).fillna(0.0).to_numpy(dtype=np.float64)


def threshold_grid(g: dict) -> np.ndarray:
    return np.round(np.arange(float(g["start"]), float(g["stop"]) + 1e-9, float(g["step"])), 6)


def f1_at(y, s, t) -> float:
    p = s >= t
    tp = int((p & y).sum())
    return 2 * tp / (2 * tp + int((p & ~y).sum()) + int((~p & y).sum()) or 1)


def fit_tau(y, s, grid) -> tuple[float, float]:
    """F1 최대 tau. 같으면 가장 작은 tau."""
    f = [f1_at(y, s, t) for t in grid]
    k = int(np.argmax(f))
    return float(grid[k]), float(f[k])


def scoped(idx: pd.DataFrame, cfg: dict) -> dict:
    out = {"all": np.ones(len(idx), bool)}
    for fac in cfg["data"]["facilities"]:
        out[fac] = (idx["facility"] == fac).to_numpy()
    return out


def boot_f1(y, s, tau, sites, n_boot, seed):
    vals = [f1_at(y[b], s[b], tau) for b in M.site_bootstrap_indices(sites, n_boot, seed)]
    return np.asarray(vals)


def ci(arr, q=(2.5, 97.5)) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return {"lo": math.nan, "hi": math.nan, "n": 0}
    return {"lo": float(np.percentile(arr, q[0])), "hi": float(np.percentile(arr, q[1])), "n": int(len(arr))}


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(fmt(c) if not isinstance(c, str) else c for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = C.base_parser(__doc__)
    ap.add_argument("--test", action="store_true", help="test 분할도 계산한다 (최종 구성을 고른 뒤 한 번)")
    ap.add_argument("--exp", nargs="+", help="평가할 실험. 주지 않으면 예측이 있는 실험 전부")
    args = ap.parse_args()
    cfg = C.load_config(args)
    paths = C.Paths(cfg)
    log, log_path = C.setup_logging("s4_evaluate", paths)
    ev = cfg["evaluate"]
    t_all = time.time()
    names = args.exp or [e for e in cfg["experiments"] if (paths.pred_dir(e) / "predict_info.json").exists()]
    if not names:
        raise C.PipelineError("평가할 예측이 없습니다. s3_predict.py 를 먼저 실행하세요.")
    out_dir = C.prepare_output(paths.eval, args.overwrite, paths)
    (out_dir / "tables").mkdir()
    splits = ["val", "test"] if args.test else ["val"]
    df = C.load_manifest(cfg, paths)
    parts = {sp: C.split_frame(df, sp) for sp in splits}
    ys = {sp: (parts[sp]["condition_raw"] == "bad").to_numpy() for sp in splits}
    grid = threshold_grid(ev["threshold_grid"])
    cls = [int(c) for c in ev["defect_classes"]]
    main_exp = cfg["experiments_main"]
    bs = ev["bootstrap"]

    scores, conds, taus, preds_normal = {}, {}, {}, {}
    for e in names:
        pinfo = C.read_json(paths.pred_dir(e) / "predict_info.json")
        conds[e] = pinfo["conditions"]
        scores[e] = {}
        for sp in splits:
            for c in conds[e]:
                p = pd.read_csv(paths.pred_dir(e) / f"preds_{sp}_{c}.csv", dtype={"filename": str})
                scores[e][(sp, c)] = image_scores(p, parts[sp]["filename"], cls)
                if c == "normal":
                    preds_normal[(e, sp)] = p
        taus[e] = fit_tau(ys["val"], scores[e][("val", "normal")], grid)
        log.info("%s: tau %.2f (val F1 %.4f)", e, *taus[e])

    # ---- 이미지 지표
    rows = []
    for e in names:
        tau = taus[e][0]
        for sp in splits:
            sc = scoped(parts[sp], cfg)
            for c in conds[e]:
                s = scores[e][(sp, c)]
                for scope, mk in sc.items():
                    m = M.binary_metrics(ys[sp][mk], s[mk], s[mk] >= tau)
                    r0 = M.binary_metrics(ys[sp][mk], None, s[mk] >= float(ev["r0_score_thr"]))
                    rows.append({"exp": e, "split": sp, "cond": c, "scope": scope, "tau": tau, **m,
                                 "r0_accuracy": r0["accuracy"], "r0_f1": r0["f1"],
                                 "r0_balanced_accuracy": r0["balanced_accuracy"]})
    img = pd.DataFrame(rows)
    img.to_csv(out_dir / "tables" / "image_metrics.csv", index=False, encoding="utf-8")

    def get(e, sp, c="normal", scope="all"):
        r = img[(img.exp == e) & (img.split == sp) & (img.cond == c) & (img.scope == scope)]
        return r.iloc[0] if len(r) else None

    selected = max(names, key=lambda e: (get(e, "val")["f1"], -names.index(e)))

    # ---- 부트스트랩 (현장 단위)
    boot = {}
    pairs = [p for p in ev["paired_diffs"] if p[0] in names and p[1] in names]
    group = bs.get("group", "site_id")   # 재표집 단위 (manifest 열 이름)
    for sp in splits:
        sites = parts[sp][group].to_numpy()
        n_boot, seed = int(bs["n_boot"]), int(bs["seed"])
        samples = list(M.site_bootstrap_indices(sites, n_boot, seed))
        f1s = {e: np.array([f1_at(ys[sp][b], scores[e][(sp, "normal")][b], taus[e][0]) for b in samples])
               for e in names}
        boot[sp] = {"f1_ci": {e: ci(f1s[e]) for e in names},
                    "paired": {f"{a}-{b}": {**ci(f1s[a] - f1s[b]), "mean": float(np.mean(f1s[a] - f1s[b])),
                                            "p_le_0": float(np.mean(f1s[a] - f1s[b] <= 0))} for a, b in pairs},
                    "n_sites": int(len(np.unique(sites))), "n_boot": n_boot}
        for e in names:
            if len(conds[e]) > 1:
                for c in conds[e][1:]:
                    d = np.array([f1_at(ys[sp][b], scores[e][(sp, "normal")][b], taus[e][0])
                                  - f1_at(ys[sp][b], scores[e][(sp, c)][b], taus[e][0]) for b in samples])
                    boot[sp]["paired"][f"{e}:normal-{c}"] = {**ci(d), "mean": float(np.mean(d)),
                                                             "p_le_0": float(np.mean(d <= 0))}

    # ---- 박스 지표 (normal 조건)
    boxes = {}
    for sp in splits:
        gt = M.load_gt(paths, parts[sp], sp, int(cfg["data"]["image_size"]), M.resized_map(paths, sp))
        for e in names:
            pb = M.preds_by_image(preds_normal[(e, sp)], parts[sp]["filename"])
            boxes[f"{e}/{sp}"] = M.evaluate_boxes(gt, pb, parts[sp], cfg)
            o = boxes[f"{e}/{sp}"]["overall"]
            log.info("박스 %s %s: mAP50 %.4f mAP75 %.4f mAP50-95 %.4f", e, sp, o["mAP50"], o["mAP75"], o["mAP50_95"])

    kpi = None
    if "test" in splits and main_exp in names:
        t = get(main_exp, "test")
        kpi = {"exp": main_exp,
               "kpi1_f1": float(t["f1"]), "kpi1_ci": boot["test"]["f1_ci"][main_exp],
               "kpi1_target": float(ev["kpi1_target"]), "kpi1_met": bool(t["f1"] >= float(ev["kpi1_target"])),
               "kpi2_r0_accuracy": float(t["r0_accuracy"]), "kpi2_target": float(ev["kpi2_target"]),
               "kpi2_met": bool(t["r0_accuracy"] >= float(ev["kpi2_target"]))}

    res = {"splits": splits, "experiments": {e: C.experiment(cfg, e) for e in names}, "tau": taus,
           "selected_by_val_f1": selected, "main": main_exp, "kpi": kpi, "bootstrap": boot, "boxes": boxes,
           "image_metrics_csv": str(out_dir / "tables" / "image_metrics.csv"), "log": str(log_path),
           "sec": time.time() - t_all}
    C.write_json(out_dir / "metrics.json", res)

    # ---- 보고서
    facs = cfg["data"]["facilities"]
    md = ["# CARE-YOLO 평가", "", f"분할: {', '.join(splits)}" + ("" if args.test else " (test 는 계산하지 않음. 최종 구성을 고른 뒤 --test)"),
          "", "이미지 점수 = rebar(c)·crack 박스 최고 점수, tau = val F1 최대. R0 = 점수 ≥ 0.5.", "",
          "## 실험", "", md_table(["실험", "입력", "상황", "현장 균형", "설명"],
                                 [[e, C.experiment(cfg, e)["mode"], "+".join(C.experiment(cfg, e)["context"]) or "없음",
                                   C.experiment(cfg, e)["site_balance_power"], cfg["experiments"][e].get("desc", "")]
                                  for e in names])]
    for sp in splits:
        md += ["", f"## {sp} 이미지 지표 (normal)", "",
               md_table(["실험", "tau", "F1", "95% CI", "정밀도", "재현율", "AUROC", "R0 정확도"] + facs,
                        [[e, taus[e][0], get(e, sp)["f1"],
                          f"[{fmt(boot[sp]['f1_ci'][e]['lo'], 3)}, {fmt(boot[sp]['f1_ci'][e]['hi'], 3)}]",
                          get(e, sp)["precision"], get(e, sp)["recall"], get(e, sp)["auroc"], get(e, sp)["r0_accuracy"]]
                         + [get(e, sp, scope=f)["f1"] for f in facs] for e in names])]
        if boot[sp]["paired"]:
            md += ["", f"### {sp} 짝지은 F1 차이 ({group} 단위 부트스트랩 {boot[sp]['n_boot']}회, {boot[sp]['n_sites']}개)", "",
                   md_table(["비교", "평균 차이", "95% CI", "P(차이 ≤ 0)"],
                            [[k, v["mean"], f"[{fmt(v['lo'], 3)}, {fmt(v['hi'], 3)}]", v["p_le_0"]]
                             for k, v in boot[sp]["paired"].items()])]
        multi = [e for e in names if len(conds[e]) > 1]
        for e in multi:
            md += ["", f"### {sp} {e} 상황 조건별 F1 (같은 모델, 같은 tau)", "",
                   md_table(["조건", "F1", "AUROC"] + facs,
                            [[c, get(e, sp, c)["f1"], get(e, sp, c)["auroc"]] + [get(e, sp, c, f)["f1"] for f in facs]
                             for c in conds[e]])]
        md += ["", f"### {sp} 박스 지표 (normal)", "",
               md_table(["실험", "mAP50", "mAP75", "mAP50-95", "rebar AP50", "rebar(c) AP50", "crack AP50"],
                        [[e, boxes[f"{e}/{sp}"]["overall"]["mAP50"], boxes[f"{e}/{sp}"]["overall"]["mAP75"],
                          boxes[f"{e}/{sp}"]["overall"]["mAP50_95"]]
                         + [boxes[f"{e}/{sp}"]["overall"]["per_class"][k]["AP50"]
                            for k in ("rebar", "rebar_abnormal", "crack")] for e in names])]
    md += ["", f"## 선택", "", f"val F1(normal) 최고: **{selected}** ({fmt(get(selected, 'val')['f1'])}). 주안(KPI 대상): {main_exp}."]
    if kpi:
        md += ["", "## KPI (test)", "",
               f"- KPI-1 {main_exp} F1 {kpi['kpi1_f1']:.4f} [{kpi['kpi1_ci']['lo']:.3f}, {kpi['kpi1_ci']['hi']:.3f}] "
               f"(목표 {kpi['kpi1_target']}) → {'달성' if kpi['kpi1_met'] else '미달'}",
               f"- KPI-2 {main_exp} R0 정확도 {kpi['kpi2_r0_accuracy']:.4f} (목표 {kpi['kpi2_target']}) → "
               f"{'달성' if kpi['kpi2_met'] else '미달'}"]
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("평가 완료 (%.1fs). 선택 %s. 보고서 %s", time.time() - t_all, selected, out_dir / "report.md")


if __name__ == "__main__":
    sys.exit(main())

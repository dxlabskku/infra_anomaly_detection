#!/usr/bin/env python
"""S1 입력 준비: gray3·phys3 이미지, YOLO 라벨, 기후 벡터(12차원), 위치 인덱스(3개).

gray3  원본 흑백을 3채널로 복제 (example_03 과 같음)
phys3  GPR 판독 절차(깊이 보정 뒤 상대 비교)를 따른 3채널. s = g - median(g) (배경 밝기 중심)
         ch1  원본 흑백 g
         ch2  상대 진폭  clip(s / P(|s|), -1, 1)                         이미지 안 가장 강한 반사 대비
         ch3  깊이 보정 상대 진폭  s_d = s * clip(exp(k_e (y/512 - y_e)), 0.25, 4),  clip(s_d / P(|s_d|), -1, 1)
              P = phys.ref_percentile 번째 백분위 (99.9). 원본이 이미 0·255 로 포화된 이미지는 포화가 그대로 남는다
       k_e, y_e 는 장비(파일명 앞 CT·HM·NM)별로 train 정상 철근 박스(class 0)에서 log 평균 |s| 를 깊이에 회귀해 정한다
       (k_e = -기울기, y_e = 정상 철근 깊이 중앙값). 장비는 GPR 파일 자체의 정보라 위치 모달을 쓰지 않는다.
       세 채널 모두 [-1, 1] -> 0..255 (127.5 가 0). 파일은 cv2 로 [ch1, ch2, ch3] 순서로 쓴다.

출력 (작업 폴더 기준)
  prepared/{mode}/images/{split}/{filename}   3채널 PNG
  prepared/{mode}/labels/{split}/{stem}.txt   YOLO 라벨 (두 모드 같음)
  prepared/{mode}/data.yaml, prepared/phys3/phys_params.json
  prepared/prepare_info.json
  climate/climate_{split}.npy (N, 12), climate_{split}_index.csv, climate_stats.json
  location/location_{split}.npy (N, 3) int64, location_{split}_index.csv, location_vocab.json
  (--smoke) manifest_smoke.csv
"""
import os

# 16개 프로세스가 각자 BLAS 스레드를 띄우지 않게 한다 (numpy import 전에)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import common as C  # noqa: E402

cv2.setNumThreads(1)

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


# --------------------------------------------------------------------------- 기후

def parse_number(s) -> float:
    """'22.3℃', '37%', '1.2mm', '0mm' -> 숫자. 숫자가 없거나 비어 있으면 nan (결측)."""
    if s is None:
        return math.nan
    if isinstance(s, (int, float, np.integer, np.floating)):
        v = float(s)
        return v if math.isfinite(v) else math.nan
    m = _NUM_RE.search(str(s).replace(",", ""))
    return float(m.group()) if m else math.nan


def climate_raw(temperature, humidity, rain_d0, rain_d1, rain_d2, api_weights=(1.0, 0.5, 0.25)):
    """값 6개 v = [T, H, log1p(r0), log1p(r1), log1p(r2), log1p(API)] 와 결측 마스크 m (결측 1)."""
    T, H, r0, r1, r2 = (parse_number(x) for x in (temperature, humidity, rain_d0, rain_d1, rain_d2))
    rains = (r0, r1, r2)
    api = math.nan if any(math.isnan(r) for r in rains) else sum(w * r for w, r in zip(api_weights, rains))

    def _log1p(x):
        return math.log1p(x) if (not math.isnan(x) and x > -1) else math.nan

    v = np.array([T, H, _log1p(r0), _log1p(r1), _log1p(r2), _log1p(api)], dtype=np.float64)
    m = np.isnan(v).astype(np.float32)
    return v, m


def climate_matrix(df: pd.DataFrame, cfg: dict):
    cols = cfg["climate"]["columns"]
    w = cfg["climate"]["api_weights"]
    V = np.zeros((len(df), 6), dtype=np.float64)
    M = np.zeros((len(df), 6), dtype=np.float32)
    for i, row in enumerate(df[cols].itertuples(index=False)):
        V[i], M[i] = climate_raw(*row, api_weights=w)
    return V, M


def climate_stats(V_train: np.ndarray, M_train: np.ndarray):
    """train 분할에서만 평균·표준편차(모표준편차, ddof=0). 표준편차 0 이면 1로 둔다(StandardScaler 와 같은 규칙)."""
    Vm = np.where(M_train > 0, np.nan, V_train)
    n = (~np.isnan(Vm)).sum(axis=0)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(Vm, axis=0)
        sd = np.nanstd(Vm, axis=0)
    # 상수 특징 판정은 sklearn StandardScaler(_is_constant_feature)와 같다. 같은 값만 있어도
    # 부동소수점 오차로 표준편차가 1e-16 처럼 0 이 아닐 수 있어 허용 오차를 둔다.
    eps = np.finfo(np.float64).eps
    var = sd ** 2
    zero = ~(var > n * eps * var + (n * np.nan_to_num(mu) * eps) ** 2) | (n == 0)
    sd = np.where(zero, 1.0, sd)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(Vm, axis=0) if (n > 0).any() else mu
    mu = np.where(zero & (n > 0), med, mu)   # 상수 특징은 평균의 반올림 오차 없이 표준화 값이 정확히 0
    mu = np.where(n == 0, 0.0, mu)
    return mu, sd, zero


def standardize_climate(V: np.ndarray, M: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """c = [v_z (6), m (6)] float32. 결측이면 v_z = 0."""
    Z = (V - mu) / sd
    Z = np.where(M > 0, 0.0, Z)
    return np.concatenate([Z, M], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- 위치

def location_indices(df: pd.DataFrame, cfg: dict):
    """(N, 3) int64 [region, facility, part]. 어휘 i 번째 값 -> i + 1, 빈 값이나 목록 밖 값 -> 0 (결측).

    두 번째 값은 열마다 목록 밖에서 온 값의 개수(빈 값 제외).
    """
    cols = cfg["location"]["columns"]
    vocab = cfg["location"]["vocab"]
    L = np.zeros((len(df), len(cols)), dtype=np.int64)
    unknown = {}
    for j, col in enumerate(cols):
        table = {v: i + 1 for i, v in enumerate(vocab[col])}
        vals = df[col].astype(str).str.strip()
        L[:, j] = vals.map(table).fillna(0).astype(np.int64).to_numpy()
        unknown[col] = dict(Counter(v for v in vals if v and v not in table))
    return L, unknown


# --------------------------------------------------------------------------- 박스

def center_to_xyxy(x, y, w, h):
    return x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0


def clip_xyxy(box, size):
    return tuple(min(max(float(v), 0.0), float(size)) for v in box)


def xyxy_to_yolo(box, size):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0 / size, (y1 + y2) / 2.0 / size, (x2 - x1) / size, (y2 - y1) / size


def yolo_to_xyxy(cx, cy, bw, bh, size):
    return ((cx - bw / 2.0) * size, (cy - bh / 2.0) * size, (cx + bw / 2.0) * size, (cy + bh / 2.0) * size)


def json_boxes(label: dict):
    """원본 오탈자 키 그대로: annotaions.boxes[i] = {class, coordinates{x, y, width, height}} (x, y 는 중심)."""
    out = []
    for b in label.get("annotaions", {}).get("boxes", []) or []:
        c = b["coordinates"]
        out.append((int(b["class"]), float(c["x"]), float(c["y"]), float(c["width"]), float(c["height"])))
    return out


def convert_boxes(boxes, size=512, min_size=2.0, sx=1.0, sy=1.0):
    """중심 박스 -> xyxy -> [0, size] 자르기 -> 폭·높이 min_size 미만이면 버림 -> YOLO 정규화.

    sx, sy 는 리사이즈 배율(512 / 원래 폭, 512 / 원래 높이). 돌려주는 값: (YOLO 행 목록, 버린 클래스 목록)
    """
    rows, dropped = [], []
    for cls, x, y, w, h in boxes:
        x1, y1, x2, y2 = center_to_xyxy(x * sx, y * sy, w * sx, h * sy)
        x1, y1, x2, y2 = clip_xyxy((x1, y1, x2, y2), size)
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            dropped.append(cls)
            continue
        rows.append((cls, *xyxy_to_yolo((x1, y1, x2, y2), size)))
    return rows, dropped


def yolo_line(cls, cx, cy, bw, bh) -> str:
    return f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# --------------------------------------------------------------------------- phys3 채널

EQUIPMENT = ("CT", "HM", "NM")   # CT: 건축·옹벽, HM: 터널, NM: 교량


def equipment_of(filename: str) -> str:
    e = filename.split("_", 1)[0]
    if e not in EQUIPMENT:
        raise C.PipelineError(f"파일명 앞부분이 장비 코드({EQUIPMENT})가 아닙니다: {filename}")
    return e


def to_u8(x: np.ndarray) -> np.ndarray:
    """[-1, 1] -> 0..255 (0 -> 127.5)."""
    return np.clip(np.rint(127.5 + 127.5 * np.clip(x, -1.0, 1.0)), 0, 255).astype(np.uint8)


def relative(s: np.ndarray, pct: float) -> np.ndarray:
    a = float(np.percentile(np.abs(s), pct))
    return s / max(a, 1.0)


def depth_gain(n_rows: int, kappa: float, y_ref: float, clip=(0.25, 4.0)) -> np.ndarray:
    y = (np.arange(n_rows, dtype=np.float64) + 0.5) / n_rows
    return np.clip(np.exp(kappa * (y - y_ref)), clip[0], clip[1])


def phys3_channels(g: np.ndarray, kappa: float, y_ref: float, pct: float = 99.9, clip=(0.25, 4.0)) -> np.ndarray:
    """흑백 (H, W) uint8 -> (H, W, 3) uint8 [원본, 상대 진폭, 깊이 보정 상대 진폭]."""
    s = g.astype(np.float64) - float(np.median(g))
    ch2 = relative(s, pct)
    ch3 = relative(s * depth_gain(g.shape[0], kappa, y_ref, clip)[:, None], pct)
    return np.dstack([g, to_u8(ch2), to_u8(ch3)])


def _rebar_amplitudes(task):
    """(파일, 라벨) -> 정상 철근 박스(class 0)마다 (중심 깊이 0~1, log 평균 |s|)."""
    src_img, src_lbl, S = task
    g = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)
    h0, w0 = g.shape[:2]
    if (h0, w0) != (S, S):
        g = cv2.resize(g, (S, S), interpolation=cv2.INTER_LINEAR)
    s = np.abs(g.astype(np.float64) - float(np.median(g)))
    with open(src_lbl, encoding="utf-8") as f:
        boxes = json_boxes(json.load(f))
    out = []
    for cls, x, y, w, h in boxes:
        if cls != 0:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in clip_xyxy(center_to_xyxy(x * S / w0, y * S / h0, w * S / w0,
                                                                         h * S / h0), S))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append(((y1 + y2) / 2.0 / S, math.log(float(s[y1:y2, x1:x2].mean()) + 1e-3)))
    return out


def fit_depth_params(train: pd.DataFrame, paths: C.Paths, cfg: dict, pool) -> dict:
    """장비별 k_e (깊이 감쇠율, 1/이미지 높이) 와 y_e (정상 철근 깊이 중앙값). train 정상 철근 박스만 쓴다."""
    ph = cfg["phys"]
    S = int(cfg["data"]["image_size"])
    rng = np.random.default_rng(int(cfg["seed"]))
    params = {}
    for e in EQUIPMENT:
        files = [f for f in train["filename"] if equipment_of(f) == e]
        if not files:
            continue
        n = min(len(files), int(ph["fit_max_images_per_equipment"]))
        sel = sorted(rng.choice(files, size=n, replace=False))
        tasks = [(str(paths.src_image("train", f)), str(paths.src_label("train", f)), S) for f in sel]
        pts = [p for res in pool.imap(_rebar_amplitudes, tasks, chunksize=16) for p in res]
        if len(pts) < 50:
            raise C.PipelineError(f"장비 {e} 의 정상 철근 박스가 {len(pts)}개뿐이라 깊이 감쇠를 정할 수 없습니다.")
        y, a = np.array(pts).T
        slope, icpt = np.polyfit(y, a, 1)
        params[e] = {"kappa": float(-slope), "y_ref": float(np.median(y)), "n_images": n, "n_boxes": int(len(y)),
                     "corr_depth_logamp": float(np.corrcoef(y, a)[0, 1]),
                     "depth_p05_p95": [float(np.percentile(y, 5)), float(np.percentile(y, 95))]}
    return params


# --------------------------------------------------------------------------- 영상

_PRM: dict = {}


def _init_worker(prm):
    global _PRM
    _PRM = prm
    cv2.setNumThreads(1)


def _process(task):
    split, filename, src_img, src_lbl, outs = task
    S = _PRM["size"]
    g = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {src_img}")
    h0, w0 = g.shape[:2]
    resized = (h0, w0) != (S, S)
    if resized:
        g = cv2.resize(g, (S, S), interpolation=cv2.INTER_LINEAR)
    with open(src_lbl, encoding="utf-8") as f:
        label = json.load(f)
    rows, dropped = convert_boxes(json_boxes(label), S, _PRM["min_box_size"], S / w0, S / h0)
    for mode, img_path, lbl_path in outs:
        if mode == "gray3":
            img = np.dstack([g, g, g])
        else:
            p = _PRM["phys"][equipment_of(filename)]
            img = phys3_channels(g, p["kappa"], p["y_ref"], _PRM["ref_pct"], _PRM["gain_clip"])
        if not cv2.imwrite(img_path, img, [cv2.IMWRITE_PNG_COMPRESSION, _PRM["png_compression"]]):
            raise RuntimeError(f"저장 실패: {img_path}")
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("".join(yolo_line(*r) + "\n" for r in rows))
    return {
        "split": split, "filename": filename, "resized": resized, "orig_hw": (h0, w0),
        "n_boxes": len(rows) + len(dropped), "n_kept": len(rows), "dropped": dropped,
        "kept_cls": [r[0] for r in rows],
    }


# --------------------------------------------------------------------------- main

def main():
    ap = C.base_parser(__doc__)
    ap.add_argument("--workers", type=int, help="prepare.workers 덮어쓰기")
    args = ap.parse_args()
    cfg = C.load_config(args, [("prepare.workers", args.workers)])
    paths = C.Paths(cfg)
    log, log_path = C.setup_logging("s1_prepare", paths)
    C.set_seed(int(cfg["seed"]))
    log.info("설정 덮어쓰기: %s", cfg["_overrides"])
    t_start = time.time()
    notes = []

    # 스모크 부분집합 만들기
    if cfg["_smoke"]:
        full = pd.read_csv(paths.manifest_full, dtype=str, keep_default_na=False, encoding="utf-8")
        sub, smoke_notes = C.build_smoke_manifest(full, cfg)
        C.prepare_output(paths.smoke_manifest, args.overwrite, paths, make_dir=False)
        sub.to_csv(paths.smoke_manifest, index=False, encoding="utf-8")
        notes += [f"스모크: {n}" for n in smoke_notes]
        log.info("스모크 manifest %s (%d장, 불량 %d장). %s", paths.smoke_manifest, len(sub),
                 int((sub["condition_raw"] == "bad").sum()), "; ".join(smoke_notes))

    df = C.load_manifest(cfg, paths)
    counts = {s: int((df["split"] == s).sum()) for s in cfg["data"]["splits"]}
    log.info("분할별 이미지 수: %s", counts)

    modes = list(cfg["prepare"]["modes"])
    if not modes or any(m not in C.MODES for m in modes):
        raise C.PipelineError(f"입력 모드는 {C.MODES} 중에서 고릅니다: {modes}")
    for mode in modes:
        C.prepare_output(paths.prepared_mode(mode), args.overwrite, paths)
    C.prepare_output(paths.prepared / "prepare_info.json", args.overwrite, paths, make_dir=False)
    C.prepare_output(paths.climate, args.overwrite, paths)
    C.prepare_output(paths.location, args.overwrite, paths)

    # ---- 기후 벡터
    t0 = time.time()
    clim_info = {}
    V_all, M_all = {}, {}
    for split in cfg["data"]["splits"]:
        part = C.split_frame(df, split)
        V_all[split], M_all[split] = climate_matrix(part, cfg)
    mu, sd, zero_sd = climate_stats(V_all["train"], M_all["train"])
    names = cfg["climate"]["feature_names"]
    if zero_sd.any():
        zs = [names[i] for i in np.where(zero_sd)[0]]
        notes.append(f"기후: train 표준편차가 0인 특징 {zs} 은 표준편차를 1로 두었다 (StandardScaler 규칙)")
        log.warning("train 표준편차 0 인 기후 특징: %s -> 표준편차 1 사용", zs)
    stats = {
        "feature_names": names, "mean": mu.tolist(), "std": sd.tolist(), "std_was_zero": zero_sd.tolist(),
        "computed_on": "train", "n_train": int(len(V_all["train"])), "ddof": 0,
        "api": "API = r0 + 0.5*r1 + 0.25*r2", "vector": "c = [v_z(6), m(6)], 결측이면 v_z=0, m=1",
    }
    C.write_json(paths.climate / "climate_stats.json", stats)
    for split in cfg["data"]["splits"]:
        part = C.split_frame(df, split)
        cvec = standardize_climate(V_all[split], M_all[split], mu, sd)
        np.save(paths.climate / f"climate_{split}.npy", cvec)
        part[["filename"]].to_csv(paths.climate / f"climate_{split}_index.csv", index=False, encoding="utf-8")
        miss = M_all[split].sum(axis=0).astype(int)
        clim_info[split] = {"shape": list(cvec.shape), "missing_per_feature": dict(zip(names, miss.tolist())),
                            "rows_with_any_missing": int((M_all[split].sum(axis=1) > 0).sum())}
        if miss.sum() > 0:
            log.warning("[보고] %s 기후 결측: %s", split, clim_info[split]["missing_per_feature"])
            notes.append(f"기후 결측 {split}: {clim_info[split]['missing_per_feature']}")
    log.info("기후 벡터 완료 (%.1fs). train 평균 %s, 표준편차 %s", time.time() - t0,
             np.round(mu, 4).tolist(), np.round(sd, 4).tolist())

    # ---- 위치 인덱스
    loc_cols = cfg["location"]["columns"]
    loc_info = {}
    C.write_json(paths.location / "location_vocab.json", {
        "columns": loc_cols, "vocab": cfg["location"]["vocab"],
        "rule": "어휘 i 번째 값 -> i + 1, 빈 값이나 목록 밖 값 -> 0 (결측)"})
    for split in cfg["data"]["splits"]:
        part = C.split_frame(df, split)
        L, unknown = location_indices(part, cfg)
        np.save(paths.location / f"location_{split}.npy", L)
        part[["filename"]].to_csv(paths.location / f"location_{split}_index.csv", index=False, encoding="utf-8")
        n_missing = {c: int((L[:, j] == 0).sum()) for j, c in enumerate(loc_cols)}
        loc_info[split] = {"shape": list(L.shape), "missing_per_column": n_missing, "out_of_vocab": unknown}
        if any(unknown.values()):
            log.warning("[보고] %s 위치 어휘 밖 값 (결측으로 처리): %s", split, unknown)
            notes.append(f"위치 어휘 밖 값 {split}: {unknown}")
        log.info("위치 %s: %s, 결측 %s", split, L.shape, n_missing)

    # ---- 영상과 라벨
    S = int(cfg["data"]["image_size"])
    workers = int(cfg["prepare"]["workers"])
    ctx = mp.get_context("fork")
    phys = {}
    if "phys3" in modes:
        t0 = time.time()
        with ctx.Pool(workers) as pool:
            phys = fit_depth_params(C.split_frame(df, "train"), paths, cfg, pool)
        for e, p in phys.items():
            log.info("phys3 깊이 감쇠 %s: k=%.3f (깊이 끝까지 %.2f배), y_ref=%.3f, 상관 %.3f, 박스 %d개 (이미지 %d장)",
                     e, p["kappa"], math.exp(-p["kappa"]), p["y_ref"], p["corr_depth_logamp"], p["n_boxes"],
                     p["n_images"])
        C.write_json(paths.prepared_mode("phys3") / "phys_params.json", {
            "equipment": phys, "ref_percentile": float(cfg["phys"]["ref_percentile"]),
            "gain_clip": list(cfg["phys"]["gain_clip"]), "fit_on": "train 정상 철근 박스 (class 0)",
            "rule": "s=g-median(g); ch2=clip(s/P(|s|),-1,1); ch3=clip(s*g(y)/P(|s*g(y)|),-1,1), "
                    "g(y)=clip(exp(k(y-y_ref)),lo,hi); [-1,1]->0..255", "sec": time.time() - t0})
        missing_eq = sorted({equipment_of(f) for f in df["filename"]} - set(phys))
        if missing_eq:
            raise C.PipelineError(f"train 에 없는 장비가 val·test 에 있습니다: {missing_eq}")
    prm = {"size": S, "min_box_size": float(cfg["prepare"]["min_box_size"]),
           "png_compression": int(cfg["prepare"]["png_compression"]), "phys": phys,
           "ref_pct": float(cfg["phys"]["ref_percentile"]), "gain_clip": tuple(cfg["phys"]["gain_clip"])}
    for mode in modes:
        for split in cfg["data"]["splits"]:
            (paths.prepared_mode(mode) / "images" / split).mkdir(parents=True, exist_ok=True)
            (paths.prepared_mode(mode) / "labels" / split).mkdir(parents=True, exist_ok=True)
    tasks = []
    for r in df.itertuples(index=False):
        src_img, src_lbl = paths.src_image(r.split, r.filename), paths.src_label(r.split, r.filename)
        if not src_img.exists() or not src_lbl.exists():
            raise C.PipelineError(f"원본 파일이 없습니다: {src_img} / {src_lbl}")
        outs = [(m, str(paths.prepared_image(m, r.split, r.filename)), str(paths.prepared_label(m, r.split, r.filename)))
                for m in modes]
        tasks.append((r.split, r.filename, str(src_img), str(src_lbl), outs))

    log.info("영상 %d장 처리 시작 (%s, workers %d)", len(tasks), "+".join(modes), workers)
    t0 = time.time()
    results = []
    with ctx.Pool(workers, initializer=_init_worker, initargs=(prm,)) as pool:
        for i, res in enumerate(pool.imap_unordered(_process, tasks, chunksize=4), 1):
            results.append(res)
            if i % 1000 == 0 or i == len(tasks):
                el = time.time() - t0
                log.info("  %d/%d (%.1f장/s, 경과 %.0fs)", i, len(tasks), i / max(el, 1e-9), el)
    t_images = time.time() - t0

    resized = [r for r in results if r["resized"]]
    dropped = Counter(c for r in results for c in r["dropped"])
    kept = Counter(c for r in results for c in r["kept_cls"])
    if resized:
        log.warning("[보고] 512x512 가 아니어서 리사이즈한 이미지 %d장", len(resized))
        notes.append(f"리사이즈한 이미지 {len(resized)}장 (기대 0장)")
    log.info("리사이즈 %d장, 박스 유지 %s, 버림(2px 미만) %s", len(resized), dict(kept), dict(dropped))

    names_map = {int(k): v for k, v in cfg["data"]["class_names"].items()}
    # 검출기 best.pt 선택에 쓸 val: prepare.detector_val_source 가 있으면 manifest source 가 그 값인 val 행만 목록 파일로
    det_val = {"source": None, "n": counts.get("val")}
    dv = cfg["prepare"].get("detector_val_source")
    vrows = C.split_frame(df, "val")
    sel = vrows[vrows["source"] == dv] if (dv and "source" in vrows.columns) else vrows.iloc[0:0]
    if dv and len(sel) == 0:
        log.warning("[보고] detector_val_source=%s 인 val 행이 없어 val 전체로 best.pt 를 고릅니다.", dv)
        notes.append(f"detector_val_source={dv} 인 val 행이 없어 val 전체 사용")
    for mode in modes:
        val_entry = "images/val"
        if len(sel):
            lst = paths.prepared_mode(mode) / "val_detector.txt"
            lst.write_text("".join(str(paths.prepared_image(mode, "val", f)) + "\n" for f in sel["filename"]),
                           encoding="utf-8")
            val_entry = str(lst)
            det_val = {"source": dv, "n": int(len(sel))}
        data = {"path": str(paths.prepared_mode(mode)), "train": "images/train", "val": val_entry,
                "test": "images/test", "names": names_map}
        with open(paths.prepared_mode(mode) / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    if len(sel):
        log.info("검출기 best.pt 선택용 val: source=%s %d장 (val 전체 %d장)", dv, len(sel), len(vrows))

    info = {
        "modes": modes, "n_images": counts, "image_size": S,
        "image_rule": {"gray3": "원본 흑백을 3채널로 복제", "phys3": "[원본, 상대 진폭, 깊이 보정 상대 진폭] (phys_params.json)"},
        "phys_params": phys,
        "resized_count": len(resized),
        "resized": [{"split": r["split"], "filename": r["filename"], "h": r["orig_hw"][0], "w": r["orig_hw"][1]}
                    for r in resized],
        "boxes_total": int(sum(r["n_boxes"] for r in results)), "boxes_kept_by_class": dict(kept),
        "boxes_dropped_by_class": dict(dropped), "boxes_dropped_total": int(sum(dropped.values())),
        "images_without_boxes": int(sum(1 for r in results if r["n_kept"] == 0)),
        "climate": clim_info, "climate_stats": stats, "location": loc_info, "detector_val": det_val,
        "timing_sec": {"images_and_labels": t_images, "total": time.time() - t_start},
        "workers": workers, "notes": notes, "log": str(log_path), "smoke": cfg["_smoke"],
    }
    C.write_json(paths.prepared / "prepare_info.json", info)
    log.info("전처리 완료 (%.1fs). 정보: %s", time.time() - t_start, paths.prepared / "prepare_info.json")


if __name__ == "__main__":
    main()

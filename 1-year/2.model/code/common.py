#!/usr/bin/env python
"""CARE-YOLO 공통 유틸: 설정 로드, 시드, 로깅, manifest 로드, 경로, 출력 보호.

명령줄로도 쓴다.
  python common.py work-dir --config config.yaml [--smoke]   작업 폴더 경로 출력
  python common.py env --out env.txt                         실행 환경 기록
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CODE_DIR / "config.yaml"
SPLITS = ("train", "val", "test")
MODES = ("gray3", "phys3")
MANIFEST_COLUMNS = (
    "filename", "split", "fold", "facility", "kind", "condition_raw", "site_id",
    "temperature", "humidity", "rain_d0", "rain_d1", "rain_d2", "region", "part", "dst_image", "dst_label",
)
INDEX_COLUMNS = ("filename", "split", "facility", "kind", "condition_raw", "site_id")


class PipelineError(RuntimeError):
    """사양과 맞지 않는 상황. 임의로 고치지 않고 멈춘다."""


class OutputExistsError(PipelineError):
    """출력이 이미 있고 --overwrite 가 없다."""


# --------------------------------------------------------------------------- 설정

def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="설정 파일 (기본: code/config.yaml)")
    p.add_argument("--smoke", action="store_true", help="config 의 smoke.overrides 를 적용 (출력은 smoke 작업 폴더)")
    p.add_argument("--overwrite", action="store_true", help="출력이 이미 있으면 지우고 다시 만든다")
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                   help="설정 덮어쓰기. 예: --set fusion.train.lr=5e-4 (여러 번 가능)")
    return p


def _set_dotted(cfg: dict, key: str, value) -> None:
    parts = key.split(".")
    cur = cfg
    for k in parts[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"설정 키가 없습니다: {key}")
        cur = cur[k]
    if not isinstance(cur, dict) or parts[-1] not in cur:
        raise KeyError(f"설정 키가 없습니다: {key}")
    cur[parts[-1]] = value


def parse_cli_value(text: str):
    """--set 값. YAML 로 읽되, PyYAML 이 문자열로 남기는 '5e-4' 같은 지수 표기는 숫자로 바꾼다."""
    v = yaml.safe_load(text)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def get_dotted(cfg: dict, key: str):
    cur = cfg
    for k in key.split("."):
        cur = cur[k]
    return cur


def load_config(args: argparse.Namespace, extra: list[tuple[str, object]] | None = None) -> dict:
    """config.yaml -> (--smoke 이면 smoke.overrides) -> --set -> 스크립트 단축 인자 순으로 덮어쓴다."""
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    applied = []
    if getattr(args, "smoke", False):
        for k, v in cfg["smoke"]["overrides"].items():
            _set_dotted(cfg, k, v)
            applied.append({"key": k, "value": v, "source": "smoke"})
    for item in getattr(args, "sets", []) or []:
        if "=" not in item:
            raise ValueError(f"--set 형식은 KEY=VALUE 입니다: {item}")
        k, v = item.split("=", 1)
        v = parse_cli_value(v)
        _set_dotted(cfg, k.strip(), v)
        applied.append({"key": k.strip(), "value": v, "source": "--set"})
    for k, v in extra or []:
        if v is None:
            continue
        _set_dotted(cfg, k, v)
        applied.append({"key": k, "value": v, "source": "cli"})
    cfg["_smoke"] = bool(getattr(args, "smoke", False))
    cfg["_overrides"] = applied
    cfg["_config_path"] = str(Path(args.config).resolve())
    # 상대 경로는 config 파일 폴더 기준으로 바꾼다 (s2 가 작업 중 cwd 를 바꾸므로 미리 절대 경로로)
    base = Path(cfg["_config_path"]).parent
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = str((base / v).resolve())
    w = cfg["detector"]["weights"]
    if Path(w).parent != Path("."):   # 파일 이름만 있으면 Ultralytics 가 자동으로 내려받는다
        cfg["detector"]["weights"] = str((base / w).resolve())
    return cfg


def public_config(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- 경로

class Paths:
    def __init__(self, cfg: dict):
        p = cfg["paths"]
        self.aihub = Path(p["aihub_root"])
        self.subset = Path(p["subset_dir"])
        self.manifest_full = Path(p["manifest"])
        self.work = Path(p["work_dir"])
        self.logs = self.work / "logs"
        self.prepared = self.work / "prepared"
        self.climate = self.work / "climate"
        self.location = self.work / "location"
        self.detectors = self.work / "detectors"
        self.preds = self.work / "preds"
        self.eval = self.work / "eval"
        self.weights_run = self.work / "weights"           # Ultralytics 가 보조 가중치를 받는 곳(cwd)
        self.ultralytics_cfg = self.work / ".ultralytics"   # YOLO_CONFIG_DIR
        self.smoke_manifest = self.work / "manifest_smoke.csv"

    def prepared_mode(self, mode: str) -> Path:
        return self.prepared / mode

    def prepared_image(self, mode: str, split: str, filename: str) -> Path:
        return self.prepared / mode / "images" / split / filename

    def prepared_label(self, mode: str, split: str, filename: str) -> Path:
        return self.prepared / mode / "labels" / split / (Path(filename).stem + ".txt")

    def src_image(self, split: str, filename: str) -> Path:
        return self.subset / "images" / split / filename

    def src_label(self, split: str, filename: str) -> Path:
        return self.subset / "labels" / split / (Path(filename).stem + ".json")

    def det_dir(self, exp: str) -> Path:
        return self.detectors / exp

    def pred_dir(self, exp: str) -> Path:
        return self.preds / exp


def ensure_inside_work(path: Path, paths: Paths) -> None:
    """작업 폴더 밖(서브셋·원본 데이터 포함)에는 절대 쓰지 않는다."""
    work = paths.work.resolve()
    target = Path(path).resolve()
    if target != work and work not in target.parents:
        raise PipelineError(f"작업 폴더 밖에는 쓰지 않습니다: {target}")
    for protected in (paths.subset.resolve(),):
        if target == protected or protected in target.parents:
            raise PipelineError(f"보호 폴더에는 쓰지 않습니다: {target}")


def prepare_output(path: Path, overwrite: bool, paths: Paths, make_dir: bool = True) -> Path:
    """출력이 있으면 멈춘다. --overwrite 일 때만 지우고 다시 만든다."""
    path = Path(path)
    ensure_inside_work(path, paths)
    if path.exists():
        if not overwrite:
            raise OutputExistsError(f"출력이 이미 있습니다: {path}\n덮어쓰려면 --overwrite 를 주세요.")
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    if make_dir:
        path.mkdir(parents=True)
    return path


def experiment(cfg: dict, name: str) -> dict:
    """config experiments 의 실험 하나. mode(gray3|phys3), context(쓰는 보조 모달), site_balance_power."""
    exps = cfg["experiments"]
    if name not in exps:
        raise PipelineError(f"실험 이름이 config experiments 에 없습니다: {name} (있는 것: {list(exps)})")
    e = dict(exps[name])
    if e["mode"] not in MODES:
        raise PipelineError(f"실험 {name} 의 입력 모드가 {MODES} 가 아닙니다: {e['mode']}")
    bad = [m for m in e.get("context", []) if m not in ("clim", "loc")]
    if bad:
        raise PipelineError(f"실험 {name} 의 context 에 모르는 모달이 있습니다: {bad}")
    e["context"] = list(e.get("context", []))
    e["site_balance_power"] = float(e.get("site_balance_power", 0.0))
    return e


# --------------------------------------------------------------------------- 시드·로깅

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if "torch" in sys.modules:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


_FMT = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
_FILE_HANDLER: logging.Handler | None = None


def setup_logging(script: str, paths: Paths) -> tuple[logging.Logger, Path]:
    global _FILE_HANDLER
    ensure_inside_work(paths.logs, paths)
    paths.logs.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = paths.logs / f"{script}_{ts}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(_FMT)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_FMT)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)
    _FILE_HANDLER = fh
    log = logging.getLogger(script)

    def _hook(exc_type, exc, tb):
        log.error("오류로 중단합니다: %s", exc, exc_info=(exc_type, exc, tb))

    sys.excepthook = _hook
    log.info("로그 파일: %s", log_path)
    return log, log_path


def attach_ultralytics_log() -> None:
    """Ultralytics 로거(propagate=False)도 같은 로그 파일에 남긴다. ultralytics import 뒤에 부른다."""
    if _FILE_HANDLER is not None:
        lg = logging.getLogger("ultralytics")
        if _FILE_HANDLER not in lg.handlers:
            lg.addHandler(_FILE_HANDLER)


def setup_ultralytics_env(paths: Paths) -> None:
    """ultralytics import 전에 부른다. 설정·폰트·보조 가중치를 작업 폴더 안에 둔다."""
    ensure_inside_work(paths.ultralytics_cfg, paths)
    # Ultralytics 는 YOLO_CONFIG_DIR/Ultralytics 를 쓰고, 없으면 /tmp 로 빠진다. 미리 만들어 둔다.
    (paths.ultralytics_cfg / "Ultralytics").mkdir(parents=True, exist_ok=True)
    (paths.work / ".mplconfig").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(paths.ultralytics_cfg)
    os.environ["MPLCONFIGDIR"] = str(paths.work / ".mplconfig")
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")


def configure_ultralytics_settings(paths: Paths) -> None:
    """ultralytics import 뒤에 부른다. 사용 통계 전송(sync)을 끄고 기본 폴더를 작업 폴더로 둔다."""
    from ultralytics import settings
    settings.update({
        "sync": False,
        "runs_dir": str(paths.work / "runs"),
        "weights_dir": str(paths.weights_run),
        "datasets_dir": str(paths.work),
        # 외부 실험 추적 연동은 쓰지 않는다 (패키지도 설치하지 않음)
        "clearml": False, "comet": False, "dvc": False, "mlflow": False, "raytune": False,
        "tensorboard": False, "wandb": False,
    })
    # Events 는 import 때 sync 값을 읽는다. 설정 파일이 처음 만들어진 실행에서도 꺼지도록 직접 끈다.
    from ultralytics.utils import events as _events
    _events.events.enabled = False


# --------------------------------------------------------------------------- manifest

def build_smoke_manifest(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """8-3: 시설별·분할별 filename 정렬 순 처음 k장. 불량이 하나도 없으면 칸마다 불량 n장을 넣는다."""
    k = int(cfg["smoke"]["per_facility_split"])
    n_bad = int(cfg["smoke"]["bad_per_cell_if_none"])
    cells = []
    for split in cfg["data"]["splits"]:
        for fac in cfg["data"]["facilities"]:
            cell = df[(df["split"] == split) & (df["facility"] == fac)].sort_values("filename")
            if len(cell) < k:
                raise PipelineError(f"스모크 칸 {split}/{fac} 의 이미지가 {len(cell)}장으로 {k}장보다 적습니다.")
            cells.append(cell)
    sub = pd.concat([c.head(k) for c in cells])
    notes = []
    total_bad = int((sub["condition_raw"] == "bad").sum())
    if total_bad == 0:
        adjusted = []
        for cell in cells:
            bad = cell[cell["condition_raw"] == "bad"].head(n_bad)
            rest = cell[~cell["filename"].isin(bad["filename"])].head(k - len(bad))
            adjusted.append(pd.concat([rest, bad]).sort_values("filename"))
        sub = pd.concat(adjusted)
        notes.append(f"처음 {k}장에 불량이 없어 칸마다 불량 {n_bad}장을 넣었다")
    else:
        notes.append(f"처음 {k}장에 불량 {total_bad}장이 있어 조정하지 않았다")
    return sub.sort_values(["split", "filename"]).reset_index(drop=True), notes


def check_split_counts(df: pd.DataFrame, cfg: dict) -> dict:
    expected = {k: int(v) for k, v in cfg["data"]["expected_split_counts"].items()}
    actual = {s: int((df["split"] == s).sum()) for s in cfg["data"]["splits"]}
    if actual != expected:
        raise PipelineError(f"분할별 이미지 수가 기대와 다릅니다. 기대 {expected}, 실제 {actual}")
    if df["filename"].duplicated().any():
        raise PipelineError("manifest 에 중복 filename 이 있습니다.")
    return actual


def load_manifest(cfg: dict, paths: Paths) -> pd.DataFrame:
    path = paths.smoke_manifest if cfg["_smoke"] else paths.manifest_full
    if not path.exists():
        hint = " 먼저 s1_prepare.py --smoke 를 실행하세요." if cfg["_smoke"] else ""
        raise PipelineError(f"manifest 가 없습니다: {path}.{hint}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise PipelineError(f"manifest 에 열이 없습니다: {missing}")
    check_split_counts(df, cfg)
    return df.sort_values(["split", "filename"]).reset_index(drop=True)


def split_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """분할 하나. 모든 산출물의 행 순서는 filename 정렬 순으로 고정한다."""
    return df[df["split"] == split].sort_values("filename").reset_index(drop=True)


# --------------------------------------------------------------------------- 입출력

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"JSON 으로 바꿀 수 없음: {type(o)}")


def _clean_nan(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {str(k): _clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_nan(v) for v in o]
    return o


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean_nan(obj), f, ensure_ascii=False, indent=2, default=_json_default)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def package_version(name: str) -> str:
    from importlib import metadata
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "설치 안 됨"


def write_env_txt(path: Path) -> None:
    lines = [f"# 기록 시각: {dt.datetime.now().isoformat(timespec='seconds')}",
             f"# host: {platform.node()}",
             f"# python: {sys.version.split()[0]} ({sys.executable})"]
    try:
        import torch
        lines.append(f"# torch: {torch.__version__}  cuda(build): {torch.version.cuda}  "
                     f"cudnn: {torch.backends.cudnn.version()}  cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                lines.append(f"# visible gpu {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"# torch: import 실패 ({e})")
    lines.append(f"# ultralytics: {package_version('ultralytics')}")  # import 하면 설정 파일을 만들므로 메타데이터만 읽는다
    lines.append(f"# CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    lines.append("\n## nvidia-smi")
    try:
        lines.append(subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60).stdout)
    except Exception as e:  # noqa: BLE001
        lines.append(f"nvidia-smi 실패: {e}")
    lines.append("\n## pip freeze")
    lines.append(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("work-dir")
    w.add_argument("--config", default=str(DEFAULT_CONFIG))
    w.add_argument("--smoke", action="store_true")
    e = sub.add_parser("env")
    e.add_argument("--out", required=True)
    a = p.parse_args()
    if a.cmd == "work-dir":
        cfg = load_config(argparse.Namespace(config=a.config, smoke=a.smoke, sets=[]))
        print(Paths(cfg).work)
    elif a.cmd == "env":
        write_env_txt(Path(a.out))
        print(a.out)


if __name__ == "__main__":
    _cli()

"""CARE-YOLO (Context-Aware Reflection Evaluation YOLO): 기후·위치로 YOLOv8s 특징을 보정하는 검출기.

GPR 철근 반사의 세기는 깊이, 콘크리트 수분(기후), 장비·부위(위치)에 따라 기준선이 달라진다.
그래서 기후·위치를 판정 끝단에 붙이지 않고, YOLO 안의 특징 채널을 조절(FiLM)하는 데 쓴다.

  상황 인코더  기후 c (12) -> 결측이면 [t_miss (6), 1 x 6] -> Linear(12,32) -> GELU -> Linear(32,32) -> GELU -> z_c (32)
              위치 l (3)  -> 결측이면 인덱스 0 -> [Emb_region 8, Emb_facility 4, Emb_part 4] -> Linear(16,16) -> GELU -> z_l (16)
              z = 실험이 쓰는 모달의 z 를 이어 붙임 (C 32, L 16, CL 48)
  FiLM        층 k 의 출력 y (B, C_k, H, W) 에  [g_k, b_k] = Linear(|z|, 2 C_k)(z),  y' = y * (1 + g_k) + b_k
              Linear 를 0 으로 시작하므로 처음에는 GPR 단독 YOLO 와 출력이 같다.

상황 값은 배치마다 CTX.set(...) 으로 넣는다 (학습기·검증기·추론 루프). 넣지 않았으면 FiLM 을 건너뛴다(GPR 단독과 같음).
CTX 를 모델 밖에 두는 이유: Ultralytics 는 EMA 복사본·체크포인트를 따로 만들어 forward 하므로, 배치 값을 모델 속성으로
두면 어느 사본에 넣어야 하는지가 꼬인다. 각 모델은 자기 인코더로 z 를 계산하고(CareDetectionModel._predict_once),
FiLM 층은 지금 forward 중인 모델이 넣어 둔 z 만 읽는다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel

MODALITIES = ("clim", "loc")   # miss 열 순서


class ContextState:
    """현재 배치의 상황 값 (clim (B,12) float, loc (B,3) long, miss (B,2) bool) 과 forward 중인 모델의 z."""

    def __init__(self):
        self.clear()
        self.used = self.skipped = 0

    def set(self, clim: torch.Tensor, loc: torch.Tensor, miss: torch.Tensor) -> None:
        if not (clim.shape[0] == loc.shape[0] == miss.shape[0]):
            raise ValueError(f"상황 값의 배치 크기가 다릅니다: {clim.shape}, {loc.shape}, {miss.shape}")
        self.clim, self.loc, self.miss = clim, loc, miss.bool()

    def clear(self) -> None:
        self.clim = self.loc = self.miss = self.z = None

    @property
    def active(self) -> bool:
        return self.clim is not None


CTX = ContextState()


class ContextEncoder(nn.Module):
    def __init__(self, mods, clim_dim: int = 12, clim_hidden: int = 32, clim_z: int = 32,
                 loc_cards=(17, 5, 5), loc_emb_dims=(8, 4, 4), loc_z: int = 16):
        super().__init__()
        self.mods = tuple(m for m in MODALITIES if m in mods)
        if not self.mods:
            raise ValueError("상황 인코더에는 모달이 하나 이상 있어야 합니다.")
        self.z_dim = 0
        if "clim" in self.mods:
            self.n_clim_val = clim_dim // 2
            self.miss_token = nn.Parameter(torch.zeros(self.n_clim_val))
            self.clim_enc = nn.Sequential(nn.Linear(clim_dim, clim_hidden), nn.GELU(),
                                          nn.Linear(clim_hidden, clim_z), nn.GELU())
            self.z_dim += clim_z
        if "loc" in self.mods:
            self.loc_embs = nn.ModuleList(nn.Embedding(int(c), int(d)) for c, d in zip(loc_cards, loc_emb_dims))
            self.loc_enc = nn.Sequential(nn.Linear(int(sum(loc_emb_dims)), loc_z), nn.GELU())
            self.z_dim += loc_z

    def forward(self, clim: torch.Tensor, loc: torch.Tensor, miss: torch.Tensor) -> torch.Tensor:
        p = next(self.parameters())
        miss = miss.to(p.device).bool()
        zs = []
        if "clim" in self.mods:
            c = clim.to(device=p.device, dtype=p.dtype)
            fill = torch.cat([self.miss_token, torch.ones(self.n_clim_val, dtype=p.dtype, device=p.device)])
            c = torch.where(miss[:, 0:1], fill.expand_as(c), c)
            zs.append(self.clim_enc(c))
        if "loc" in self.mods:
            li = torch.where(miss[:, 1:2], torch.zeros_like(loc), loc).to(p.device).long()
            zs.append(self.loc_enc(torch.cat([emb(li[:, k]) for k, emb in enumerate(self.loc_embs)], dim=1)))
        return torch.cat(zs, dim=1)


class FiLMWrap(nn.Module):
    """Ultralytics 층 하나를 감싸 출력에 FiLM 을 건다. _predict_once 가 쓰는 f, i, type, np 를 그대로 둔다."""

    def __init__(self, layer: nn.Module, z_dim: int, channels: int):
        super().__init__()
        self.layer = layer
        self.gen = nn.Linear(z_dim, 2 * channels)
        nn.init.zeros_(self.gen.weight)
        nn.init.zeros_(self.gen.bias)
        for a in ("f", "i", "type", "np"):
            setattr(self, a, getattr(layer, a))

    def forward(self, x):
        y = self.layer(x)
        z = CTX.z
        if z is None:
            return y
        g, b = self.gen(z.to(self.gen.weight.dtype)).chunk(2, dim=1)
        return y * (1 + g[:, :, None, None].to(y.dtype)) + b[:, :, None, None].to(y.dtype)


class CareDetectionModel(DetectionModel):
    """DetectionModel + 상황 인코더. forward 마다 CTX 의 상황 값으로 z 를 만들어 FiLM 층에 넘긴다."""

    def _predict_once(self, x, *args, **kwargs):
        enc = getattr(self, "care_encoder", None)
        CTX.z = None
        if enc is not None and CTX.active:
            if CTX.clim.shape[0] == x.shape[0]:
                CTX.z = enc(CTX.clim, CTX.loc, CTX.miss)
                CTX.used += 1
            else:   # Ultralytics 내부 점검(FLOPs 계산 등)이 다른 크기로 forward 할 때. 상황 없이 돌린다
                CTX.skipped += 1
        try:
            return super()._predict_once(x, *args, **kwargs)
        finally:
            CTX.z = None


def layer_channels(model: DetectionModel, idx, imgsz: int = 64) -> dict:
    """층 번호 -> 출력 채널 수 (작은 입력으로 한 번 forward 해서 잰다)."""
    out, hooks = {}, []
    for i in idx:
        hooks.append(model.model[i].register_forward_hook(lambda m, a, y, i=i: out.__setitem__(i, int(y.shape[1]))))
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            p = next(model.parameters())
            model(torch.zeros(1, 3, imgsz, imgsz, device=p.device, dtype=p.dtype))
    finally:
        for h in hooks:
            h.remove()
        model.train(was)
    return out


def make_care_model(model: DetectionModel, mods, care_cfg: dict, loc_cards) -> DetectionModel:
    """가중치를 불러온 DetectionModel 을 제자리에서 CARE-YOLO 로 바꾼다. mods 가 비어 있으면 그대로 둔다."""
    mods = [m for m in MODALITIES if m in mods]
    if not mods:
        return model
    idx = [int(i) for i in care_cfg["film_layers"]]
    ch = layer_channels(model, idx)
    model.__class__ = CareDetectionModel
    ed = care_cfg["emb_dims"]
    model.care_encoder = ContextEncoder(
        mods, clim_dim=int(care_cfg["clim_dim"]), clim_hidden=int(care_cfg["clim_hidden"]),
        clim_z=int(care_cfg["clim_z"]), loc_cards=tuple(int(c) for c in loc_cards),
        loc_emb_dims=(int(ed["region"]), int(ed["facility"]), int(ed["part"])), loc_z=int(care_cfg["loc_z"]))
    p = next(model.parameters())
    model.care_encoder.to(device=p.device, dtype=p.dtype)
    for i in idx:
        model.model[i] = FiLMWrap(model.model[i], model.care_encoder.z_dim, ch[i]).to(device=p.device, dtype=p.dtype)
    model.care_info = {"mods": mods, "film_layers": idx, "channels": ch, "z_dim": model.care_encoder.z_dim,
                       "n_params_context": int(sum(q.numel() for q in model.care_encoder.parameters())
                                               + sum(model.model[i].gen.weight.numel() + model.model[i].gen.bias.numel()
                                                     for i in idx))}
    return model


def film_magnitude(model) -> dict:
    """층별 FiLM 생성기 가중치 크기 (0 이면 아직 GPR 단독과 같음)."""
    out = {}
    for m in getattr(model, "model", []):
        if isinstance(m, FiLMWrap):
            out[int(m.i)] = float(m.gen.weight.detach().float().abs().mean())
    return out


# --------------------------------------------------------------------------- 상황 표

class ContextTable:
    """filename -> 기후 벡터 (12) 와 위치 인덱스 (3). s1 산출물 climate/, location/ 을 읽는다.

    위치 값 중 train 에 한 번도 나오지 않은 값은 학습된 임베딩이 없으므로 결측(0)으로 바꾼다 (unseen_to_missing).
    """

    def __init__(self, climate_dir: Path, location_dir: Path, splits=("train", "val", "test"),
                 unseen_to_missing: bool = True):
        import pandas as pd
        clim, loc, names, split_of = [], [], [], []
        for sp in splits:
            ci = pd.read_csv(Path(climate_dir) / f"climate_{sp}_index.csv", dtype=str, keep_default_na=False)
            li = pd.read_csv(Path(location_dir) / f"location_{sp}_index.csv", dtype=str, keep_default_na=False)
            if list(ci["filename"]) != list(li["filename"]):
                raise ValueError(f"{sp} 기후·위치 index 순서가 다릅니다.")
            clim.append(np.load(Path(climate_dir) / f"climate_{sp}.npy").astype(np.float32))
            loc.append(np.load(Path(location_dir) / f"location_{sp}.npy").astype(np.int64))
            names += list(ci["filename"])
            split_of += [sp] * len(ci)
        self.clim = np.concatenate(clim)
        self.loc = np.concatenate(loc)
        split_of = np.asarray(split_of)
        self.unseen = {}
        if unseen_to_missing and "train" in splits:
            tr = split_of == "train"
            for j in range(self.loc.shape[1]):
                seen = set(np.unique(self.loc[tr, j])) - {0}
                bad = ~np.isin(self.loc[:, j], list(seen)) & (self.loc[:, j] != 0)
                self.unseen[j] = {sp: int((bad & (split_of == sp)).sum()) for sp in splits}
                self.loc[bad, j] = 0
        self.row = {fn: i for i, fn in enumerate(names)}
        if len(self.row) != len(names):
            raise ValueError("상황 표에 중복 filename 이 있습니다.")

    def lookup(self, filenames):
        try:
            idx = np.fromiter((self.row[Path(f).name] for f in filenames), dtype=np.int64)
        except KeyError as e:
            raise KeyError(f"상황 표에 없는 이미지: {e}") from None
        return self.clim[idx], self.loc[idx]


def context_batch(table: ContextTable, filenames, mods, device, drop_p: float = 0.0, gen: torch.Generator | None = None,
                  force_missing=(), perm: dict | None = None):
    """CTX.set 에 넣을 텐서. 실험이 쓰지 않는 모달은 결측으로 둔다.

    drop_p      학습 중 모달별 결측 확률 (ModDrop)
    force_missing 평가 때 결측으로 둘 모달 ('clim', 'loc')
    perm        평가 때 순열 검정: {'clim': filename->대체 filename, 'loc': ...}
    """
    names = [Path(f).name for f in filenames]
    clim, loc = table.lookup(names)
    if perm:
        if "clim" in perm:
            clim, _ = table.lookup([perm["clim"][n] for n in names])
        if "loc" in perm:
            _, loc = table.lookup([perm["loc"][n] for n in names])
    B = len(names)
    miss = torch.zeros(B, 2, dtype=torch.bool)
    for j, m in enumerate(MODALITIES):
        if m not in mods or m in force_missing:
            miss[:, j] = True
        elif drop_p > 0:
            miss[:, j] |= torch.rand(B, generator=gen) < drop_p
    return (torch.as_tensor(clim, device=device), torch.as_tensor(loc, device=device), miss.to(device))

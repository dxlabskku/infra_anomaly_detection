import copy
import io

import numpy as np
import pytest
import torch

import care_model as CM
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import ModelEMA

CARE_CFG = {"film_layers": [2, 15, 18, 21], "clim_dim": 12, "clim_hidden": 32, "clim_z": 32, "loc_z": 16,
            "emb_dims": {"region": 8, "facility": 4, "part": 4}}
LOC_CARDS = (17, 5, 5)


@pytest.fixture(scope="module")
def base():
    torch.manual_seed(0)
    m = DetectionModel("yolov8s.yaml", nc=3, verbose=False).eval()
    return m


def ctx(B, seed=0, miss=None):
    g = torch.Generator().manual_seed(seed)
    clim = torch.randn(B, 12, generator=g)
    loc = torch.stack([torch.randint(1, 17, (B,), generator=g), torch.randint(1, 5, (B,), generator=g),
                       torch.randint(1, 5, (B,), generator=g)], 1)
    return clim, loc, torch.zeros(B, 2, dtype=torch.bool) if miss is None else miss


def out0(m, x):
    with torch.no_grad():
        y = m(x)
    return (y[0] if isinstance(y, (list, tuple)) else y)


def make(base, mods=("clim", "loc")):
    return CM.make_care_model(copy.deepcopy(base), mods, CARE_CFG, LOC_CARDS).eval()


def test_identity_at_init(base):
    m = make(base)
    assert isinstance(m, CM.CareDetectionModel)
    x = torch.rand(2, 3, 128, 128)
    CM.CTX.set(*ctx(2))
    try:
        assert torch.allclose(out0(m, x), out0(base, x), atol=1e-5)
    finally:
        CM.CTX.clear()


def test_film_changes_output_and_missing_is_consistent(base):
    m = make(base)
    for w in m.model:
        if isinstance(w, CM.FiLMWrap):
            torch.nn.init.normal_(w.gen.weight, std=0.05)
    x = torch.rand(1, 3, 128, 128).repeat(2, 1, 1, 1)
    clim, loc, miss = ctx(2)
    CM.CTX.set(clim, loc, miss)
    y = out0(m, x)
    assert not torch.allclose(y[0], y[1])          # 같은 영상, 다른 상황 -> 다른 출력
    CM.CTX.set(clim, loc, torch.ones(2, 2, dtype=torch.bool))
    y = out0(m, x)
    assert torch.allclose(y[0], y[1], atol=1e-5)    # 둘 다 결측이면 원래 값과 상관없이 같다
    CM.CTX.clear()


def test_batch_mismatch_is_skipped(base):
    m = make(base)
    for w in m.model:
        if isinstance(w, CM.FiLMWrap):
            torch.nn.init.normal_(w.gen.weight, std=0.05)
    CM.CTX.used = CM.CTX.skipped = 0
    CM.CTX.set(*ctx(4))
    x = torch.rand(1, 3, 64, 64)
    try:
        y = out0(m, x)
        CM.CTX.clear()
        assert torch.allclose(y, out0(m, x), atol=1e-5)
        assert CM.CTX.skipped == 1
    finally:
        CM.CTX.clear()


def test_pickle_deepcopy_and_ema(base):
    m = make(base)
    for w in m.model:
        if isinstance(w, CM.FiLMWrap):
            torch.nn.init.normal_(w.gen.weight, std=0.05)
    buf = io.BytesIO()
    torch.save({"model": m}, buf)
    buf.seek(0)
    m2 = torch.load(buf, weights_only=False)["model"].eval()
    x = torch.rand(2, 3, 64, 64)
    CM.CTX.set(*ctx(2))
    try:
        assert torch.allclose(out0(m, x), out0(m2, x), atol=1e-6)
    finally:
        CM.CTX.clear()
    sd = m.state_dict()
    ptrs = [v.data_ptr() for v in sd.values() if v.is_floating_point() and v.numel()]
    assert len(ptrs) == len(set(ptrs))              # 같은 텐서가 두 키로 들어가 EMA 가 두 번 갱신되는 일이 없다
    ema = ModelEMA(m)
    ema.update(m)
    assert isinstance(ema.ema, CM.CareDetectionModel)


def test_single_modality_dims(base):
    assert make(base, ("clim",)).care_encoder.z_dim == 32
    assert make(base, ("loc",)).care_encoder.z_dim == 16
    assert make(base, ()).__class__ is DetectionModel


def test_film_layer_channels(base):
    ch = make(base).care_info["channels"]
    assert ch == {2: 64, 15: 128, 18: 256, 21: 512}

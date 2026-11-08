import math

import numpy as np
import pytest

import common as C
from s1_prepare import depth_gain, equipment_of, phys3_channels, to_u8


def test_to_u8_range():
    assert to_u8(np.array([-1.0, 0.0, 1.0, -5.0, 5.0])).tolist() == [0, 128, 255, 0, 255]


def test_depth_gain_monotonic_and_clipped():
    g = depth_gain(512, kappa=3.0, y_ref=0.25, clip=(0.25, 4.0))
    assert np.all(np.diff(g) >= 0)
    assert g.min() >= 0.25 and g.max() <= 4.0
    assert abs(g[int(0.25 * 512)] - 1.0) < 0.02


def test_equipment_of():
    assert equipment_of("CT_230813162459_002_100.32.399.389.png") == "CT"
    with pytest.raises(C.PipelineError):
        equipment_of("XX_1.png")


def _layer(img, cy, amp):
    """밝기 127 배경 위에 가로로 긴 반사층(철근 줄)을 그린다. 영상의 약 1% 를 차지한다."""
    y = int(cy * 512)
    img[y - 2:y + 3, :] += amp


def test_phys3_depth_correction_equalizes_attenuated_rebars():
    kappa, y_ref = 4.0, 0.25
    img = np.full((512, 512), 127.0)
    a0 = 100.0
    for cy in (0.15, 0.35):
        _layer(img, cy, a0 * math.exp(-kappa * cy))
    g = np.clip(img, 0, 255).astype(np.uint8)
    out = phys3_channels(g, kappa, y_ref)
    assert out.shape == (512, 512, 3) and out.dtype == np.uint8
    assert np.array_equal(out[..., 0], g)

    def peak(ch, cy, cx):
        y = int(cy * 512)
        return float(np.abs(ch[y - 2:y + 3, cx - 2:cx + 3].astype(float) - 127.5).max())
    shallow2, deep2 = peak(out[..., 1], 0.15, 150), peak(out[..., 1], 0.35, 350)
    shallow3, deep3 = peak(out[..., 2], 0.15, 150), peak(out[..., 2], 0.35, 350)
    assert deep2 < 0.6 * shallow2                    # 상대 진폭만으로는 깊은 철근이 약하게 보인다
    assert abs(deep3 - shallow3) < 0.15 * shallow3   # 깊이 보정 뒤에는 비슷하다

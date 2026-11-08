import numpy as np
import pandas as pd
import torch

import care_model as CM


def _write(tmp, split, names, clim, loc):
    (tmp / "climate").mkdir(exist_ok=True)
    (tmp / "location").mkdir(exist_ok=True)
    np.save(tmp / "climate" / f"climate_{split}.npy", np.asarray(clim, np.float32))
    np.save(tmp / "location" / f"location_{split}.npy", np.asarray(loc, np.int64))
    pd.DataFrame({"filename": names}).to_csv(tmp / "climate" / f"climate_{split}_index.csv", index=False)
    pd.DataFrame({"filename": names}).to_csv(tmp / "location" / f"location_{split}_index.csv", index=False)


def _table(tmp):
    _write(tmp, "train", ["a.png", "b.png"], np.ones((2, 12)), [[1, 1, 1], [2, 2, 1]])
    _write(tmp, "val", ["c.png"], np.zeros((1, 12)), [[5, 1, 2]])          # 지역 5, 부위 2 는 train 에 없음
    _write(tmp, "test", ["d.png"], np.full((1, 12), 2.0), [[2, 3, 0]])     # 시설 3 은 train 에 없음
    return CM.ContextTable(tmp / "climate", tmp / "location")


def test_unseen_location_to_missing(tmp_path):
    t = _table(tmp_path)
    _, loc = t.lookup(["c.png", "d.png", "a.png"])
    assert loc.tolist() == [[0, 1, 0], [2, 0, 0], [1, 1, 1]]
    assert t.unseen[0] == {"train": 0, "val": 1, "test": 0}


def test_context_batch_modes(tmp_path):
    t = _table(tmp_path)
    clim, loc, miss = CM.context_batch(t, ["/x/images/train/a.png", "b.png"], ["clim"], "cpu")
    assert clim.shape == (2, 12) and loc.shape == (2, 3)
    assert miss[:, 0].tolist() == [False, False] and miss[:, 1].tolist() == [True, True]
    _, _, miss = CM.context_batch(t, ["a.png", "b.png"], ["clim", "loc"], "cpu", force_missing=("loc",))
    assert miss.tolist() == [[False, True], [False, True]]
    _, _, miss = CM.context_batch(t, ["a.png", "b.png"], ["clim", "loc"], "cpu", drop_p=1.0,
                                  gen=torch.Generator().manual_seed(0))
    assert miss.all()


def test_context_batch_permutation(tmp_path):
    t = _table(tmp_path)
    clim, loc, _ = CM.context_batch(t, ["a.png", "d.png"], ["clim", "loc"], "cpu",
                                    perm={"clim": {"a.png": "d.png", "d.png": "a.png"}})
    assert clim[:, 0].tolist() == [2.0, 1.0]          # 기후만 바뀌고
    assert loc.tolist() == [[1, 1, 1], [2, 0, 0]]      # 위치는 그대로

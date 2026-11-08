import os
import sys
from pathlib import Path

import pytest
import yaml

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
# care_model 이 ultralytics 를 import 하므로 그 전에 설정 폴더를 작업 폴더 안으로 둔다 (사용자 홈에 설정 파일을 만들지 않게)
(CODE.parent / ".ultralytics" / "Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(CODE.parent / ".ultralytics")
os.environ.setdefault("YOLO_AUTOINSTALL", "False")


@pytest.fixture(scope="session")
def cfg():
    with open(CODE / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)

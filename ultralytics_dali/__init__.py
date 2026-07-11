# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

__version__ = "0.1.0"

import importlib
import os
from typing import TYPE_CHECKING

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
	os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

# ===== DALI patch: 표준 ultralytics에 use_dali cfg 등록 및 dataset/dataloader builder 교체 =====
# 모델 클래스는 아래 __getattr__에서 lazy import되므로, 이 패치가 표준 trainer들의
# `from ultralytics.data import build_dataloader, build_yolo_dataset` 바인딩보다 먼저 실행된다.
# build_dataloader까지 교체해야 _attach_dali_provider가 실제로 호출되어 DALI 디코드가 동작한다.
from ultralytics.cfg import DEFAULT_CFG_DICT, DEFAULT_CFG
import ultralytics.data.build as _std_build
import ultralytics.data as _std_data
from ultralytics_dali.data import build as _dali_build

if "use_dali" not in DEFAULT_CFG_DICT:
	DEFAULT_CFG_DICT["use_dali"] = False
	setattr(DEFAULT_CFG, "use_dali", False)

for _name in ("build_yolo_dataset", "build_grounding", "build_dataloader"):
	setattr(_std_build, _name, getattr(_dali_build, _name))
	if hasattr(_std_data, _name):
		setattr(_std_data, _name, getattr(_dali_build, _name))
del _name
# ===== end DALI patch =====

settings = SETTINGS

MODELS = ("YOLO", "YOLOWorld", "YOLOE", "NAS", "SAM", "FastSAM", "RTDETR")

__all__ = (
	"__version__",
	"ASSETS",
	*MODELS,
	"checks",
	"download",
	"settings",
)

if TYPE_CHECKING:
	# Enable hints for type checkers
	from ultralytics.models import YOLO, YOLOWorld, YOLOE, NAS, SAM, FastSAM, RTDETR  # noqa


def __getattr__(name: str):
	"""Lazy-import model classes on first access."""
	if name in MODELS:
		return getattr(importlib.import_module("ultralytics.models"), name)
	raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
	"""Extend dir() to include lazily available model names for IDE autocompletion."""
	return sorted(set(globals()) | set(MODELS))


if __name__ == "__main__":
	print(__version__)
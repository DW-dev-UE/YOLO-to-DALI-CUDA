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

# ===== DALI patch: 표준 ultralytics에 use_dali cfg 등록 및 dataset builder 교체 =====
from ultralytics.cfg import DEFAULT_CFG_DICT, DEFAULT_CFG
import ultralytics.data.build as _std_build
import ultralytics.data as _std_data
from ultralytics_dali.data import build as _dali_build

if "use_dali" not in DEFAULT_CFG_DICT:
	DEFAULT_CFG_DICT["use_dali"] = False
	setattr(DEFAULT_CFG, "use_dali", False)

_std_build.build_yolo_dataset = _dali_build.build_yolo_dataset
if hasattr(_std_data, "build_yolo_dataset"):
	_std_data.build_yolo_dataset = _dali_build.build_yolo_dataset
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
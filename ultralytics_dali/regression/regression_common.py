from __future__ import annotations

import csv
import importlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml

try:
	import torch
except Exception:
	torch = None


def ensure_repo_on_path(start: str | Path | None = None) -> Path:
	start_path = Path(start or os.getcwd()).resolve()
	candidates = [start_path, *start_path.parents]
	for root in candidates:
		if (root / "ultralytics" / "data" / "build.py").exists():
			root_str = str(root)
			if root_str not in sys.path:
				sys.path.insert(0, root_str)
			return root
		if (root / "data" / "build.py").exists():
			root_str = str(root)
			if root_str not in sys.path:
				sys.path.insert(0, root_str)
			return root
	raise FileNotFoundError("Could not find repo root with data/build.py")


def load_yaml(path: str | Path) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f)
	return data or {}


def save_json(path: str | Path, data: dict) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, ensure_ascii=True)


def seed_all(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	if torch is not None:
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)


def import_string(path: str):
	module_name, attr_name = path.rsplit(".", 1)
	module = importlib.import_module(module_name)
	return getattr(module, attr_name)


def deep_merge(base: dict, override: dict | None) -> dict:
	out = dict(base)
	for k, v in (override or {}).items():
		if isinstance(v, dict) and isinstance(out.get(k), dict):
			out[k] = deep_merge(out[k], v)
		else:
			out[k] = v
	return out


def make_cfg(entry: dict, use_dali: bool):
	ensure_repo_on_path()
	from ultralytics.cfg import IterableSimpleNamespace

	defaults = {
		"imgsz": int(entry.get("imgsz", 640)),
		"rect": bool(entry.get("rect", False)),
		"cache": entry.get("cache", None),
		"single_cls": bool(entry.get("single_cls", False)),
		"fraction": float(entry.get("fraction", 1.0)),
		"classes": entry.get("classes", None),
		"task": entry.get("task", "detect"),
		"use_dali": bool(use_dali),
		"deterministic": bool(entry.get("deterministic", True)),
		"mosaic": float(entry.get("mosaic", 1.0)),
		"mixup": float(entry.get("mixup", 0.0)),
		"cutmix": float(entry.get("cutmix", 0.0)),
		"copy_paste": float(entry.get("copy_paste", 0.0)),
		"degrees": float(entry.get("degrees", 0.0)),
		"translate": float(entry.get("translate", 0.1)),
		"scale": float(entry.get("scale", 0.5)),
		"shear": float(entry.get("shear", 0.0)),
		"perspective": float(entry.get("perspective", 0.0)),
		"fliplr": float(entry.get("fliplr", 0.5)),
		"flipud": float(entry.get("flipud", 0.0)),
		"hsv_h": float(entry.get("hsv_h", 0.015)),
		"hsv_s": float(entry.get("hsv_s", 0.7)),
		"hsv_v": float(entry.get("hsv_v", 0.4)),
		"erasing": float(entry.get("erasing", 0.0)),
		"auto_augment": entry.get("auto_augment", None),
		"mask_ratio": int(entry.get("mask_ratio", 4)),
		"overlap_mask": bool(entry.get("overlap_mask", True)),
		"bgr": float(entry.get("bgr", 0.0)),
		"gpu_augment": bool(entry.get("gpu_augment", use_dali)),
	}
	cfg = deep_merge(defaults, entry.get("cfg_overrides", None))
	return IterableSimpleNamespace(**cfg)


def load_data_for_entry(entry: dict) -> dict:
	data_field = entry.get("data_dict", None)
	if isinstance(data_field, dict):
		return data_field
	data_path = entry.get("data", None)
	if not data_path:
		return {}
	return load_yaml(data_path)


def build_dataset(entry: dict, use_dali: bool):
	ensure_repo_on_path()
	from ultralytics.data.build import build_grounding, build_yolo_dataset

	cfg = make_cfg(entry, use_dali=use_dali)
	batch = int(entry.get("batch", 4))
	mode = entry.get("mode", "train")
	rect = bool(entry.get("rect", False))
	stride = int(entry.get("stride", 32))
	kind = entry.get("kind", "yolo")

	if kind == "grounding":
		return build_grounding(
			cfg=cfg,
			img_path=entry["img_path"],
			json_file=entry["json_file"],
			batch=batch,
			mode=mode,
			rect=rect,
			stride=stride,
			max_samples=int(entry.get("max_samples", 80)),
		)

	data = load_data_for_entry(entry)
	return build_yolo_dataset(
		cfg=cfg,
		img_path=entry["img_path"],
		batch=batch,
		data=data,
		mode=mode,
		rect=rect,
		stride=stride,
		multi_modal=bool(entry.get("multi_modal", False)),
	)


def meta_of(value):
	if torch is not None and isinstance(value, torch.Tensor):
		return {
			"kind": "torch.Tensor",
			"shape": list(value.shape),
			"dtype": str(value.dtype),
			"device": str(value.device),
		}
	if isinstance(value, np.ndarray):
		return {
			"kind": "np.ndarray",
			"shape": list(value.shape),
			"dtype": str(value.dtype),
		}
	if isinstance(value, dict):
		return {k: meta_of(v) for k, v in value.items()}
	if isinstance(value, list):
		return {
			"kind": "list",
			"len": len(value),
			"items": [meta_of(v) for v in value[:8]],
		}
	if isinstance(value, tuple):
		return {
			"kind": "tuple",
			"len": len(value),
			"items": [meta_of(v) for v in value[:8]],
		}
	if hasattr(value, "shape") and hasattr(value, "dtype"):
		return {
			"kind": type(value).__name__,
			"shape": list(value.shape),
			"dtype": str(value.dtype),
		}
	return {
		"kind": type(value).__name__,
		"repr": repr(value)[:120],
	}


def compare_meta(a, b, path: str = "root") -> list[dict]:
	mismatches = []
	if type(a) != type(b):
		mismatches.append({"path": path, "left_type": type(a).__name__, "right_type": type(b).__name__})
		return mismatches

	if isinstance(a, dict):
		left_keys = set(a.keys())
		right_keys = set(b.keys())
		if left_keys != right_keys:
			mismatches.append({
				"path": path,
				"left_keys": sorted(left_keys),
				"right_keys": sorted(right_keys),
			})
		for key in sorted(left_keys & right_keys):
			mismatches.extend(compare_meta(a[key], b[key], f"{path}.{key}"))
		return mismatches

	if isinstance(a, list):
		if len(a) != len(b):
			mismatches.append({"path": path, "left_len": len(a), "right_len": len(b)})
		limit = min(len(a), len(b), 8)
		for i in range(limit):
			mismatches.extend(compare_meta(a[i], b[i], f"{path}[{i}]"))
		return mismatches

	if isinstance(a, tuple):
		if len(a) != len(b):
			mismatches.append({"path": path, "left_len": len(a), "right_len": len(b)})
		limit = min(len(a), len(b), 8)
		for i in range(limit):
			mismatches.extend(compare_meta(a[i], b[i], f"{path}[{i}]"))
		return mismatches

	if a != b:
		mismatches.append({"path": path, "left": a, "right": b})
	return mismatches


def read_csv_rows(path: str | Path) -> list[dict]:
	rows = []
	with open(path, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			rows.append(row)
	return rows


def maybe_float(value):
	try:
		return float(value)
	except Exception:
		return value

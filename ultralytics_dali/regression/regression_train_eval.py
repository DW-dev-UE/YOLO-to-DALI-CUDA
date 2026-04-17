from __future__ import annotations

import argparse
from pathlib import Path

from regression_common import import_string, load_yaml, maybe_float, read_csv_rows, save_json


def extract_run_summary(run_dir: str | Path) -> dict:
	run_dir = Path(run_dir)
	summary = {
		"run_dir": str(run_dir),
		"results_csv": None,
		"rows": 0,
		"last_row": {},
		"loss_keys": {},
		"map_keys": {},
	}
	csv_path = run_dir / "results.csv"
	if not csv_path.exists():
		return summary

	rows = read_csv_rows(csv_path)
	summary["results_csv"] = str(csv_path)
	summary["rows"] = len(rows)
	if not rows:
		return summary

	last = {k: maybe_float(v) for k, v in rows[-1].items()}
	summary["last_row"] = last
	for k, v in last.items():
		if "loss" in k.lower():
			summary["loss_keys"][k] = v
		if "map" in k.lower() or "fitness" in k.lower():
			summary["map_keys"][k] = v
	return summary


def build_train_kwargs(entry: dict, use_dali: bool, project_dir: str, run_name: str, device: str, epochs: int) -> dict:
	kwargs = {
		"data": entry["data"],
		"epochs": int(entry.get("epochs", epochs)),
		"imgsz": int(entry.get("imgsz", 640)),
		"batch": int(entry.get("batch", 16)),
		"device": entry.get("device", device),
		"project": project_dir,
		"name": run_name,
		"workers": int(entry.get("workers", 4 if not use_dali else 0)),
		"use_dali": bool(use_dali),
		"plots": bool(entry.get("plots", False)),
		"val": bool(entry.get("val", True)),
	}
	for key, value in (entry.get("train_overrides", {}) or {}).items():
		kwargs[key] = value
	return kwargs


def run_one(entry: dict, out_dir: Path, device: str, epochs: int) -> dict:
	name = entry.get("name", entry.get("task", "task"))
	runner_path = entry.get("runner", "ultralytics.YOLO")
	runner_cls = import_string(runner_path)
	model_arg = entry["model"]

	result = {
		"name": name,
		"task": entry.get("task", "detect"),
		"runs": {},
		"ok": True,
	}

	for label, use_dali in (("cv2", False), ("dali", True)):
		run_name = f"{name}_{label}"
		project_dir = str(out_dir / name)
		kwargs = build_train_kwargs(entry, use_dali, project_dir, run_name, device, epochs)
		try:
			model = runner_cls(model_arg)
			train_result = model.train(**kwargs)
			save_dir = None
			if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir"):
				save_dir = str(model.trainer.save_dir)
			elif hasattr(train_result, "save_dir"):
				save_dir = str(train_result.save_dir)
			else:
				save_dir = str(Path(project_dir) / run_name)
			result["runs"][label] = {
				"save_dir": save_dir,
				"summary": extract_run_summary(save_dir),
				"train_kwargs": kwargs,
			}
		except Exception as e:
			result["runs"][label] = {
				"error": f"{type(e).__name__}: {e}",
				"train_kwargs": kwargs,
			}
			result["ok"] = False

	return result


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--spec", required=True)
	parser.add_argument("--out-dir", required=True)
	parser.add_argument("--out-json", required=True)
	parser.add_argument("--device", default="0")
	parser.add_argument("--epochs", type=int, default=1)
	args = parser.parse_args()

	spec = load_yaml(args.spec)
	tasks = spec.get("tasks", [])
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	results = {
		"spec": str(Path(args.spec).resolve()),
		"out_dir": str(out_dir.resolve()),
		"tasks": [],
	}

	for entry in tasks:
		try:
			results["tasks"].append(run_one(entry, out_dir, args.device, args.epochs))
		except Exception as e:
			results["tasks"].append({
				"name": entry.get("name", entry.get("task", "task")),
				"task": entry.get("task", "detect"),
				"ok": False,
				"error": f"{type(e).__name__}: {e}",
			})

	save_json(args.out_json, results)
	print(f"wrote: {args.out_json}")

	failed = [x for x in results["tasks"] if not x.get("ok", False)]
	if failed:
		raise SystemExit(1)


if __name__ == "__main__":
	main()

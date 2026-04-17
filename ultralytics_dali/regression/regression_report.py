from __future__ import annotations

import argparse

from regression_common import load_yaml, save_json


def compare_task(task_result: dict, max_map_drop: float, max_loss_rise: float) -> dict:
	name = task_result.get("name", "task")
	cv2_run = task_result.get("runs", {}).get("cv2", {})
	dali_run = task_result.get("runs", {}).get("dali", {})
	out = {
		"name": name,
		"task": task_result.get("task", "detect"),
		"ok": True,
		"map_deltas": {},
		"loss_deltas": {},
		"errors": [],
	}

	if cv2_run.get("error"):
		out["errors"].append({"run": "cv2", "error": cv2_run["error"]})
		out["ok"] = False
	if dali_run.get("error"):
		out["errors"].append({"run": "dali", "error": dali_run["error"]})
		out["ok"] = False
	if out["errors"]:
		return out

	cv2_summary = cv2_run.get("summary", {})
	dali_summary = dali_run.get("summary", {})
	for key, cv2_value in cv2_summary.get("map_keys", {}).items():
		dali_value = dali_summary.get("map_keys", {}).get(key, None)
		if isinstance(cv2_value, (int, float)) and isinstance(dali_value, (int, float)):
			delta = dali_value - cv2_value
			out["map_deltas"][key] = delta
			if delta < -abs(max_map_drop):
				out["ok"] = False

	for key, cv2_value in cv2_summary.get("loss_keys", {}).items():
		dali_value = dali_summary.get("loss_keys", {}).get(key, None)
		if isinstance(cv2_value, (int, float)) and isinstance(dali_value, (int, float)):
			delta = dali_value - cv2_value
			out["loss_deltas"][key] = delta
			if delta > abs(max_loss_rise):
				out["ok"] = False

	return out


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--train-json", required=True)
	parser.add_argument("--parity-json", default=None)
	parser.add_argument("--out-json", required=True)
	parser.add_argument("--max-map-drop", type=float, default=0.01)
	parser.add_argument("--max-loss-rise", type=float, default=0.05)
	args = parser.parse_args()

	train_data = load_yaml(args.train_json) if args.train_json.endswith((".yaml", ".yml")) else None
	if train_data is None:
		import json
		with open(args.train_json, "r", encoding="utf-8") as f:
			train_data = json.load(f)

	parity_data = None
	if args.parity_json:
		if args.parity_json.endswith((".yaml", ".yml")):
			parity_data = load_yaml(args.parity_json)
		else:
			import json
			with open(args.parity_json, "r", encoding="utf-8") as f:
				parity_data = json.load(f)

	results = {
		"train_json": args.train_json,
		"parity_json": args.parity_json,
		"tasks": [],
		"ok": True,
	}

	parity_by_name = {}
	if parity_data:
		for item in parity_data.get("tasks", []):
			parity_by_name[item.get("name", "task")] = item

	for item in train_data.get("tasks", []):
		comp = compare_task(item, args.max_map_drop, args.max_loss_rise)
		parity_item = parity_by_name.get(comp["name"], None)
		if parity_item is not None:
			comp["parity_ok"] = bool(parity_item.get("ok", False))
			if not comp["parity_ok"]:
				comp["ok"] = False
		results["tasks"].append(comp)
		if not comp["ok"]:
			results["ok"] = False

	save_json(args.out_json, results)
	print(f"wrote: {args.out_json}")
	if not results["ok"]:
		raise SystemExit(1)


if __name__ == "__main__":
	main()

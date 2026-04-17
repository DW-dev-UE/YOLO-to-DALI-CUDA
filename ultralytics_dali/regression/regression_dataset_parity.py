from __future__ import annotations

import argparse
from pathlib import Path

from regression_common import build_dataset, compare_meta, load_yaml, meta_of, save_json, seed_all


def pick_indices(length: int, count: int) -> list[int]:
	return list(range(min(length, count)))


def run_entry(entry: dict, samples: int, batch_size: int, seed: int) -> dict:
	name = entry.get("name", entry.get("task", "task"))
	result = {
		"name": name,
		"task": entry.get("task", "detect"),
		"kind": entry.get("kind", "yolo"),
		"sample_checks": [],
		"batch_check": {},
		"ok": True,
	}

	ds_cv2 = build_dataset(entry, use_dali=False)
	ds_dali = build_dataset(entry, use_dali=True)
	indices = entry.get("sample_indices", None) or pick_indices(min(len(ds_cv2), len(ds_dali)), samples)

	for idx in indices:
		seed_all(seed)
		a = ds_cv2[idx]
		seed_all(seed)
		b = ds_dali[idx]
		ma = meta_of(a)
		mb = meta_of(b)
		mismatches = compare_meta(ma, mb)
		result["sample_checks"].append({
			"index": idx,
			"left": ma,
			"right": mb,
			"mismatches": mismatches,
			"ok": len(mismatches) == 0,
		})
		if mismatches:
			result["ok"] = False

	batch_indices = entry.get("batch_indices", None)
	if batch_indices is None:
		batch_indices = pick_indices(min(len(ds_cv2), len(ds_dali)), min(batch_size, int(entry.get("batch", batch_size))))

	seed_all(seed)
	batch_a = ds_cv2.collate_fn([ds_cv2[i] for i in batch_indices])
	seed_all(seed)
	batch_b = ds_dali.collate_fn([ds_dali[i] for i in batch_indices])
	meta_a = meta_of(batch_a)
	meta_b = meta_of(batch_b)
	mismatches = compare_meta(meta_a, meta_b)
	result["batch_check"] = {
		"indices": batch_indices,
		"left": meta_a,
		"right": meta_b,
		"mismatches": mismatches,
		"ok": len(mismatches) == 0,
	}
	if mismatches:
		result["ok"] = False

	return result


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--spec", required=True)
	parser.add_argument("--out", required=True)
	parser.add_argument("--samples", type=int, default=8)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--seed", type=int, default=123)
	args = parser.parse_args()

	spec = load_yaml(args.spec)
	tasks = spec.get("tasks", [])
	results = {
		"spec": str(Path(args.spec).resolve()),
		"tasks": [],
	}

	for entry in tasks:
		try:
			results["tasks"].append(run_entry(entry, args.samples, args.batch_size, args.seed))
		except Exception as e:
			results["tasks"].append({
				"name": entry.get("name", entry.get("task", "task")),
				"task": entry.get("task", "detect"),
				"ok": False,
				"error": f"{type(e).__name__}: {e}",
			})

	save_json(args.out, results)

	failed = [x for x in results["tasks"] if not x.get("ok", False)]
	print(f"wrote: {args.out}")
	print(f"tasks: {len(results['tasks'])}")
	print(f"failed: {len(failed)}")
	if failed:
		raise SystemExit(1)


if __name__ == "__main__":
	main()

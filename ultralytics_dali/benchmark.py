# cv2 vs DALI dataloader throughput A/B benchmark
#
# Usage:
#   python -m ultralytics_dali.benchmark --data path/to/data.yaml --imgsz 1024 --batch 16 --workers 8
#
# Measures images/sec of the full train-mode dataloader (decode + augment + collate)
# with the identical config, once per backend. Detect-family tasks only
# (detect / segment / pose / obb).

from __future__ import annotations

import argparse
import gc
import time

import ultralytics_dali  # noqa: F401 - applies the DALI patch to standard ultralytics

from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER, colorstr

from ultralytics_dali.data import build as B


def _infinite_batches(loader):
	while True:
		yield from loader


def bench_backend(use_dali: bool, args, data) -> dict | None:
	name = "dali" if use_dali else "cv2"
	cfg = get_cfg()
	cfg.task = args.task
	cfg.imgsz = args.imgsz
	cfg.use_dali = use_dali

	dataset = None
	loader = None
	try:
		dataset = B.build_yolo_dataset(cfg, data[args.split], args.batch, data, mode="train")
		loader = B.build_dataloader(dataset, args.batch, args.workers, shuffle=True)
	except (RuntimeError, ImportError) as e:
		LOGGER.error(colorstr("benchmark: ") + f"{name} backend unavailable: {e}")
		return None

	provider = type(getattr(dataset, "image_provider", None)).__name__
	LOGGER.info(colorstr("benchmark: ") + f"{name}: provider={provider}, batches/epoch={len(loader)}")

	it = _infinite_batches(loader)
	for _ in range(args.warmup):
		next(it)

	n_img = 0
	t0 = time.perf_counter()
	for _ in range(args.num_batches):
		batch = next(it)
		n_img += int(batch["img"].shape[0])
	dt = time.perf_counter() - t0

	prov = getattr(dataset, "image_provider", None)
	if prov is not None and hasattr(prov, "close"):
		prov.close()
	del loader, dataset
	gc.collect()

	return {"name": name, "provider": provider, "img_s": n_img / dt, "s_batch": dt / args.num_batches, "images": n_img}


def main() -> None:
	ap = argparse.ArgumentParser(description="cv2 vs DALI dataloader throughput benchmark")
	ap.add_argument("--data", required=True, help="dataset yaml (detect/segment/pose/obb)")
	ap.add_argument("--task", default="detect", choices=["detect", "segment", "pose", "obb"])
	ap.add_argument("--split", default="train", choices=["train", "val"])
	ap.add_argument("--imgsz", type=int, default=640)
	ap.add_argument("--batch", type=int, default=16)
	ap.add_argument("--workers", type=int, default=8)
	ap.add_argument("--num-batches", type=int, default=50, help="timed batches per backend")
	ap.add_argument("--warmup", type=int, default=5, help="untimed warmup batches per backend")
	ap.add_argument("--backends", default="cv2,dali", help="comma list, e.g. 'cv2', 'dali', 'cv2,dali'")
	args = ap.parse_args()

	data = check_det_dataset(args.data)
	results = []
	for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
		r = bench_backend(backend == "dali", args, data)
		if r:
			results.append(r)

	print(f"\n{'backend':<10}{'provider':<22}{'img/s':>10}{'s/batch':>10}{'images':>9}")
	for r in results:
		print(f"{r['name']:<10}{r['provider']:<22}{r['img_s']:>10.1f}{r['s_batch']:>10.3f}{r['images']:>9}")
	if len(results) == 2:
		base, test = results[0], results[1]
		print(f"\n{test['name']} vs {base['name']}: {test['img_s'] / base['img_s']:.2f}x throughput")


if __name__ == "__main__":
	main()

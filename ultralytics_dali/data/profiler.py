import os
import time
from collections import defaultdict
from contextlib import contextmanager
from threading import RLock


class StageProfiler:
	def __init__(self):
		self.enabled = os.getenv("YOLO_PROFILE", "0") == "1"
		self.data = defaultdict(lambda: {"count": 0, "total": 0.0, "max": 0.0})
		self.lock = RLock()
		self.print_every = int(os.getenv("YOLO_PROFILE_PRINT_EVERY", "200"))

	def add(self, name: str, dt: float):
		if not self.enabled:
			return

		need_report = False
		with self.lock:
			item = self.data[name]
			item["count"] += 1
			item["total"] += dt
			if dt > item["max"]:
				item["max"] = dt

			total_count = self.data["__global__"]["count"] + 1
			self.data["__global__"]["count"] = total_count
			if self.print_every > 0 and total_count % self.print_every == 0:
				need_report = True

		if need_report:
			self.report(prefix="[PROFILE] ")

	@contextmanager
	def measure(self, name: str):
		if not self.enabled:
			yield
			return
		t0 = time.perf_counter()
		try:
			yield
		finally:
			self.add(name, time.perf_counter() - t0)

	def report(self, prefix: str = ""):
		if not self.enabled:
			return
		with self.lock:
			items = []
			for k, v in self.data.items():
				if k == "__global__":
					continue
				count = v["count"]
				total = v["total"]
				maxv = v["max"]
				avg = total / count if count else 0.0
				items.append((total, k, count, avg, maxv))
			items.sort(reverse=True)

		print(prefix + "stage summary")
		for total, name, count, avg, maxv in items:
			print(
				f"{prefix}{name:32s} "
				f"count={count:8d} "
				f"total={total * 1000.0:10.3f}ms "
				f"avg={avg * 1000.0:10.3f}ms "
				f"max={maxv * 1000.0:10.3f}ms"
			)
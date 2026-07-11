# DALI batch decoder - GPU decode + aspect-preserving resize

from __future__ import annotations
from .profiler import PROFILE

from typing import List, Tuple
from collections import OrderedDict
from threading import RLock

import cv2
import numpy as np

from ultralytics.utils import LOGGER

try:
	from nvidia.dali import fn, pipeline_def, types
	DALI_AVAILABLE = True
except ImportError:
	DALI_AVAILABLE = False


DALI_SUPPORTED_EXT = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "pnm", "ppm", "pgm", "pbm"}


class DaliBatchDecoder:
	def __init__(self, batch_size: int, max_size: int | None = None, channels: int = 3,
	             device_id: int = 0, num_threads: int = 4):
		if not DALI_AVAILABLE:
			raise ImportError("nvidia-dali not installed")
		self.batch_size = int(batch_size)
		self.max_size = None if max_size is None else int(max_size)
		self.channels = int(channels)
		self.device_id = int(device_id)
		self.num_threads = int(num_threads)
		self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
		self._fallback_warned = False
		self._build()

	def _build(self) -> None:
		max_size = self.max_size
		out_type = types.GRAY if self.channels == 1 else types.BGR

		# prefetch_queue_depth must stay 1: run() feeds exactly one batch per
		# iteration, and a deeper queue would demand that many feeds before the
		# first run() can return.
		@pipeline_def(
			batch_size=self.batch_size,
			num_threads=self.num_threads,
			device_id=self.device_id,
			prefetch_queue_depth=1,
			exec_async=True,
			exec_pipelined=True,
		)
		def _pipe():
			jpegs = fn.external_source(name="jpegs", device="cpu", no_copy=False)
			shapes = fn.peek_image_shape(jpegs)
			imgs = fn.decoders.image(jpegs, device="mixed", output_type=out_type)
			if max_size is not None and max_size > 0:
				imgs = fn.resize(
					imgs,
					resize_longer=max_size,
					interp_type=types.INTERP_LINEAR,
					antialias=False,
				)
			return imgs, shapes

		self.pipeline = _pipe()
		self.pipeline.build()

	@staticmethod
	def _is_dali_supported(path: str) -> bool:
		ext = path.rpartition(".")[-1].lower()
		return ext in DALI_SUPPORTED_EXT

	def _cv2_decode_one(self, path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
		with PROFILE.measure("cv2.decode_one.total"):
			with PROFILE.measure("cv2.decode_one.imread"):
				im = cv2.imread(path, self.cv2_flag)
			if im is None:
				raise FileNotFoundError(f"Image not readable: {path}")
			h0, w0 = im.shape[:2]
			if self.max_size is not None and self.max_size > 0:
				r = self.max_size / max(h0, w0)
				if r != 1:
					w = min(int(round(w0 * r)), self.max_size)
					h = min(int(round(h0 * r)), self.max_size)
					with PROFILE.measure("cv2.decode_one.resize"):
						im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
			if im.ndim == 2:
				im = im[..., None]
			return np.ascontiguousarray(im), (h0, w0)

	def run(self, file_paths: List[str]) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
		"""Decode images, returning (images, original (h, w) shapes) in input order."""
		actual = len(file_paths)

		with PROFILE.measure("dali.run.total"):
			dali_slots = []
			fallback_slots = []
			for i, p in enumerate(file_paths):
				if self._is_dali_supported(p):
					dali_slots.append(i)
				else:
					fallback_slots.append(i)

			result_imgs: List[np.ndarray] = [None] * actual
			result_shapes: List[Tuple[int, int]] = [None] * actual

			if fallback_slots:
				with PROFILE.measure("dali.run.cv2_fallback_unsupported"):
					for i in fallback_slots:
						with PROFILE.measure("dali.run.cv2_fallback_one"):
							img, shp = self._cv2_decode_one(file_paths[i])
						result_imgs[i] = img
						result_shapes[i] = shp

			# Fixed-size chunks so the single pipeline built for this decoder serves
			# every request; the final short chunk is padded by repeating its last item.
			for start in range(0, len(dali_slots), self.batch_size):
				chunk = dali_slots[start:start + self.batch_size]
				self._run_chunk(file_paths, chunk, result_imgs, result_shapes)

			return result_imgs, result_shapes

	def _run_chunk(self, file_paths: List[str], chunk: List[int],
	               result_imgs: List[np.ndarray], result_shapes: List[Tuple[int, int]]) -> None:
		byte_batch = []

		with PROFILE.measure("dali.run.file_read_total"):
			for i in chunk:
				with PROFILE.measure("dali.run.file_read_one"):
					with open(file_paths[i], "rb") as fp:
						byte_batch.append(np.frombuffer(fp.read(), dtype=np.uint8))

		with PROFILE.measure("dali.run.pad_to_batch"):
			while len(byte_batch) < self.batch_size:
				byte_batch.append(byte_batch[-1])

		try:
			with PROFILE.measure("dali.run.feed_input"):
				self.pipeline.feed_input("jpegs", byte_batch)

			with PROFILE.measure("dali.run.pipeline_run"):
				out_imgs, out_shapes = self.pipeline.run()

			with PROFILE.measure("dali.run.as_cpu"):
				imgs_cpu = out_imgs.as_cpu() if hasattr(out_imgs, "as_cpu") else out_imgs
				shapes_cpu = out_shapes.as_cpu() if hasattr(out_shapes, "as_cpu") else out_shapes

			with PROFILE.measure("dali.run.unpack_outputs"):
				for k, i in enumerate(chunk):
					with PROFILE.measure("dali.run.unpack_one"):
						arr = np.asarray(imgs_cpu.at(k))
						if arr.ndim == 2:
							arr = arr[..., None]
						result_imgs[i] = np.ascontiguousarray(arr)

						shp = np.asarray(shapes_cpu.at(k))
						result_shapes[i] = (int(shp[0]), int(shp[1]))

		except Exception as e:
			if not self._fallback_warned:
				self._fallback_warned = True
				LOGGER.warning(f"DALI: batch decode failed, falling back to cv2 for this batch: {e}")
			with PROFILE.measure("dali.run.cv2_fallback_batch_exception"):
				for i in chunk:
					with PROFILE.measure("dali.run.cv2_fallback_one"):
						img, shp = self._cv2_decode_one(file_paths[i])
					result_imgs[i] = img
					result_shapes[i] = shp


class DaliImageProvider:
	"""Lazy, worker-local DALI image provider.

	The object is intentionally picklable so it can be shipped to spawned DataLoader workers.
	Each worker lazily builds its own DALI decoders the first time it needs them.

	Cache entries and decode() results are (image, (h0, w0)) pairs: when max_size is set
	the pipeline resizes on GPU, so the original shape can no longer be read off the image.
	"""

	def __init__(self, channels: int = 3, device_id: int = 0, num_threads: int = 2,
	             max_cached: int = 4096, batch_size: int = 16, max_size: int | None = None):
		if not DALI_AVAILABLE:
			raise ImportError("nvidia-dali not installed")
		self.channels = int(channels)
		self.device_id = int(device_id)
		self.num_threads = int(num_threads)
		self.max_cached = int(max_cached)
		self.batch_size = max(1, int(batch_size))
		self.max_size = None if max_size is None else int(max_size)
		self._decoders = {}
		self._cache = OrderedDict()
		self._lock = RLock()

	def __getstate__(self):
		return {
			"channels": self.channels,
			"device_id": self.device_id,
			"num_threads": self.num_threads,
			"max_cached": self.max_cached,
			"batch_size": self.batch_size,
			"max_size": self.max_size,
		}

	def __setstate__(self, state):
		self.channels = int(state["channels"])
		self.device_id = int(state["device_id"])
		self.num_threads = int(state["num_threads"])
		self.max_cached = int(state["max_cached"])
		self.batch_size = max(1, int(state.get("batch_size", 16)))
		self.max_size = None if state.get("max_size") is None else int(state["max_size"])
		self._decoders = {}
		self._cache = OrderedDict()
		self._lock = RLock()

	@staticmethod
	def _ensure_cuda_ready() -> None:
		try:
			import torch
			if torch.cuda.is_available() and not torch.cuda.is_initialized():
				torch.cuda.init()
		except Exception:
			pass

	def _get_decoder(self, batch_size: int) -> DaliBatchDecoder:
		bs = int(batch_size)
		dec = self._decoders.get(bs)
		if dec is None:
			self._ensure_cuda_ready()
			dec = DaliBatchDecoder(
				batch_size=bs,
				max_size=self.max_size,
				channels=self.channels,
				device_id=self.device_id,
				num_threads=self.num_threads,
			)
			self._decoders[bs] = dec
		return dec

	def decode(self, f: str) -> tuple[np.ndarray | None, tuple[int, int] | None]:
		imgs, shapes = self._get_decoder(1).run([f])
		return imgs[0], shapes[0]

	def prime(self, indices: list[int], paths: list[str]) -> None:
		if not indices:
			return

		missing_indices = []
		missing_paths = []
		with self._lock:
			for idx, path in zip(indices, paths):
				key = int(idx)
				if key not in self._cache:
					missing_indices.append(key)
					missing_paths.append(path)

		if not missing_indices:
			return

		# Always decode through the fixed dataloader-batch-size decoder (run() chunks
		# internally), so each worker builds at most two pipelines: batch and single.
		imgs, shapes = self._get_decoder(self.batch_size).run(missing_paths)

		with self._lock:
			for idx, img, shp in zip(missing_indices, imgs, shapes):
				self._cache[idx] = (img, shp)
				self._cache.move_to_end(idx)

			while len(self._cache) > self.max_cached:
				self._cache.popitem(last=False)

	def take(self, index: int, f: str | None = None) -> tuple[np.ndarray, tuple[int, int]] | None:
		key = int(index)
		with self._lock:
			return self._cache.pop(key, None)

	def close(self) -> None:
		with self._lock:
			self._cache.clear()
		self._decoders.clear()

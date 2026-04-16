# DALI batch decoder - GPU decode + aspect-preserving resize

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

try:
	from nvidia.dali import fn, pipeline_def, types
	DALI_AVAILABLE = True
except ImportError:
	DALI_AVAILABLE = False


DALI_SUPPORTED_EXT = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "pnm", "ppm", "pgm", "pbm"}


class DaliBatchDecoder:
	"""Batch decode + resize on GPU. Variable output shape per sample."""

	def __init__(self, batch_size: int, max_size: int, channels: int = 3,
	             device_id: int = 0, num_threads: int = 4):
		if not DALI_AVAILABLE:
			raise ImportError("nvidia-dali not installed")
		self.batch_size = int(batch_size)
		self.max_size = int(max_size)
		self.channels = int(channels)
		self.device_id = int(device_id)
		self.num_threads = int(num_threads)
		self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
		self._build()

	def _build(self) -> None:
		max_size = self.max_size
		out_type = types.GRAY if self.channels == 1 else types.BGR

		@pipeline_def(batch_size=self.batch_size, num_threads=self.num_threads,
		              device_id=self.device_id, prefetch_queue_depth=1,
		              exec_async=False, exec_pipelined=False)
		def _pipe():
			jpegs = fn.external_source(name="jpegs", device="cpu", no_copy=False)
			shapes = fn.peek_image_shape(jpegs)
			imgs = fn.decoders.image(jpegs, device="mixed", output_type=out_type)
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
		im = cv2.imread(path, self.cv2_flag)
		if im is None:
			raise FileNotFoundError(f"Image not readable: {path}")
		h0, w0 = im.shape[:2]
		r = self.max_size / max(h0, w0)
		if r != 1:
			w = min(int(round(w0 * r)), self.max_size)
			h = min(int(round(h0 * r)), self.max_size)
			im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
		if im.ndim == 2:
			im = im[..., None]
		return np.ascontiguousarray(im), (h0, w0)

	def run(self, file_paths: List[str]) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
		"""Decode a batch. Unsupported or corrupt files transparently fall back to cv2."""
		actual = len(file_paths)

		# split into dali-eligible and fallback indices
		dali_slots = []
		fallback_slots = []
		for i, p in enumerate(file_paths):
			if self._is_dali_supported(p):
				dali_slots.append(i)
			else:
				fallback_slots.append(i)

		result_imgs: List[np.ndarray] = [None] * actual
		result_shapes: List[Tuple[int, int]] = [None] * actual

		# cv2 fallback for unsupported extensions
		for i in fallback_slots:
			img, shp = self._cv2_decode_one(file_paths[i])
			result_imgs[i] = img
			result_shapes[i] = shp

		# dali path (if any remain)
		if dali_slots:
			byte_batch = []
			for i in dali_slots:
				with open(file_paths[i], "rb") as fp:
					byte_batch.append(np.frombuffer(fp.read(), dtype=np.uint8))
			while len(byte_batch) < self.batch_size:
				byte_batch.append(byte_batch[-1])

			try:
				self.pipeline.feed_input("jpegs", byte_batch)
				out_imgs, out_shapes = self.pipeline.run()
				imgs_cpu = out_imgs.as_cpu() if hasattr(out_imgs, "as_cpu") else out_imgs
				shapes_cpu = out_shapes.as_cpu() if hasattr(out_shapes, "as_cpu") else out_shapes

				for k, i in enumerate(dali_slots):
					arr = np.asarray(imgs_cpu.at(k))
					if arr.ndim == 2:
						arr = arr[..., None]
					result_imgs[i] = np.ascontiguousarray(arr)
					shp = np.asarray(shapes_cpu.at(k))
					result_shapes[i] = (int(shp[0]), int(shp[1]))
			except Exception:
				# whole-batch dali failure: fall back per-sample to cv2
				for i in dali_slots:
					img, shp = self._cv2_decode_one(file_paths[i])
					result_imgs[i] = img
					result_shapes[i] = shp

		return result_imgs, result_shapes
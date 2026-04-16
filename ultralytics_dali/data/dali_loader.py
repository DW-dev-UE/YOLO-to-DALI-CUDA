# DALI-backed dataloader. Drop-in replacement for InfiniteDataLoader when use_dali=True

from __future__ import annotations

import math
from typing import Iterator, List

import torch
from torch.utils.data import distributed

from .dali_pipeline import DALI_AVAILABLE, DaliBatchDecoder


class DaliDataLoader:
	"""Iterates dataset with batched DALI decode. Not infinite - trainer reinits per epoch."""

	def __init__(self, dataset, batch_size: int, shuffle: bool = True, rank: int = -1,
	             drop_last: bool = False, num_threads: int = 4):
		self.dataset = dataset
		self.batch_size = int(batch_size)
		self.shuffle = shuffle
		self.rank = rank
		self.drop_last = drop_last

		device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
		self.decoder = DaliBatchDecoder(
			batch_size=self.batch_size,
			max_size=dataset.imgsz,
			channels=getattr(dataset, "channels", 3),
			device_id=device_id,
			num_threads=num_threads,
		)

		if rank != -1:
			if shuffle:
				self.sampler = distributed.DistributedSampler(dataset, shuffle=True)
			else:
				# lazy import to avoid circular dependency
				from .build import ContiguousDistributedSampler
				self.sampler = ContiguousDistributedSampler(dataset, batch_size=batch_size, shuffle=False)
		else:
			self.sampler = None

	def __len__(self) -> int:
		n = len(self.sampler) if self.sampler is not None else len(self.dataset)
		if self.drop_last:
			return n // self.batch_size
		return math.ceil(n / self.batch_size)

	def set_epoch(self, epoch: int) -> None:
		if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
			self.sampler.set_epoch(epoch)

	def reset(self) -> None:
		"""Compat with InfiniteDataLoader.reset(). No-op here."""
		return

	def __iter__(self) -> Iterator:
		if self.sampler is not None:
			indices = list(self.sampler)
		elif self.shuffle:
			indices = torch.randperm(len(self.dataset)).tolist()
		else:
			indices = list(range(len(self.dataset)))

		for start in range(0, len(indices), self.batch_size):
			batch_idx = indices[start:start + self.batch_size]
			if self.drop_last and len(batch_idx) < self.batch_size:
				break
			yield self._assemble_batch(batch_idx)

	def _assemble_batch(self, batch_idx: List[int]):
		ds = self.dataset
		paths = [ds.im_files[i] for i in batch_idx]
		decoded, ori_shapes = self.decoder.run(paths)

		samples = []
		for j, idx in enumerate(batch_idx):
			img = decoded[j]
			if img.ndim == 2:
				img = img[..., None]
			h0, w0 = ori_shapes[j]

			ds.ims[idx] = img
			ds.im_hw0[idx] = (h0, w0)
			ds.im_hw[idx] = img.shape[:2]

			if ds.augment and idx not in ds.buffer:
				ds.buffer.append(idx)
				if 1 < len(ds.buffer) >= ds.max_buffer_length:
					old = ds.buffer.pop(0)
					if getattr(ds, "cache", None) != "ram":
						ds.ims[old] = None
						ds.im_hw0[old] = None
						ds.im_hw[old] = None

			samples.append(ds[idx])

		return ds.collate_fn(samples)


def dali_available() -> bool:
	return DALI_AVAILABLE
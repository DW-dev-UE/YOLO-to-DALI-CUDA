# DALI-backed dataloader. Drop-in replacement for InfiniteDataLoader when use_dali=True

from __future__ import annotations
from .profiler import PROFILE

import math
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Iterator, List

import torch
from torch.utils.data import distributed

from ultralytics.utils import LOGGER

from .dali_pipeline import DALI_AVAILABLE, DaliImageProvider


class DaliPrefetchLoader:
	"""Parity-first loader that only prefetches decoded images into the provider cache."""

	def __init__(
		self,
		dataset,
		batch_size: int,
		shuffle: bool = True,
		rank: int = -1,
		drop_last: bool = False,
		num_threads: int = 4,
	):
		self.dataset = dataset
		self.batch_size = int(batch_size)
		self.shuffle = shuffle
		self.rank = rank
		self.drop_last = drop_last
		self.num_threads = int(num_threads)

		if rank != -1:
			if shuffle:
				self.sampler = distributed.DistributedSampler(dataset, shuffle=True)
			else:
				from .build import ContiguousDistributedSampler
				self.sampler = ContiguousDistributedSampler(dataset, batch_size=batch_size, shuffle=False)
		else:
			self.sampler = None

		provider = getattr(dataset, "image_provider", None)
		if provider is None or not hasattr(provider, "prime"):
			if not hasattr(dataset, "set_image_provider"):
				raise RuntimeError("Dataset does not support set_image_provider()")
			device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
			dataset.set_image_provider(
				DaliImageProvider(
					channels=getattr(dataset, "channels", 3),
					device_id=device_id,
					num_threads=self.num_threads,
				)
			)
			provider = dataset.image_provider

		self.provider = provider
		self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dali_prefetch")

	def __len__(self) -> int:
		n = len(self.sampler) if self.sampler is not None else len(self.dataset)
		if self.drop_last:
			return n // self.batch_size
		return math.ceil(n / self.batch_size)

	def set_epoch(self, epoch: int) -> None:
		if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
			self.sampler.set_epoch(epoch)

	def reset(self) -> None:
		return

	def __del__(self):
		try:
			self.executor.shutdown(wait=False, cancel_futures=True)
		except Exception:
			pass

	def _ordered_indices(self) -> List[int]:
		if self.sampler is not None:
			return list(self.sampler)
		if self.shuffle:
			return torch.randperm(len(self.dataset)).tolist()
		return list(range(len(self.dataset)))

	def _make_batches(self, indices: List[int]) -> List[List[int]]:
		out = []
		for start in range(0, len(indices), self.batch_size):
			batch_idx = indices[start:start + self.batch_size]
			if self.drop_last and len(batch_idx) < self.batch_size:
				break
			out.append(batch_idx)
		return out

	def _prime_batch(self, batch_idx: List[int]) -> None:
		ds = self.dataset
		with PROFILE.measure("loader.prefetch.prime_batch"):
			paths = [ds.im_files[i] for i in batch_idx]
			self.provider.prime(batch_idx, paths)

	def _submit_prime(self, batch_idx: List[int]) -> Future:
		with PROFILE.measure("loader.prefetch.submit"):
			return self.executor.submit(self._prime_batch, list(batch_idx))

	def __iter__(self) -> Iterator:
		ds = self.dataset
		indices = self._ordered_indices()
		batches = self._make_batches(indices)

		next_future = None
		if batches:
			next_future = self._submit_prime(batches[0])

		for bi, batch_idx in enumerate(batches):
			with PROFILE.measure("loader.prefetch.wait"):
				if next_future is not None:
					next_future.result()

			if bi + 1 < len(batches):
				next_future = self._submit_prime(batches[bi + 1])
			else:
				next_future = None

			samples = []
			with PROFILE.measure("loader.prefetch_batch.total"):
				for idx in batch_idx:
					with PROFILE.measure("loader.prefetch_batch.ds_getitem"):
						samples.append(ds[idx])

				with PROFILE.measure("loader.prefetch_batch.collate"):
					yield ds.collate_fn(samples)


def dali_available() -> bool:
	return DALI_AVAILABLE
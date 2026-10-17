"""Batched, CPU-only data fetching for the shared FBSA/backprop convergence loop.

Worker classes live here without importing runtime ANNpt settings, so spawned
workers can unpickle them when the benchmark configures its modules in memory.
"""

import torch as pt


class TabularBatchDataset(pt.utils.data.Dataset):
	def __init__(self, dataset, classFieldName):
		self.classFieldName = classFieldName
		#with_format returns a view; prepared Arrow files and their metadata stay intact.
		self.dataset = dataset.with_format("numpy", columns=["features", classFieldName],
			output_all_columns=False)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		batch = self.dataset[index]
		return (pt.as_tensor(batch["features"], dtype=pt.float32),
			pt.as_tensor(batch[self.classFieldName], dtype=pt.long))

	def __getitems__(self, indices):
		#One Arrow fetch and tensor conversion per batch, including shuffled batches.
		#Sequential evaluation can use an Arrow slice instead of gathering each row.
		if(len(indices) and all(index == indices[0] + offset for offset, index in enumerate(indices))):
			indices = slice(indices[0], indices[0] + len(indices))
		return self[indices]


def collateTabularBatch(batch):
	#The batched dataset already returns two tensors; do not split and stack rows.
	return batch


def createConvergenceDataLoader(dataset, *, batchSize, shuffle, seed, numWorkers,
		pinMemory, classFieldName=None):
	kwargs = {}
	if(classFieldName is not None):
		dataset = TabularBatchDataset(dataset, classFieldName)
		kwargs["collate_fn"] = collateTabularBatch
	#Separate sampler and worker RNGs keep minibatch order independent of worker
	#count and persistent-worker reuse, and do not consume the model's RNG stream.
	if(shuffle):
		sampler = pt.utils.data.RandomSampler(dataset, generator=pt.Generator().manual_seed(seed))
	else:
		sampler = pt.utils.data.SequentialSampler(dataset)
	if(numWorkers > 0):
		#Workers only touch CPU data; spawning avoids inheriting an active CUDA context.
		kwargs["multiprocessing_context"] = "spawn"
	return pt.utils.data.DataLoader(dataset, batch_size=batchSize, sampler=sampler,
		drop_last=False, num_workers=numWorkers, pin_memory=pinMemory,
		persistent_workers=numWorkers > 0, generator=pt.Generator().manual_seed(seed), **kwargs)

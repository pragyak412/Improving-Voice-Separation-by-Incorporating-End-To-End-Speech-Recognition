import torch
from torch.utils.data import Dataset
import os
from scipy.io import wavfile
import numpy as np

import config


class dataLoader(Dataset):

	def __init__(self, type_='train'):

		"""
		Get all the file names.
		Depending upon type of dataLoader
		"""

		self.type_ = type_
		self.allFileNames = sorted(os.listdir(config.dataSetPath[self.type_]))

	def mix(self, speaker1, speaker2):

		return (speaker1 + speaker2)/2

	def normalise(self, audio):

		audio = audio - audio.min()
		audio = 2 * audio / audio.max()
		audio = audio - audio.mean()
		return audio

	def __getitem__(self, item):

		"""
		Generate two random numbers (0, len(self.allFileNames)) which must not be the same
		Load the two audio files, check dtype
		divide by max int16
		Mix the speakers
		Convert to required dtype
		:param item:
		:return:
			mixture: dtype = torch.FloatTensor, shape = [24000]
			target: dtype = torch.FloatTensor, shape = [2, 24000]
			size: dtype = torch.FloatTensor, shape = [1], value=24000
		"""

		while True:
			sample1, sample2 = np.random.choice(self.allFileNames, 2, replace=False)

			fs, sample1 = wavfile.read(config.dataSetPath[self.type_]+'/'+sample1)
			fs, sample2 = wavfile.read(config.dataSetPath[self.type_]+'/'+sample2)

			if sample1.shape[0] > fs*3 and sample2.shape[0] > fs*3:
				break

		start1, start2 = np.random.randint(0, sample1.shape[0]-fs*3), np.random.randint(0, sample2.shape[0]-fs*3)
		end1, end2 = start1+fs*3, start2+fs*3

		sample1, sample2 = sample1[start1:end1], sample2[start2:end2]

		max_value = np.iinfo(sample1.dtype).max

		sample1, sample2 = sample1.astype(np.float32), sample2.astype(np.float32)
		sample1, sample2 = sample1 / max_value, sample2 / max_value

		sample1 = self.normalise(sample1)
		sample2 = self.normalise(sample2)

		mixture = self.normalise(sample1 + sample2)

		sample1, sample2, mixture = torch.from_numpy(sample1), torch.from_numpy(sample2), torch.from_numpy(mixture)
		target = torch.cat([sample1.unsqueeze(0), sample2.unsqueeze(0)], dim=0)

		return mixture, target, 24000

	def __len__(self):

		return config.iterations[self.type_]


if __name__ == "__main__":

	from main import seed
	from torch.utils.data import DataLoader
	import config

	seed()

	data_loader = dataLoader('train')
	data_loader = DataLoader(
		data_loader, num_workers=config.numWorkers['train'], batch_size=config.batchSize['train'], shuffle=True
	)

	for i in data_loader:
		pass
from torch.utils.data import Dataset
import config
import os
import numpy as np
from scipy.io.wavfile import read


def normalise(audio):

	audio = audio - audio.min()
	maxAudio = audio.max()

	if maxAudio is None:
		return None
	audio = 2 * audio / maxAudio
	audio = audio - audio.mean()
	return audio


class AVSpeech(Dataset):

	def __init__(self, type_):

		self.type_ = type_

		self.base_audio_path = config.dataset['AVSpeech']['base_audio_path'][self.type_]

		all_file_names = np.array([i[:-4] for i in sorted(os.listdir(self.base_audio_path))])

		choice_files_names = np.random.choice(all_file_names, size=config.iterations[self.type_])

		shuffled = choice_files_names.copy()

		np.random.shuffle(shuffled)

		self.speaker_1 = []
		self.speaker_2 = []

		for i, j in zip(choice_files_names, shuffled):

			if i != j:
				self.speaker_1.append(i)
				self.speaker_2.append(j)
			else:
				speaker_1, speaker_2 = np.random.choice(all_file_names, size=2, replace=False)
				self.speaker_1.append(speaker_1)
				self.speaker_2.append(speaker_2)

	def __getitem__(self, item):

		while True:

			audio_1 = read(self.base_audio_path + '/' + self.speaker_1[item]+'.wav')[1]/np.iinfo(np.int16).max
			audio_2 = read(self.base_audio_path + '/' + self.speaker_2[item]+'.wav')[1]/np.iinfo(np.int16).max

			if audio_1.shape[0] > 8000 * 3 and audio_2.shape[0] > 8000 * 3:

				start1 = np.random.randint(0, len(audio_1) - 24000)
				start2 = np.random.randint(0, len(audio_2) - 24000)
				end1, end2 = start1 + 24000, start2 + 24000
				audio_1, audio_2 = audio_1[start1:end1], audio_2[start2:end2]
				audio_1 = normalise(audio_1)

				if audio_1 is None:
					continue

				audio_2 = normalise(audio_2)

				if audio_2 is None:
					continue

				mixture = normalise(audio_1 + audio_2)

				return \
					mixture.astype(np.float32), \
					np.concatenate([audio_1[None], audio_2[None]], axis=0).astype(np.float32), \
					[self.speaker_1[item], self.speaker_2[item]]
			else:

				item = np.random.randint(len(self.speaker_1))

	def __len__(self):

		return config.iterations[self.type_]


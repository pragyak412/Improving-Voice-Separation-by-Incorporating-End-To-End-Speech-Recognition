from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import os
from scipy.io.wavfile import write
import numpy as np
import torch
from shutil import copyfile

from dataloader import dataLoader
from conv_tasnet import ConvTasNet
from pit_criterion import cal_loss
import config


def save(mixture, target, output, no):

	mixture = mixture.data.cpu().numpy()
	target = target.data.cpu().numpy()
	output = output.data.cpu().numpy()

	batch_size = output.shape[0]

	base = config.synthesis + '/' + str(no) + '/'

	os.makedirs(base, exist_ok=True)

	for i in range(batch_size):

		speaker1 = output[i, 0] - np.mean(output[i, 0])
		speaker2 = output[i, 1] - np.mean(output[i, 1])

		speaker1 = speaker1/np.max(np.abs(speaker1))
		speaker2 = speaker2/np.max(np.abs(speaker2))

		os.makedirs(base + str(i), exist_ok=True)

		write(base + str(i) + '/targetSpeaker1.wav', 8000, (target[i, 0]*np.iinfo(np.int16).max).astype(np.int16))
		write(base + str(i) + '/targetSpeaker2.wav', 8000, (target[i, 1]*np.iinfo(np.int16).max).astype(np.int16))

		write(base + str(i) + '/mixture.wav', 8000, (mixture[i] * np.iinfo(np.int16).max).astype(np.int16))

		write(base + str(i) + '/outputSpeaker1.wav', 8000, (speaker1 * np.iinfo(np.int16).max).astype(np.int16))
		write(base + str(i) + '/outputSpeaker2.wav', 8000, (speaker2 * np.iinfo(np.int16).max).astype(np.int16))


def test(model, data_loader):

	model.eval()
	all_loss = []

	iterator = tqdm(data_loader)

	with torch.no_grad():

		for no, (mixture, target, size) in enumerate(iterator):

			mixture = mixture.cuda()
			target = target.cuda()
			size = size.cuda()

			output = model(mixture)

			loss, max_snr, estimate_source, reorder_estimate_source = cal_loss(target, output, size)

			loss = loss / config.optimizer_iteration

			all_loss.append(loss.item() * config.optimizer_iteration)

			if (no + 1) % config.periodic_output == 0:
				save(mixture, target, output, no)

			iterator.set_description(
				'L: {0:.3f}| Avg L: {1:.3f}'.format(
					loss.item() * config.optimizer_iteration,
					np.array(all_loss)[-min(1000, len(all_loss)):].mean(),
				)
			)

	return all_loss


def main(modelPath):

	copyfile('config.py', config.basePath + '/config.py')

	data_loader = dataLoader('test')
	data_loader = DataLoader(
		data_loader, num_workers=config.numWorkers['test'], batch_size=config.batchSize['test'], shuffle=True
	)

	model = ConvTasNet(N=256, L=20, B=256, H=512, P=3, X=8, R=4, C=2).cuda()
	model = nn.DataParallel(model)

	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])

	print('Total number of trainable parameters: ', params)

	savedModel = torch.load(modelPath)

	model.load_state_dict(savedModel['state_dict'])

	print('Loaded the model: {0}'.format(modelPath))

	allLoss = test(model, data_loader)

	print('Average Loss: {0:.6f}'.format(np.mean(allLoss)))


if __name__ == "__main__":

	import sys

	main(modelPath=sys.argv[1])

from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import os
from scipy.io.wavfile import write
import numpy as np
import torch
import matplotlib.pyplot as plt
from shutil import copyfile

from dataloader import dataLoader
from conv_tasnet import ConvTasNet
from pit_criterion import cal_loss
from utils import init_fn
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
		target[i, 0] = target[i, 0]/np.max(np.abs(target[i, 0]))
		target[i, 1] = target[i, 1]/np.max(np.abs(target[i, 1]))
		mixture[i] = mixture[i]/np.max(np.abs(mixture[i]))

		os.makedirs(base + str(i), exist_ok=True)

		write(base + str(i) + '/targetSpeaker1.wav', 8000, (target[i, 0]*np.iinfo(np.int16).max).astype(np.int16))
		write(base + str(i) + '/targetSpeaker2.wav', 8000, (target[i, 1]*np.iinfo(np.int16).max).astype(np.int16))

		write(base + str(i) + '/mixture.wav', 8000, (mixture[i] * np.iinfo(np.int16).max).astype(np.int16))

		write(base + str(i) + '/outputSpeaker1.wav', 8000, (speaker1 * np.iinfo(np.int16).max).astype(np.int16))
		write(base + str(i) + '/outputSpeaker2.wav', 8000, (speaker2 * np.iinfo(np.int16).max).astype(np.int16))


def train(model, data_loader, optimizer, all_loss):

	model.train()
	optimizer.zero_grad()

	iterator = tqdm(data_loader)

	def change_lr(no_i):
		for i in config.lr:
			if i == no_i:
				print('Learning Rate Changed to ', config.lr[i])
				for param_group in optimizer.param_groups:
					param_group['lr'] = config.lr[i]

	for no, (mixture, target, size) in enumerate(iterator):

		change_lr(no)

		if config.preTrained:
			if no < config.startingNo:
				continue

		mixture = mixture.cuda()
		target = target.cuda()
		size = size.cuda()

		output = model(mixture)

		loss, max_snr, estimate_source, reorder_estimate_source = cal_loss(target, output, size)

		loss = loss / config.optimizer_iteration

		all_loss.append(loss.item() * config.optimizer_iteration)

		loss.backward()

		if (no + 1) % config.optimizer_iteration == 0:
			optimizer.step()
			optimizer.zero_grad()

		if (no + 1) % config.periodic_output == 0:
			save(mixture, target, output, no)

		iterator.set_description(
			'L: {0:.3f}| Avg L: {1:.3f}'.format(
				loss.item() * config.optimizer_iteration,
				np.array(all_loss)[-min(1000, len(all_loss)):].mean(),
			)
		)

		if (no + 1) % config.periodic_save == 0:

			torch.save(
				{
					'state_dict': model.state_dict(),
					'optimizer': optimizer.state_dict()
				}, config.basePath + '/' + str((no + 1) // config.optimizer_iteration) + '_model.pkl')

			np.save(config.basePath + '/loss_plot_training.npy', all_loss)
			plt.plot(all_loss)
			plt.savefig(config.basePath + '/loss_plot_training.png')
			plt.clf()

	return all_loss


def main():

	os.system('cp -r ../ConvTasNet "{0}"'.format(config.basePath+'/savedCode'))

	data_loader = dataLoader('train')
	data_loader = DataLoader(
		data_loader, num_workers=config.numWorkers['train'], batch_size=config.batchSize['train'], 
		shuffle=True, worker_init_fn=init_fn,
	)

	model = ConvTasNet(N=512, L=20, B=512, H=512, P=3, X=8, R=4, C=2).cuda()
	model = nn.DataParallel(model)

	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])

	print('Total number of trainable parameters: ', params)

	optimizer = Adam(model.parameters(), config.lr[1])

	if config.preTrained:

		savedModel = torch.load(config.preTrainedModel)
		model.load_state_dict(savedModel['state_dict'])
		optimizer.load_state_dict(savedModel['optimizer'])
		all_loss = np.load(config.preTrainedLossPath).tolist()

		print('Loaded preTrained model')
	else:

		all_loss = []

	all_loss = train(model, data_loader, optimizer, all_loss)

	torch.save(
		{
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict()
		}, config.basePath + '/final_model.pkl')

	np.save(config.basePath + '/loss_plot_training.npy', all_loss)
	plt.plot(all_loss)
	plt.savefig(config.basePath + '/loss_plot_training.png')
	plt.clf()


if __name__ == "__main__":

	main()

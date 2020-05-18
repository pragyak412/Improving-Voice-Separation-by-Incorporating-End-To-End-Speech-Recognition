from conv_tasnet import ConvTasNet
import torch
from tqdm import tqdm
import config
from pit_criterion import SISNRPIT
from dataloader import AVSpeech
from torch.utils.data import DataLoader
import os
import numpy as np
from scipy.io.wavfile import write
from dataloader import normalise
from torch.nn import DataParallel


def saving(currentNo, estimated, target, mixture, iteration=0):

	estimated = estimated.data.cpu().numpy()
	target = target.data.cpu().numpy()
	mixture = mixture.data.cpu().numpy()

	base = config.temporary_save_path['test'] + '/' + str(currentNo) + '/' + str(iteration)
	os.makedirs(base, exist_ok=True)

	for i in range(estimated.shape[0]):

		os.makedirs(base+ '/' + str(i), exist_ok=True)

		target[i, 0] = normalise(target[i, 0])
		target[i, 1] = normalise(target[i, 1])

		estimated[i, 0] = normalise(estimated[i, 0])
		estimated[i, 1] = normalise(estimated[i, 1])

		mixture[i] = normalise(mixture[i])

		write(
			base + '/' + str(i) + '/' + 'target_0.wav', 8000,
			(target[i, 0]*np.iinfo(np.int16).max).astype(np.int16))
		write(
			base + '/' + str(i) + '/' + 'target_1.wav', 8000,
			(target[i, 1]*np.iinfo(np.int16).max).astype(np.int16))

		write(
			base + '/' + str(i) + '/' + 'estimated_0.wav', 8000,
			(estimated[i, 0]*np.iinfo(np.int16).max).astype(np.int16))
		write(
			base + '/' + str(i) + '/' + 'estimated_1.wav', 8000,
			(estimated[i, 1]*np.iinfo(np.int16).max).astype(np.int16))

		write(
			base + '/' + str(i) + '/' + 'mixture.wav', 8000,
			(mixture[i]*np.iinfo(np.int16).max).astype(np.int16))


def test(cur_test, model, dataloader, loss_func):

	model.eval()
	iterator = tqdm(dataloader)

	all_loss = []

	with torch.no_grad():

		for no, (mixture, target, path_i) in enumerate(iterator):

			if config.use_cuda:
				mixture = mixture.cuda()
				target = target.cuda()

			separated = model(mixture)

			loss = loss_func(separated, target)

			all_loss.append(loss.item())

			iterator.set_description('Average Loss: '+str(np.array(all_loss).mean()))

			if no % config.periodic_synthesis_test == 0 and no != 0:
				saving(currentNo=cur_test, estimated=separated, target=target, mixture=mixture, iteration=no)

	return all_loss


def main():

	os.system('cp -r ../ConvTasNet "{0}"'.format(config.basePath+'/savedCode'))


	model = DataParallel(ConvTasNet(C=2))
	dataloader = AVSpeech('test')
	dataloader = DataLoader(dataloader, batch_size=config.batchsize['test'], num_workers=config.num_workers['test'], worker_init_fn=init_fn)
	loss_func = SISNRPIT()

	if config.use_cuda:
		model = model.cuda()


	config.pretrained_test = [
		'',
	]

	for cur_test in config.pretrained_test:

		print('Currently working on: ', cur_test.split('/')[-1])

		model.load_state_dict(torch.load(cur_test)['model_state_dict'])

		total_loss = test(cur_test.split('/')[-1].split('.')[0], model, dataloader, loss_func)

		torch.cuda.empty_cache()

		print('Average Loss for ', cur_test.split('/')[-1], 'is: ', np.mean(total_loss))

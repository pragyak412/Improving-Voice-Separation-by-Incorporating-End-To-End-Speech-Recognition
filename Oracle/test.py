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


def init_fn(worker_id):

	"""
	Function to make the pytorch dataloader deterministic
	:param worker_id: id of the parallel worker
	:return:
	"""

	np.random.seed(0 + worker_id)


def saving(estimated, target, mixture, iteration=0):

	estimated = estimated.data.cpu().numpy()
	target = target.data.cpu().numpy()
	mixture = mixture.data.cpu().numpy()

	os.makedirs(config.temporary_save_path['test'] + '/' + str(iteration), exist_ok=True)

	for i in range(estimated.shape[0]):

		os.makedirs(config.temporary_save_path['test'] + '/' + str(iteration) + '/' + str(i), exist_ok=True)

		target[i, 0] = normalise(target[i, 0])
		target[i, 1] = normalise(target[i, 1])

		estimated[i, 0] = normalise(estimated[i, 0])
		estimated[i, 1] = normalise(estimated[i, 1])

		mixture[i] = normalise(mixture[i])

		write(
			config.temporary_save_path['test'] + '/' + str(iteration) + '/' + str(i) + '/' + 'target_0.wav', 8000,
			(target[i, 0]*np.iinfo(np.int16).max).astype(np.int16))
		write(
			config.temporary_save_path['test'] + '/' + str(iteration) + '/' + str(i) + '/' + 'target_1.wav', 8000,
			(target[i, 1]*np.iinfo(np.int16).max).astype(np.int16))

		write(
			config.temporary_save_path['test'] + '/' + str(iteration) + '/' + str(i) + '/' + 'estimated_0.wav', 8000,
			(estimated[i, 0]*np.iinfo(np.int16).max).astype(np.int16))
		write(
			config.temporary_save_path['test'] + '/' + str(iteration) + '/' + str(i) + '/' + 'estimated_1.wav', 8000,
			(estimated[i, 1]*np.iinfo(np.int16).max).astype(np.int16))

		write(
			config.temporary_save_path['test'] + '/' + str(iteration) + '/' + str(i) + '/' + 'mixture.wav', 8000,
			(mixture[i]*np.iinfo(np.int16).max).astype(np.int16))


def test(model, dataloader):

	model.eval()
	
	iterator = tqdm(dataloader)

	all_loss = []
	loss_func = SISNRPIT()

	with torch.no_grad():

		for no, (mixture, target, path_i) in enumerate(iterator):

			if config.use_cuda:
				mixture = mixture.cuda()
				target = target.cuda()
			
			separated = model(mixture, target)

			loss = loss_func(separated, target)

			all_loss.append(loss.item())

			iterator.set_description('Average Loss: '+str(np.array(all_loss).mean()))

			if no % config.periodic_synthesis == 0 and no != 0:
				saving(estimated=separated, target=target, mixture=mixture, iteration=no)

	return all_loss


def main():

	os.system('cp -r ../ConvTasNet "{0}"'.format(config.basePath+'/savedCode'))

	model = DataParallel(ConvTasNet(C=2, test_with_asr=True))
	dataloader = AVSpeech('test')
	dataloader = DataLoader(
		dataloader, batch_size=config.batchsize['test'], num_workers=config.num_workers['test'], worker_init_fn=init_fn)

	if config.use_cuda:
		model = model.cuda()

	config.pretrained_test = [
		'/home/SharedData/Pragya/Experiments/Oracle/2020-05-20 15:23:34.411560/116662.pth'
	]

	for cur_test in config.pretrained_test:

		print('Currently working on: ', cur_test.split('/')[-1])

		model.load_state_dict(torch.load(cur_test)['model_state_dict'])

		total_loss = test(model, dataloader)

		torch.cuda.empty_cache()

		print('Average Loss for ', cur_test.split('/')[-1], 'is: ', np.mean(total_loss))

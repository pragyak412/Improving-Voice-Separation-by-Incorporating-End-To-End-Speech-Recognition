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
import matplotlib.pyplot as plt
from shutil import copyfile
from torch.nn import DataParallel
import random


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

	os.makedirs(config.temporary_save_path['train'] + '/' + str(iteration), exist_ok=True)

	for i in range(estimated.shape[0]):

		os.makedirs(config.temporary_save_path['train'] + '/' + str(iteration) + '/' + str(i), exist_ok=True)

		target[i, 0] = normalise(target[i, 0])
		target[i, 1] = normalise(target[i, 1])

		estimated[i, 0] = normalise(estimated[i, 0])
		estimated[i, 1] = normalise(estimated[i, 1])

		mixture[i] = normalise(mixture[i])

		write(config.temporary_save_path['train'] + '/' + str(iteration) + '/' + str(i) + '/' + 'target_0.wav', 8000, (target[i, 0]*np.iinfo(np.int16).max).astype(np.int16))
		write(config.temporary_save_path['train'] + '/' + str(iteration) + '/' + str(i) + '/' + 'target_1.wav', 8000, (target[i, 1]*np.iinfo(np.int16).max).astype(np.int16))

		write(config.temporary_save_path['train'] + '/' + str(iteration) + '/' + str(i) + '/' + 'estimated_0.wav', 8000, (estimated[i, 0]*np.iinfo(np.int16).max).astype(np.int16))
		write(config.temporary_save_path['train'] + '/' + str(iteration) + '/' + str(i) + '/' + 'estimated_1.wav', 8000, (estimated[i, 1]*np.iinfo(np.int16).max).astype(np.int16))

		write(config.temporary_save_path['train'] + '/' + str(iteration) + '/' + str(i) + '/' + 'mixture.wav', 8000, (mixture[i]*np.iinfo(np.int16).max).astype(np.int16))

					
def train(model, dataloader, optimizer, loss_func, all_loss):

	def change_lr(no):

		for i in config.lr:
			if i == no:
				print('Learning Rate Changed to ', config.lr[i])
				for param_group in optimizer.param_groups:
					param_group['lr'] = config.lr[i]

	model.train()
	optimizer.zero_grad()
	iterator = tqdm(dataloader)

	if all_loss is None:
		all_loss = []
	
	base_loss = []
	to_mean = np.zeros([1000])
	base_mean = np.zeros([1000])

	for no, (mixture, target, path_i) in enumerate(iterator):

		change_lr(no)

		if config.pretrained:
			if no <= config.start:
				to_mean[no % 1000] = all_loss[no]
				continue

		if config.use_cuda:
			mixture = mixture.cuda()
			target = target.cuda()

		loss_base = loss_func(mixture.unsqueeze(1).repeat(1, 2, 1), target)

		separated = model(mixture)

		loss = loss_func(separated, target)

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		all_loss.append(loss.item())
		base_loss.append(loss_base.item())

		to_mean[no % 1000] = loss.item()
		base_mean[no % 1000] = loss_base.item()

		if no < 1000:
			end = no + 1
		else:
			end = 1000

		loss_mean = int(to_mean[0:end].mean()*1000000)/1000000
		base_loss_mean = int(base_mean[0:end].mean()*1000000)/1000000
		improvement = -int((loss_mean - base_loss_mean)*1000000)/1000000

		iterator.set_description(
			'Average Loss: '+str(loss_mean) +
			' | Base Loss: '+str(base_loss_mean) +
			' | Improvement: '+str(improvement))

		if no % config.periodic_synthesis == 0 and no != 0:
			saving(estimated=separated, target=target, mixture=mixture, iteration=no)

		if no % config.periodic_checkpoint == 0 and no != 0:
			torch.save({
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'iterations': no,
			}, config.model_save_path + '/' + str(no) + '.pth')

			np.save(config.model_save_path + '/Loss.npy', all_loss)
			plt.plot(all_loss)
			plt.savefig(config.model_save_path + '/Loss.png')
			plt.clf()


def main():

	os.system('cp -r ../ConvTasNet "{0}"'.format(config.basePath+'/savedCode'))

	model = DataParallel(ConvTasNet(C=2))

	print('Total Parameters: ', sum(p.numel() for p in model.parameters()))
	
	dataloader = AVSpeech('train')
	
	loss_func = SISNRPIT()

	if config.use_cuda:
		model = model.cuda()

	optimizer = torch.optim.Adam(model.parameters(), lr=config.lr[1])

	if config.pretrained:
		saved_model = torch.load(config.pretrained_train)
		model.load_state_dict(saved_model['model_state_dict'])
		optimizer.load_state_dict(saved_model['optimizer_state_dict'])
		saved_loss = np.load(config.loss_path).tolist()
	else:
		saved_loss = None

	dataloader = DataLoader(dataloader, batch_size=config.batchsize['train'], num_workers=config.num_workers['train'], worker_init_fn=init_fn)

	train(model, dataloader, optimizer, loss_func, saved_loss)

if __name__ == "__main__":

	main()

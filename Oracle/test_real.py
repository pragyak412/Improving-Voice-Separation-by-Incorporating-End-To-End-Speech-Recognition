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
            (target[i, 0] * np.iinfo(np.int16).max).astype(np.int16))
        write(
            config.temporary_save_path['test'] + '/' + str(iteration) + '/' + str(i) + '/' + 'target_1.wav', 8000,
            (target[i, 1] * np.iinfo(np.int16).max).astype(np.int16))

        write(
            config.temporary_save_path['test'] + '/' + str(iteration) + '/' + str(i) + '/' + 'estimated_0.wav', 8000,
            (estimated[i, 0] * np.iinfo(np.int16).max).astype(np.int16))
        write(
            config.temporary_save_path['test'] + '/' + str(iteration) + '/' + str(i) + '/' + 'estimated_1.wav', 8000,
            (estimated[i, 1] * np.iinfo(np.int16).max).astype(np.int16))

        write(
            config.temporary_save_path['test'] + '/' + str(iteration) + '/' + str(i) + '/' + 'mixture.wav', 8000,
            (mixture[i] * np.iinfo(np.int16).max).astype(np.int16))


def test(convtasnet_audio_without_asr_model, convtasnet_audio_with_asr_model, dataloader, loss_func):
    convtasnet_audio_without_asr_model.eval()
    convtasnet_audio_with_asr_model.eval()
    iterator = tqdm(dataloader)

    all_loss = []

    with torch.no_grad():

        for no, (mixture, target, path_i) in enumerate(iterator):

            if config.use_cuda:
                mixture = mixture.cuda()
                target = target.cuda()

            separated_initial = convtasnet_audio_without_asr_model(mixture)
            separated = convtasnet_audio_with_asr_model(mixture, separated_initial)

            loss = loss_func(separated, target)

            all_loss.append(loss.item())

            iterator.set_description('Average Loss: ' + str(np.array(all_loss).mean()))

            if no % config.periodic_synthesis == 0 and no != 0:
                saving(estimated=separated, target=target, mixture=mixture, iteration=no)

    return all_loss


def main():

    os.system('cp -r ../ConvTasNet "{0}"'.format(config.basePath+'/savedCode'))

    convtasnet_audio_with_asr_model = DataParallel(ConvTasNet(C=2, test_with_asr=True)).cuda()

    convtasnet_audio_without_asr_model = DataParallel(ConvTasNet(C=2, asr_addition=False)).cuda()
    dataloader = AVSpeech('test')
    dataloader = DataLoader(dataloader, batch_size=config.batchsize['test'], num_workers=config.num_workers['test'],
                            worker_init_fn=init_fn)
    loss_func = SISNRPIT()

    convtasnet_model = config.convtasnet_audio_model
    convtasnet_asr_model = [
        '/home/SharedData/Pragya/Experiments/Oracle/2020-05-20 15:23:34.411560/116662.pth'
    ]

    for conv_asr_test in convtasnet_asr_model:
        print('Currently working convtasnet on: ', convtasnet_model.split('/')[-1])
        print('Currently working E2ESpeechSaparation on: ', conv_asr_test.split('/')[-1])

        convtasnet_audio_without_asr_model.load_state_dict(
            torch.load(convtasnet_model)['model_state_dict'])
        convtasnet_audio_with_asr_model.load_state_dict(
            torch.load(conv_asr_test)['model_state_dict'])

        total_loss = test(convtasnet_audio_without_asr_model, convtasnet_audio_with_asr_model, dataloader, loss_func)

        torch.cuda.empty_cache()

        print('Average Loss for ', conv_asr_test.split('/')[-1], 'is: ', np.mean(total_loss))

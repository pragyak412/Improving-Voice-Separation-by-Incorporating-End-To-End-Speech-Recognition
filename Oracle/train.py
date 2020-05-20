import os
import random
from scipy.io.wavfile import write
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch


from conv_tasnet import ConvTasNet
from dataloader import AVSpeech
from dataloader import normalise
from pit_criterion import SISNR
import config


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

        write(config.temporary_save_path['train'] + '/' + str(iteration) + '/' + str(i) + '/' + 'target_0.wav', 8000,
              (target[i, 0] * np.iinfo(np.int16).max).astype(np.int16))
        write(config.temporary_save_path['train'] + '/' + str(iteration) + '/' + str(i) + '/' + 'target_1.wav', 8000,
              (target[i, 1] * np.iinfo(np.int16).max).astype(np.int16))

        write(config.temporary_save_path['train'] + '/' + str(iteration) + '/' + str(i) + '/' + 'estimated_0.wav', 8000,
              (estimated[i, 0] * np.iinfo(np.int16).max).astype(np.int16))
        write(config.temporary_save_path['train'] + '/' + str(iteration) + '/' + str(i) + '/' + 'estimated_1.wav', 8000,
              (estimated[i, 1] * np.iinfo(np.int16).max).astype(np.int16))

        write(config.temporary_save_path['train'] + '/' + str(iteration) + '/' + str(i) + '/' + 'mixture.wav', 8000,
              (mixture[i] * np.iinfo(np.int16).max).astype(np.int16))


def train(all_loss, dataloader, model, optimizer, loss_func):
    model.train()
    optimizer.zero_grad()
    iterator = tqdm(dataloader)

    def change_lr(no):
        for i in config.lr:
            if i == no:
                print('Learning Rate Changed to ', config.lr[i])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.lr[i]

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

        separated = model(mixture, target)

        loss = loss_func(separated, target) / config.optimizer_steps

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        all_loss.append(loss.item() * config.optimizer_steps)
        base_loss.append(loss_base.item())

        to_mean[no % 1000] = loss.item() * config.optimizer_steps
        base_mean[no % 1000] = loss_base.item()

        if no < 1000:
            end = no + 1
        else:
            end = 1000

        loss_mean = int(to_mean[0:end].mean() * 1000000) / 1000000
        base_loss_mean = int(base_mean[0:end].mean() * 1000000) / 1000000
        improvement = -int((loss_mean - base_loss_mean) * 1000000) / 1000000

        iterator.set_description(
            'Average Loss: ' + str(loss_mean) +
            ' | Base Loss: ' + str(base_loss_mean) +
            ' | Improvement: ' + str(improvement))

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

    if (len(iterator) + 1) % config.optimizer_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return model, optimizer


def get_initial_model_optimizer():
    """
    Loading pretrained model of convtasnet and ASR, for domain translation of asr's features
    to convtasnet, Domain translation block is used
    """
    from ETESpeechRecognition.model import E2E as ASR
    from domainTranslation import DomainTranslation
    import ETESpeechRecognition.config as asrConfig

    # loading convtasnet model
    trained_convtasnet_audio_model = torch.load(config.convtasnet_audio_model, map_location=torch.device('cpu'))

    convtasnet_audio_with_asr_model = DataParallel(ConvTasNet(C=2))

    model_state_dict = trained_convtasnet_audio_model['model_state_dict']

    # adding random weights to model for new block addition
    model_state_dict['module.separator.network.0.gamma'] = torch.cat(
        [model_state_dict['module.separator.network.0.gamma'], torch.randn(size=[1, 512, 1])],
        dim=1
    )
    model_state_dict['module.separator.network.0.beta'] = torch.cat(
        [model_state_dict['module.separator.network.0.beta'], torch.randn(size=[1, 512, 1])],
        dim=1
    )
    model_state_dict['module.separator.network.1.weight'] = torch.cat(
        [model_state_dict['module.separator.network.1.weight'], torch.randn(size=[512, 512, 1])],
        dim=1
    )

    convtasnet_audio_with_asr_model.load_state_dict(trained_convtasnet_audio_model['model_state_dict'])

    print('Total Parameters in ConvTasNet without ASR model: ',
          sum(p.numel() for p in convtasnet_audio_with_asr_model.parameters()))

    convtasnet_audio_with_asr_model.module.domainTranslation = DomainTranslation()

    optimizer_init = torch.optim.Adam(convtasnet_audio_with_asr_model.parameters(), lr=config.lr[1])

    if config.use_cuda:
        convtasnet_audio_with_asr_model = convtasnet_audio_with_asr_model.cuda()

    # Loading ASR model
    asr_model = DataParallel(ASR(idim=80, odim=5002, args=asrConfig.ModelArgs(), get_features=True))
    if config.use_cuda:
        trained_asr_model = torch.load(config.asr_model)
    else:
        trained_asr_model = torch.load(config.asr_model, map_location=torch.device('cpu'))

    asr_model.load_state_dict(trained_asr_model['model'])

    convtasnet_audio_with_asr_model.module.asr = asr_model

    print('Total Parameters in ConvTasNet with ASR model: ',
          sum(p.numel() for p in convtasnet_audio_with_asr_model.parameters()))

    return convtasnet_audio_with_asr_model, optimizer_init


def seed():
    # This removes randomness, makes everything deterministic

    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True


def main():

    os.system('cp -r ../Oracle "{0}"'.format(config.basePath+'/savedCode'))

    os.environ['CUDA_VISIBLE_DEVICES'] = config.num_cuda

    model, optimizer = get_initial_model_optimizer()

    if config.pretrained:
        saved_model = torch.load(config.pretrained_train)
        model.load_state_dict(saved_model['model_state_dict'])
        optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        del saved_model
        saved_loss = np.load(config.pretrained_train_loss).tolist()
    else:
        saved_loss = None

    dataloader = AVSpeech('train')
    dataloader = DataLoader(
        dataloader, batch_size=config.batchsize['train'],
        num_workers=config.num_workers['train'], worker_init_fn=init_fn
    )

    loss_func = SISNR()

    model, optimizer = train(saved_loss, dataloader, model, optimizer, loss_func)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iterations': -1,
    }, config.model_save_path + '/final_model.pth')

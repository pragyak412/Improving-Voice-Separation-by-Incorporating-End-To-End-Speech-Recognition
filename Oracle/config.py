from datetime import datetime
import os

seed = 0
num_cuda = '0'
use_cuda = True
batchsize = {
	'train': 5*len(num_cuda.split(',')),
	'test': 10*len(num_cuda.split(','))
}
num_workers = {
	'train': 5*len(num_cuda.split(',')),
	'test': 7*len(num_cuda.split(','))
}

optimizer_iterations = 3 // len(num_cuda.split(','))
optimizer_steps = 3/len(num_cuda.split(','))

reduceFactor = 3

lr = {
	1: 1.3e-3,
	10000 * optimizer_iterations // reduceFactor: 7.5e-4,
	30000 * optimizer_iterations // reduceFactor: 5e-4,
	50000 * optimizer_iterations // reduceFactor: 1e-4,
	78000 * optimizer_iterations // reduceFactor: 5e-5,
	100000 * optimizer_iterations // reduceFactor: 1e-5,
}


iterations = {
	'train': int(3000000 / (reduceFactor*1.5)),
	'test': 150000
}

dataset = {
	'AVSpeech': {
		'base_audio_path': {
			'train': '/home/SharedData/Pragya/AVSpeech77HTest71HTrain/train_wav_8000',
			'test': '/home/SharedData/Pragya/AVSpeech77HTest71HTrain/test_wav_8000',
		}
	}
}

num_features = 512

num_speakers = 2

periodic_synthesis = 10000 // reduceFactor
periodic_checkpoint = 50000 // reduceFactor // 2

basePath = '/home/SharedData/Pragya/Experiments/Oracle/'+str(datetime.now())

os.makedirs(basePath, exist_ok=True)

temporary_save_path = {
	'train': basePath + '/train_synthesis',
	'test': basePath + '/test_synthesis',
}

os.makedirs(temporary_save_path['train'], exist_ok=True)
os.makedirs(temporary_save_path['test'], exist_ok=True)

convtasnet_audio_model = '/home/SharedData/Pragya/ModelsToUse/AudioOnlyConvTasNet.pth'
asr_model = '/home/SharedData/Pragya/ModelsToUse/ASR.pth'

model_save_path = basePath
pretrained_test = '/home/SharedData/Pragya/Experiments/ConvTasNet/2020-05-10 19:36:07.985530/40000_model.pkl'

pretrained = False
pretrained_train = '/home/SharedData/Pragya/Experiments/Oracle/2020-05-19 14:38:44.674244/99996.pth'
pretrained_train_loss = '/home/SharedData/Pragya/Experiments/Oracle/2020-05-19 14:38:44.674244/Loss.npy'
start = 99996

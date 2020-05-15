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
	'train': 4*len(num_cuda.split(',')),
	'test': 4*len(num_cuda.split(','))
}

optimizer_iterations = 3 // len(num_cuda.split(','))

lr = {
	1:1.3e-3,
	10000*optimizer_iterations:7.5e-4,
	30000*optimizer_iterations:5e-4,
	50000*optimizer_iterations:1e-4,
	78000*optimizer_iterations:5e-5,
}


iterations = {
	'train': 3000000,
	'test': 150000
}

dataset = {
	'AVSpeech': {
		'base_audio_path': {
			'train': '/home/SharedData/Pragya/AVSpeech77HTest71HTrain/train_wav_8000',
			'test': '/home/SharedData/Pragya/AVSpeech77HTest71HTrain/test_wav_8000'
		}
	}
}

num_speakers = 2

periodic_synthesis = 10000
periodic_checkpoint = 50000

basePath = '/home/SharedData/Pragya/Experiments/ConvTasNet/'+str(datetime.now())

os.makedirs(basePath, exist_ok=True)

temporary_save_path = {
	'train': basePath + '/train_synthesis',
	'test': basePath+ '/test_synthesis',
}

os.makedirs(temporary_save_path['train'], exist_ok=True)
os.makedirs(temporary_save_path['test'], exist_ok=True)


model_save_path = basePath
pretrained_test = '/home/SharedData/Pragya/Experiments/ConvTasNet/2020-05-10 19:36:07.985530/40000_model.pkl'

pretrained = False
pretrained_train = ''
start = 132000

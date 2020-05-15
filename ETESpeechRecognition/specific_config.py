import os

path_to_download = '/home/SharedData/Pragya/LibriSpeech/LibriSpeech'
base_model_path = '/home/SharedData/Pragya/LibriSpeech/models/'

os.makedirs(path_to_download, exist_ok=True)
os.makedirs(base_model_path, exist_ok=True)

test_model = base_model_path + '03:06:41.232423/LibriSpeech_train960.9.107.1032.16:41:24.492658.pth'
cache_dir = '/home/SharedData/Pragya/LibriSpeech/Cache'

resume = {
	'restart': False,
	'model_path':
		'/home/SharedData/Pragya/LibriSpeech/models/03:06:41.232423/LibriSpeech_train960.3.144.7695.17:03:10.451102.pth'
}

num_cuda = '0'

os.environ["CUDA_VISIBLE_DEVICES"] = num_cuda

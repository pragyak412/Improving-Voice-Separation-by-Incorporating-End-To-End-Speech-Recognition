from datetime import datetime
import os

seed = 1

numWorkers = {
	'train': 6,
	'test': 6
}

batchSize = {
	'train': 5,
	'test': 10
}

num_cuda = '0'
optimizer_iteration = 3 // len(num_cuda.split(','))

lr = {
	1: 1.3e-3,
	10000*optimizer_iteration: 7.5e-4,
	30000*optimizer_iteration: 5e-4,
	50000*optimizer_iteration: 1e-4,
	78000*optimizer_iteration: 5e-5,
	100000*optimizer_iteration: 2.5e-5,
	120000*optimizer_iteration: 1e-5,
	150000*optimizer_iteration: 5e-6,
	175000*optimizer_iteration: 1e-6,
}

iterations = {
	'train': 3000000,
	'test': 150000
}

dataSetPath = {
	'train': '',
	'test': ''
}

preTrained = False
preTrainedBasePath = ''
preTrainedModel = preTrainedBasePath + '/'
preTrainedLossPath = preTrainedBasePath + '/loss_plot_training.npy'
startingNo = 40000*3

if not preTrained:
	basePath = '/'+str(datetime.now())
else:
	basePath = preTrainedBasePath

synthesis = basePath + '/synthesis'

os.makedirs(basePath, exist_ok=True)
os.makedirs(synthesis, exist_ok=True)

periodic_output = 3000
periodic_save = 15000


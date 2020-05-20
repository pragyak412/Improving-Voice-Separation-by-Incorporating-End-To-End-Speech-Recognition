import click
import torch
import numpy as np
import random
import config


def seed():

	# This removes randomness, makes everything deterministic

	torch.cuda.manual_seed_all(config.seed)  # if you are using multi-GPU.
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed(config.seed)
	np.random.seed(config.seed)
	random.seed(config.seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


@click.group()
def main():
	seed()
	pass


@main.command()
def train():

	"""
	Training using strong supervision on Synth-Text dataset
	:return: None
	"""

	import os
	import train
	import config

	os.environ['CUDA_VISIBLE_DEVICES'] = config.num_cuda
	train.main()


@main.command()
def testOracleWithTarget():

	"""
	Training using strong supervision on Synth-Text dataset
	:return: None
	"""

	import os
	import test
	import config

	os.environ['CUDA_VISIBLE_DEVICES'] = config.num_cuda
	test.main()


@main.command()
def testOracleWithEstimated():

	"""
	Training using strong supervision on Synth-Text dataset
	:return: None
	"""

	import os
	import test_real
	import config

	os.environ['CUDA_VISIBLE_DEVICES'] = config.num_cuda
	test_real.main()


if __name__ == "__main__":

	main()

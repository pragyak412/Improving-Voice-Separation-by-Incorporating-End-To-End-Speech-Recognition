import os
from mutagen.mp3 import MP3
from tqdm import tqdm
import sys


def calcTotalLength(path):

	totalLength = 0

	iterator = tqdm(os.listdir(path))

	for no, file in enumerate(iterator):
		if file.split('.')[-1] == "mp3":

			try:
				audio = MP3(path + '/' + file)
				totalLength = totalLength + audio.info.length
				iterator.set_description("Current Avg Length: " + str(totalLength/(no + 1)) + ' Total Length: ' + str(totalLength))
			except:
				print('Could Not read file:', file, '\n')

	print('Average Length: ', totalLength/len(iterator))
	print('Total Length: ', totalLength)

if __name__ == "__main__":

	calcTotalLength(path = sys.argv[1])
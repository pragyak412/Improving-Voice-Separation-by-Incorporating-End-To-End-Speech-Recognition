import os
from scipy.io import wavfile
from tqdm import tqdm
import sys
import numpy as np

def getBadFiles( inpath, outpath):
    
    iterator = tqdm(os.listdir(inpath))
    corrupt_data_file = open(outpath, 'a')
    for no, files in enumerate(iterator):
        try:
            fs, data = wavfile.read(inpath +'/' + files)
            if (np.all(data==0)):
                corrupt_data_file.write(inpath +'/' + files+'\n')
        except:
           corrupt_data_file.write(inpath +'/' + files+'\n')

if __name__ == "__main__":
    inpath= sys.argv[1]
    outpath = sys.argv[2]
    getBadFiles(inpath,outpath)
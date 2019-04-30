import os
import glob
import errno
import random
import urllib.request
import numpy as np
from scipy.io import loadmat


class CWRU:

    def __init__(self, exp, rpm, length, test_ratio=0.25):
        if exp not in ('12DriveEndFault', '12FanEndFault', '48DriveEndFault'):
            print(f"wrong experiment name: {exp}")
            exit(1)
        if rpm not in ('1797', '1772', '1750', '1730'):
            print(f"wrong rpm value: {rpm}")
            exit(1)
        # root directory of all data
        rdir = os.path.join(os.path.expanduser('~'), 'Datasets/CWRU')

        fmeta = os.path.join(os.path.dirname(__file__), 'metadata.txt')
        all_lines = open(fmeta).readlines()
        lines = []
        for line in all_lines:
            l = line.split()
            if (l[0] == exp or l[0] == 'NormalBaseline') and l[1] == rpm:
                lines.append(l)

        self.length = length  # sequence length
        self.test_ratio = test_ratio # ratio of testing set
        self._load_and_slice_data(rdir, lines)
        # shuffle training and test arrays
        self._shuffle()
        self.labels = tuple(line[2] for line in lines)
        self.nclasses = len(self.labels)  # number of classes

    def _mkdir(self, path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                print(f"can't create directory '{path}'")
                exit(1)

    def _download(self, fpath, link):
        print(f"Downloading to: '{fpath}'")
        urllib.request.URLopener().retrieve(link, fpath)

    def _load_and_slice_data(self, rdir, infos):
        self.X_train = np.zeros((0, self.length))
        self.X_test = np.zeros((0, self.length))
        self.y_train = []
        self.y_test = []
        for idx, info in enumerate(infos):
            # directory of this file
            fdir = os.path.join(rdir, info[0], info[1])
            self._mkdir(fdir)
            fpath = os.path.join(fdir, info[2] + '.mat')
            if not os.path.exists(fpath):
                self._download(fpath, info[3].rstrip('\n'))

            mat_dict = loadmat(fpath)
            # print(mat_dict.keys())
            key = list(filter(lambda x: 'DE_time' in x, mat_dict.keys()))[0]
            time_series = mat_dict[key][:, 0]

            print(f"Shape of timeseries file {info[2] + '.mat'}: {time_series.shape}, Label: {idx}")

            idx_last = -(time_series.shape[0] % self.length)
            # Reshape into target size of time series samples
            clips = time_series[:idx_last].reshape(-1, self.length)

            n = clips.shape[0]
            # Default: 75% for training, 25% for testing
            # n_split = 3 * n // 4
            n_split = int(n * (1 - self.test_ratio))
            self.X_train = np.vstack((self.X_train, clips[:n_split]))
            self.X_test = np.vstack((self.X_test, clips[n_split:]))
            self.y_train += [idx] * n_split
            self.y_test += [idx] * (clips.shape[0] - n_split)

    def _shuffle(self):
        # shuffle training samples
        index = list(range(self.X_train.shape[0]))
        random.Random(0).shuffle(index)
        self.X_train = self.X_train[index]
        self.y_train = np.array(tuple(self.y_train[i] for i in index))

        # shuffle test samples
        index = list(range(self.X_test.shape[0]))
        random.Random(0).shuffle(index)
        self.X_test = self.X_test[index]
        self.y_test = np.array(tuple(self.y_test[i] for i in index))

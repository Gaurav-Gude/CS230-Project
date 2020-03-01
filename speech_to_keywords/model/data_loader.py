import random
import numpy as np
import os
import sys

import torch
import torch.utils.data as tdata
import torchaudio as ta
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_sequence
import pickle

import utils 


class DataLoader(object):
    """
    Handles all aspects of the data. Stores the dataset_params, vocabulary and tags with their mappings to indices.
    """
    def __init__(self, params):
        """
        Loads dataset_params, vocabulary and tags. Ensure you have run `build_vocab.py` on data_dir before using this
        class.

        Args:
            params: (Params) hyperparameters of the training process. This function modifies params and appends
                    dataset_params (such as vocab size, num_of_tags etc.) to params.
        """

        self.datasets = {}
        # printmax = None
        # count = 0
        # count10 = 0
        # count0 = 0
        # count15 = 0
        # count20 = 0
        # count25 = 0
        # count30 = 0
        # print("Lengh" + str( len(ta.datasets.LIBRISPEECH(params.librispeech_root, url=params.libri_train_split, download=False))))
        # for i in ta.datasets.LIBRISPEECH(params.librispeech_root, url=params.libri_train_split, download=False):
        #     count += 1
        #     if i[0].size()[1] > 480000:
        #         count30 +=1
        #     elif i[0].size()[1] > 400000:
        #         count25 +=1
        #     elif i[0].size()[1] > 320000:
        #         count20 +=1
        #     elif i[0].size()[1] > 240000:
        #         count15 +=1
        #     elif i[0].size()[1] > 160000:
        #         count10 +=1
        #     else :
        #         count0 +=1
        #
        #     if ( count %100 == 0) :
        #         print(count)
        #
        # print(count0)
        # print(count10)
        # print(count15)
        # print(count20)
        # print(count25)
        # print(count30)

        self.datasets['train'] = ta.datasets.LIBRISPEECH(params.librispeech_root, url=params.libri_train_split, download=False)
        self.datasets['dev'] = ta.datasets.LIBRISPEECH(params.librispeech_root, url=params.libri_validation_split, download=False)
        if 'libri_test_split' in params.dict:
            self.datasets['test'] = ta.datasets.LIBRISPEECH(params.librispeech_root, url=params.libri_test_split,
                                                             download=False)
        else:
            self.datasets['test'] = ta.datasets.LIBRISPEECH(params.librispeech_root, url=params.libri_validation_split,
                                                             download=False)
        self.mfcc = ta.transforms.MFCC(melkwargs={ 'n_fft' : 320 } )        #Read the files
        # with open(params.word2ArrayFile, "rb") as handle :
        #     self.word2Array = pickle.load(handle)
        with open(params.word2IdxFile, "rb") as handle:
            self.word2Idx = pickle.load(handle)
        params.dict['vocab_size'] = len(self.word2Idx) + 1 # word2Idx doesn't contain the padding idx so, +1

    def data_info(self, type):
        ret = {}
        ret['size'] = len(self.datasets[type])
        ret['type'] = type
        ret['data'] = self.datasets[type]
        return ret

    def load_data(self, types):
        """
        Loads the data for each type in types from data_dir.

        Args:
            types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
            data_dir: (string) directory containing the dataset

        Returns:
            data: (dict) contains the data with labels for each type in types

        """
        data = {}
        for split in ['train', 'dev', 'test']:
            if split in types:
                data[split] = self.data_info(split)

        return data

    def data_iterator(self, data_info, params, shuffle=False):
        """
        Returns a generator that yields batches data with labels. Batch size is params.batch_size. Expires after one
        pass over the data.

        Args:
            data: (dict) contains data which has keys 'data', 'labels' and 'size'
            params: (Params) hyperparameters of the training process.
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch_data: (Variable) dimension batch_size x seq_len with the sentence data
            batch_labels: (Variable) dimension batch_size x seq_len with the corresponding labels

        """

        collate_fn = lambda d: collate_libri(d, self.mfcc, params, self.word2Idx)

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        data_loader = tdata.DataLoader(self.datasets[data_info['type']], batch_size=params.batch_size, shuffle=shuffle, collate_fn=collate_fn)

        # one pass over data
        for data in enumerate(data_loader):
            # fetch waveforms and keywords

            yield data[1][0], data[1][1]


def collate_libri(data, mfcc, params, word2Idx):
    inputData=[]
    labels=[]
    for d in data:
        mfcc_feaures = mfcc.forward(d[0])
        if mfcc_feaures.size(2) > params.max_mfcc_seq_length :
            mfcc_feaures = mfcc_feaures[:, :,  0:params.max_mfcc_seq_length]
        if params.cuda:
            mfcc_feaures = mfcc_feaures.cuda()
        mfcc_feaures = Variable(mfcc_feaures[0].permute(1, 0))
        inputData.append(mfcc_feaures)

        labels.append(word2Idx.get(d[2].split()[0], word2Idx['<unk>']))

    inputData.sort(key=lambda x: x.size()[0], reverse=True)
    labels_stacked = torch.LongTensor(labels)


    packed_sequence = rnn.pack_sequence(inputData)
    if params.cuda:
        packed_sequence = packed_sequence.cuda()
        labels_stacked = labels_stacked.cuda()
    return packed_sequence, labels_stacked

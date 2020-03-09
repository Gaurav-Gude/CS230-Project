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

import regex as re
regex_only_alphanum = re.compile('[^A-Za-z0-9]')


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
        with open(params.word2ArrayFile, "rb") as handle:
            self.word2Array = pickle.load(handle)
        with open(params.embeddingArrayFile, "rb") as handle :
            self.embeddingArray = np.load(handle)
        with open(params.word2IdxFile, "rb") as handle:
            self.word2Idx = pickle.load(handle)
        params.dict['vocab_size'] = len(self.word2Idx) #+ 1 word2Idx doesn't contain the padding idx so, +1

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

        collate_fn = lambda d: collate_libri(d, params, self.word2Idx, self.word2Array)

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        data_loader = tdata.DataLoader(self.datasets[data_info['type']], batch_size=params.batch_size, shuffle=shuffle, collate_fn=collate_fn)

        # one pass over data
        for data in enumerate(data_loader):
            # fetch waveforms and keywords

            yield data[1][0], data[1][1]

def get_label(book, chapter, utterance_id, root_dir, word2Idx):
    candidate_line = None
    with open(os.path.join(root_dir, str(book), str(chapter)), "r") as file:
        for line in file.readlines():
            words = line.split()
            if len(words) > 1 \
                    and words[0] == "{0:d}-{1:d}-{2:04d}".format(book, chapter, utterance_id) :
                return word2Idx.get(words[1], word2Idx['<unk>'])

    return 0


def collate_libri(data, params, word2Idx, word2Array):
    inputData=[]
    labels=[]
    for d in data:
        words = d[2].split()
        output_words = []
        word_count = 0
        label = get_label(d[3], d[4], d[5], params.labels_dir, word2Idx)
        label_pos = 0
        for word in words:
            word_count += 1
            if word_count > params.max_seq_len :
                break
            word = regex_only_alphanum.sub('', word)
            word = word2Idx.get(word, word2Idx['<unk>'])
            if label == word :
                label_pos = word_count - 1
            wordVec = torch.FloatTensor(word2Array.get(word, word2Array['<unk>']))
            if params.cuda :
                wordVec = wordVec.cuda()

            output_words.append(wordVec)

        inputData.append(torch.stack(output_words))
        labels.append(label_pos)

    sorted_idx = sorted(range(len(inputData)), key=lambda x: inputData[x].size()[0], reverse=True)
    inputData = [inputData[i] for i in sorted_idx ]
    labels = [labels[i] for i in sorted_idx]
    labels_stacked = torch.LongTensor(labels)


    packed_sequence = rnn.pack_sequence(inputData)
    if params.cuda:
        packed_sequence = packed_sequence.cuda()
        labels_stacked = labels_stacked.cuda()
    return packed_sequence, labels_stacked

import random
import numpy as np
import os
import sys

import torch
import torch.utils.data as tdata
import torchaudio as ta
from torch.autograd import Variable

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

        self.train_data = ta.datasets.LIBRISPEECH(params.librispeech_root, url=params.libri_train_split, download=False)
        self.validation_data = ta.datasets.LIBRISPEECH(params.librispeech_root, url=params.libri_validation_split, download=False)

    def load_data(self, types, data_dir):
        """
        Loads the data for each type in types from data_dir.

        Args:
            types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
            data_dir: (string) directory containing the dataset

        Returns:
            data: (dict) contains the data with labels for each type in types

        """
        data = {}
        for split in ['train', 'val', 'test']:
            if split in types:
                sentences_file = os.path.join(data_dir, split, "sentences.txt")
                labels_file = os.path.join(data_dir, split, "labels.txt")
                data[split] = {}

        return data

    def data_iterator(self, type, params, shuffle=False):
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

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        dataset = self.train_data if type == 'train' else self.validation_data
        data_loader = tdata.DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffle)

        # one pass over data
        for data in enumerate(data_loader):
            # fetch waveforms and keywords

            yield data[0], data[1]

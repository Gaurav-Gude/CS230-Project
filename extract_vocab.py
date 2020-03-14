
from pathlib import Path
import re
import string
import numpy as np
import pickle
import sklearn.feature_extraction.text as text
import os

corpus_full = {}
regex = re.compile('[^A-Za-z0-9 \']')
regex_only_alphanum = re.compile('[^A-Za-z0-9]')



with open("data/LibriSpeech/word2idx.glove.6B.300d.pkl", "rb") as handle :
    vocabulary = pickle.load(handle)
    with open("data/LibriSpeech/vocab.glove.6B.300d.txt", "w") as f :
        for key in vocabulary.keys():
            f.write(key + "\n")

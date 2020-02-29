# import speech_recognition as sr
#
# # obtain path to "english.wav" in the same folder as this script
# from os import path
# #AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "english.wav")
# # AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "french.aiff")
# # AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "chinese.flac")
# AUDIO_FILE = "/Users/cmandal/Developer/Stanford CS230/End-to-end-ASR-Pytorch/data/LibriSpeech/train-clean-100/19/198/19-198-0002.flac"
#
# # use the audio file as the audio source
# r = sr.Recognizer()
# with sr.AudioFile(AUDIO_FILE) as source:
#     audio = r.record(source)  # read the entire audio file
#
# # recognize speech using Sphinx
# try:
#     print("Sphinx thinks you said " + r.recognize_sxphinx(audio))
# except sr.UnknownValueError:
#     print("Sphinx could not understand audio")
# except sr.RequestError as e:
#     print("Sphinx error; {0}".format(e))
from pathlib import Path
import re
import string
import numpy as np
import pickle

word2idx = {}
word2Array = {}
embeddings = []


with open("data/LibriSpeech/vectors.txt", 'r') as read_file:
    i = 0
    for line in read_file:
        i += 1
        words = line.split()
        word2idx[words[0]] = i
        if (len(embeddings) == 0) :
            #padding token
            embeddings.append(np.zeros(len(words)-1, dtype='float64'))
            word2Array['<pad>'] = np.zeros(len(words) - 1, dtype='float64')
        embeddings.append(np.asfarray(words[1:]))
        word2Array[words[0]] = np.asfarray(words[1:])
embeddingArray = np.stack(embeddings, axis=0)

with open("data/LibriSpeech/embeddingArray.npy", "wb") as numpy_file :
    np.save(numpy_file, embeddingArray)

with open("data/LibriSpeech/word2idx.pkl", "wb") as word2idx_file :
    pickle.dump(word2idx, word2idx_file, protocol=pickle.HIGHEST_PROTOCOL)

with open("data/LibriSpeech/word2array.pkl", "wb") as word2array_file :
    pickle.dump(word2Array, word2array_file, protocol=pickle.HIGHEST_PROTOCOL)
## Test
with open("data/LibriSpeech/word2idx.pkl", "rb") as handle :
    unserialized_data = pickle.load(handle)
    print(unserialized_data == word2idx)
with open("data/LibriSpeech/embeddingArray.npy", "rb") as handle :
    unserialized_data = np.load(handle)
    print(unserialized_data == embeddingArray)
with open("data/LibriSpeech/word2array.pkl", "rb") as handle:
    unserialized_data = pickle.load(handle)
    print(unserialized_data == word2Array)
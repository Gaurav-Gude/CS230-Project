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

regex1 = re.compile('[%s]' % re.escape(string.punctuation))
regex = re.compile('[^A-Za-z0-9 \']')

with open("data/LibriSpeech/books.all.txt", 'w') as write_file:
    for path in Path('data/LibriSpeech/books/ascii').rglob('*.txt'):
        # print(str(path.absolute()))
        with open(file=str(path.absolute()), encoding='ISO-8859-1', mode='r') as read_file:
            for line in read_file.readlines():
                write_file.writelines([regex.sub(' ', line.upper())])
    for path in Path('data/LibriSpeech/books/utf-8').rglob('*.utf-8'):
        # print(str(path.absolute()))
        with open(file=str(path.absolute()), encoding='utf-8', mode='r') as read_file:
            for line in read_file.readlines():
                write_file.writelines([regex.sub(' ', line.upper())])
    for path in Path('data/LibriSpeech/books/utf-8').rglob('*.txt'):
        # print(str(path.absolute()))
        with open(file=str(path.absolute()), encoding='utf-8', mode='r') as read_file:
            for line in read_file.readlines():
                write_file.writelines([regex.sub(' ', line.upper())])

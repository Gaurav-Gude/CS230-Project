
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

def add_docs(root_path):
    for path in Path(root_path).rglob('*.txt'):
        # print(str(path.absolute()))
        with open(file=str(path.absolute()), encoding='ISO-8859-1', mode='r') as read_file:
            for line in read_file.readlines():
                words = line.split()
                for i in range(1, len(words)):
                    words[i] = regex_only_alphanum.sub('', words[i])
                corpus_full[words[0]] = " ".join(words[1:])

# with open("data/LibriSpeech/books.all.txt", 'w') as write_file:
add_docs('data/LibriSpeech/train-clean-100/')
add_docs('data/LibriSpeech/dev-clean/')

corpus_idx = {}
corpus_list = []

with open("data/LibriSpeech/word2idx.pkl", "rb") as handle :
    vocabulary = pickle.load(handle)
vocabulary['<pad>'] = 0
count = 0
for key, value in corpus_full.items():
    corpus_idx[key] = count
    corpus_list.append(value)
    count+=1

cv=text.CountVectorizer(lowercase=False, token_pattern=r"(?u)[A-Za-z0-9\']+", max_df=0.01)
word_count_vector = cv.fit_transform(corpus_list)
tfIdf = text.TfidfTransformer()
tfIdf.fit(word_count_vector)

feature_names=cv.get_feature_names()

test = corpus_list[0]

def sort_coo(coo_matrix) :
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: x[1], reverse=True)

def keywords(sorted_items, feature_names, n ) :
    out = []
    for i in range(n) :
        if len(sorted_items) > i :
            out.append(feature_names[sorted_items[i][0]])
    return out

root_dir = 'data/LibriSpeech/labels/'
count_no_keywords = 0

for key, value in corpus_full.items():
    key_params = key.split('-')
    file_name = root_dir + key_params[0] + os.sep + key_params[1]
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    tf_idf_vector = tfIdf.transform(cv.transform([value]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    out = keywords(sorted_items, feature_names, 1)
    #Ignore words wiht contractions
    candidate = None
    if(len(out) > 0 ):
        with open(file_name, "a") as f:
            print(out)
            f.write(key + " " + out[0] + os.linesep)
    else :
        count_no_keywords += 1

print(count_no_keywords)
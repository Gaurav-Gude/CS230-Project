import numpy as np
import os,glob
from sklearn.feature_extraction.text import TfidfVectorizer

######### Define empty corpus list

corpus = []

######### Read txt files and append to corpus list

folder_path = 'C:/Users/gaura/OneDrive/Desktop/Winter Quarter 19-20/CS230/Project/train-clean-100-250-word-docs'
for filename in glob.glob(os.path.join(folder_path, '*.txt')):
  with open(filename, 'r') as f:
    text = f.read()    
    corpus.append(text) 

######### Define maximum keywords/features needed

maxfeatures = 20

######### Define array of max df 

maxdf = np.linspace(0.3,0.02,16)           

######### Perform TF-IDF for every maxdf value; list becomes accurate as value decreases

Xtemp = []
X = []

for i in range(maxdf.shape[0]):
    vectorizer = TfidfVectorizer(max_features=maxfeatures,max_df=maxdf[i])
    Y = vectorizer.fit_transform(corpus)
    X = vectorizer.get_feature_names()
    
    if (X == Xtemp):
        break
    else:
        Xtemp = X
print(X)        
print(vectorizer.idf_)

######### Storing keywords in new file
  
file1 = open("Keywords.txt","w+")
file1.write(str(X))
file1.close()

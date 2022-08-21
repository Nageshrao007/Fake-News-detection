#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"
CSC 790 Final Project

Team Members: Alaaldin Dwaik, Francisco Ruiz, Nabil Shawkat, Nagesh Pasyadala 

#This program can be run by clicking F5. 


"""

#%%
import math

#import time
from collections import Counter
from operator import itemgetter, mul
from itertools import combinations

#%%
import nltk
import numpy as np
#from numpy import dot
from numpy.linalg import norm
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer



#Importing libraries
import os
import os.path
import nltk as tk
from nltk.stem import PorterStemmer

#%% importing naive_bayes classifier
from sklearn.naive_bayes import MultinomialNB
import sys
import multiprocessing as mp
import parallelProcesses as PP
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

#%%
#Declaring global variables
celeb_fake_dict = {}
celeb_legit_dict = {}
news_fake_dict = {}
news_legit_dict = {}

#%%
#Downloads stopwords for removal
tk.download('stopwords')
stop_words = set(stopwords.words('english'))

#%%
#Downloads tokenizer
tk.download('punkt')

#%%
#Setting folder path 
path = os.getcwd() 
celeb_fake_file_path = os.getcwd() + '/fakeNewsDatasets/celebrityDataset/fake/'
celeb_legit_file_path = os.getcwd() + '/fakeNewsDatasets/celebrityDataset/legit/'
news_fake_file_path = os.getcwd() + '/fakeNewsDatasets/fakeNewsDataset/fake/'
news_legit_file_path = os.getcwd() + '/fakeNewsDatasets/fakeNewsDataset/legit/'

#%%
all_path_array = [celeb_fake_file_path, celeb_legit_file_path, news_fake_file_path, news_legit_file_path]

#Creating blank dictionary for inverted index
dictionary = {}

#Creating stemmer
ps = PorterStemmer()

#%%
#Function for reading a file and storing each line in a returned array
def process_file(file_name):

    final_text_array = []
    
    # file_number = file_name.replace('.fake', '').strip()
    # file_number = file_number.replace('fake', '').strip()
    # file_number = file_number.replace('.legit', '').strip()
    # file_number = file_number.replace('legit', '').strip()
    # file_number = file_number.replace('biz', '').strip()
    # file_number = file_number.replace('.txt', '').strip()

    file = open(file_name, encoding = 'utf8')   #Opens the file
 
    for line in file:
        
        words = tk.word_tokenize(line)                          #TK function to tokenize each line

        new_words = [word for word in words if word.isalnum()]   #TK function to remove punctuation

        new_words = [word for word in new_words if not word.lower() in stop_words]  #TK function to remove stopwords

        new_words = [ps.stem(word) for word in new_words]   #TK function to stem remaining words
   
        #Only append list if not empty (skips any blank lines)
        if new_words:
            final_text_array.extend(new_words)                  #Appends the words to the text array

    return final_text_array                           #Returns the array and file number
#%%
#Function for loading all files from the relevant directories
def load_files(path_array):
    ix = 1
    for path in path_array:
        
        os.chdir(path)
        
        for file in os.listdir(path): 
            
            file_number = ix
            ix = ix +1

            processed = process_file(file)

            if path == celeb_fake_file_path:
                celeb_fake_dict[file_number] = processed
            elif path == celeb_legit_file_path:
                celeb_legit_dict[file_number] = processed
            elif path == news_fake_file_path:
                news_fake_dict[file_number] = processed
            elif path == news_legit_file_path:
                news_legit_dict[file_number] = processed  

#MAIN FUNCTION
#%%
load_files(all_path_array)

#%%
# inverted_index = {**celeb_fake_dict, **celeb_legit_dict , **news_fake_dict, **news_legit_dict}
inverted_index = [celeb_fake_dict, celeb_legit_dict ,news_fake_dict, news_legit_dict]
#%%
classes = {}
ix = 0
for tokensDict in inverted_index:
    for fileName in tokensDict.keys():
        classes[fileName] = 1 - (ix%2)
    ix = ix +1

#%%

tokens = {}
documentFrequency = {}
termFrequency = {}

#%%
def DocumentFrequencyFunction(inverted_index):
    DF = {}
    for tokensDict in inverted_index:
        for fileName, tokensList in tokensDict.items():
            for token in set(tokensList):
                d = DF.get(token,0)
                DF[token] = d+1
    return DF

#%%
def genericFunction(DF, inverted_index, N):
    TF = {}
    TF_IDF = {}

    for tokensDict in inverted_index:
         for fileName, tokensList in tokensDict.items():
            tokens = Counter(tokensList)
            tDict = {}
            tfidfDict = {}
            for token in tokens:
                tDict[token] = tokens[token]
                tfidfDict[token] = tokens[token] * math.log(N / (DF[token]))

            TF[fileName] = tDict
            TF_IDF[fileName] = tfidfDict
    
    return TF,TF_IDF

                                                                 
#%%
documentFrequency = DocumentFrequencyFunction(inverted_index)
#%%
termFrequency, termFrequencyInverseDocumentFrequency = genericFunction(documentFrequency, inverted_index,
len(classes))

#%%
MyIndex = termFrequencyInverseDocumentFrequency

uniqueTerms = list(documentFrequency.keys()) # template

N = len(MyIndex.keys()) #num of documents in the collection

arr = np.zeros((N,len(uniqueTerms)+1))

for r,doc in enumerate(MyIndex.keys()):
    arr[r,0] = classes[doc]
    for c,term in enumerate(uniqueTerms):
        dIndex = MyIndex[doc]
        tfIdf = dIndex.get(term,0)
        arr[r,c+1] = tfIdf
        
#%%       
#Calculating Mutual Information


# %%
#if running (.py) file with ipython kernel
#to enable MultiProcessing (Windows Only)
sys.modules['__main__'].__file__ = 'ipython'

#%% get sample to test for best features
# we will use 40% of the data to get the best features to use
# this will allow us to reduce the time and resources required
rem,Sample = train_test_split(arr, test_size=0.3, random_state=2,stratify=arr[:,0])

#%% Creating Parts for MultiProcessing
# we need to split data , so we can distribute it over 
# the different processes
parts = []

partSize = 100 # each part will contain 100 (column)

idx = 0
totalSize = len(uniqueTerms)
print('totalSize',totalSize)

while idx < totalSize:
    batchSize = partSize
    if idx + batchSize > totalSize:
        batchSize = totalSize - idx

    colIdx = idx+1

    partIdx = np.r_[0:1,colIdx:colIdx+batchSize]
    subTerms = uniqueTerms[idx:idx+batchSize]
    parts.append((subTerms,arr[:,partIdx].copy(),partIdx[1:]))

    # print('status:',idx,batchSize,colIdx)
    idx = idx + partSize

print(f'done partitioning, # of parts {len(parts)}')
# print(parts)

#%% PreProcess the parts (get statics)
if __name__ ==  '__main__': 
    pool = mp.Pool()
    vector_statics = pool.starmap(PP.vectorsPreProcessing,parts)

    pool.terminate()

print('done')

#%%
len(vector_statics)

#%% Calculate the Scores (using the statics we have collected previously)
if __name__ ==  '__main__': 
    pool = mp.Pool()
    MutualScores = pool.map(PP.calculateMutualInfo,vector_statics)

    pool.terminate()

print('done')

#%%
# MutualScores
#%% combining the scores <get one scores vector>
# scores : [col#,score]
scores = np.zeros((len(uniqueTerms),2),dtype=np.float64)
print(scores.shape)
for batch in MutualScores:
        for colNum,score in batch:
            colIndex = int(colNum) - 1
            scores[colIndex][0] = colNum
            scores[colIndex][1] = score
#%% Order the scores based on 2nd column (score) (desc)
scores = scores[np.argsort(scores[:,1])][::-1]
#%%
#remove features where the score == 0
scores = scores[scores[:,1] > 0]
#make values bigger <just to allow use to read>
scores[:,1] = scores[:,1] * 100000
scores.shape
#%% #get histogram for scores 
# this even reduce the search time for best K
n,egdes = np.histogram (scores[:,1],bins=100)
n[n>0]
#%%
#seems like first 15 bins contain the most important scores
topN = np.sum(n[:1])
print(topN)


#%% function to create / train / validate the NB Classifier
def getClassifier(X_train,X_test,y_train,y_test):
    clf = MultinomialNB()

    #train
    clf.fit(X_train,y_train)

    #validate
    predictions = clf.predict(X_test)

    correct = 0.0
    for i in range(predictions.shape[0]):
        if predictions[i] == y_test[i]:
            correct += 1

    # print(f'correct:{correct}, accuracy:{correct / predictions.shape[0]:0.2f}%')

    accuracy = correct / predictions.shape[0]
    return clf,accuracy 

#%% Function to test for Range of K features and get results
def TestForK(scores,arr,topNStart,step):
    topN = topNStart

    results = []

    print(f'Total # of features is {{{scores.shape[0]}}}.')

    while topN <= scores.shape[0]:
        selected_features = scores[:topN,0].copy().astype(np.int)
        selected_features = np.insert(selected_features,0,0,axis=0)

        dt = arr[:,selected_features]
        X = dt[:,1:]
        Y = dt[:,0]

        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, stratify = Y,random_state=2)

        clf,accuracy = getClassifier(X_train, X_test, y_train, y_test)
        print(f'for Top {{{topN}}} features , accuracy is:{{{accuracy}}}.')

        results.append(
            (
               topN,accuracy 
            )
        )
        
        if topN + step > scores.shape[0] and topN < scores.shape[0]:
            topN = scores.shape[0]
        else:
            topN = topN + step

    return results

#%% run the test on the sample 
res = TestForK(scores,Sample,1,1)

# %% prepare data for plotting
res_np = np.zeros((len(res),2))
for i,r in enumerate(res):
    res_np[i,0] = r[0] # topN
    res_np[i,1] = r[1] # accuracy

#%% plot the results
plt.plot(res_np[:,0],res_np[:,1],'-')
plt.xlabel('Top K', fontsize=18)
plt.ylabel('Accuracy', fontsize=16)

#%%
sorted(res,key= lambda x:(x[1],-x[0]),reverse=True)
# %% getting best Tok K
bestResult = sorted(res,key= lambda x:(x[1],-x[0]),reverse=True)[0]
bestK = bestResult[0]
bestAccuracy = bestResult[1]

print(f'bestK:{bestK},bestAccuracy:{bestAccuracy}')
# %% Run on the Full DataSet
best_features = scores[:bestK,0].copy().astype(np.int)
best_features = np.insert(best_features,0,0,axis=0)

dt = arr[:,best_features]
X = dt[:,1:]
Y = dt[:,0]

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, stratify = Y,random_state=2)

clf,accuracy = getClassifier(X_train, X_test, y_train, y_test)
print(f'accuracy is:{{{accuracy}}}.')

# %%

# %%

# %%

# %%

# %%

# %%

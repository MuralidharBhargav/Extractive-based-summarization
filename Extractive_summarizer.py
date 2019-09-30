#import necessary libraries
import re
import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

#reading file
file = open("Z:/rahul dravid.txt","r") 
data=file.readlines() 
file.close()

#define preprocessing steps
#lower case
#remove everything inside []
#remove 's
#fetch only ascii characters

def preprocessor(text):
    newString = text.lower()
    newString = re.sub("[\(\[].*?[\)\]]", "", newString)
    newString = re.sub("'s","",newString)
    newString = re.sub("[^'0-9.a-zA-Z]", " ", newString)
    tokens=newString.split()
    return (" ".join(tokens)).strip()

#call above function
text=[]
for i in data:
    text.append(preprocessor(i))

all_sentences=[]    
for i in text:
    sentences=i.split(".")       
    for i in sentences:
        if(i!=''):
            all_sentences.append(i.strip())

# tokenizing the sentences for training word2vec
tokenized_text = [] 
for i in all_sentences:
    tokenized_text.append(i.split()) 

#define word2vec model
model_w2v = gensim.models.Word2Vec(
            tokenized_text,
            size=200, # desired no. of features/independent variables 
            window=5, # context window size
            min_count=2,
            sg = 0, # 1 for cbow model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 2, # no.of cores
            seed = 34)

#train word2vec
model_w2v.train(tokenized_text, total_examples= len(tokenized_text), epochs=model_w2v.epochs)

#define function to obtain sentence embedding
def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError: # handling the case where the token is not in vocabulary
                         
            continue
    if count != 0:
        vec /= count
    return vec

#call above function
wordvec_arrays = np.zeros((len(tokenized_text), 200))
for i in range(len(tokenized_text)):
    wordvec_arrays[i,:] = word_vector(tokenized_text[i], 200)

# similarity matrix
sim_mat = np.zeros([len(wordvec_arrays), len(wordvec_arrays)])

#compute similarity score
for i in range(len(wordvec_arrays)):
  for j in range(len(wordvec_arrays)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(wordvec_arrays[i].reshape(1,200), wordvec_arrays[j].reshape(1,200))[0,0]

#Generate a graph
nx_graph = nx.from_numpy_array(sim_mat)
#compute pagerank scores
scores = nx.pagerank(nx_graph)
#sort the scores
sorted_x = sorted(scores.items(), key=lambda kv: kv[1],reverse=True)


sent_list=[]
for i in sorted_x:
    sent_list.append(i[0])

#extract top 10 sentences
num=10
summary=''
for i in range(num):
    summary=summary+all_sentences[sent_list[i]]+'. '
print(summary)
    






 

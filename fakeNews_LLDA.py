# Import the LLDA (labeled latent dirichlet allocation) class that we're so interested in
from llda import LLDA
# Import our code for combining the "real" and "fake" datsets into something usable
from datasetInvestigation import extractDataset
# Numpy is always handy
import numpy as np
# This lets us split the data into training and testing sets
from sklearn.model_selection import train_test_split
# This lets us call map with str.split (thanks http://stackoverflow.com/questions/8461230/python-map-string-split-list)
from operator import methodcaller
# And the dev we're borrowing from  prints to sys.stderr in the nltk example, so for now lets just keep doing that
import sys

# Grab the combined dataset and turn it into a numpy array I guess
totalDataset = np.array(extractDataset());
# Put the class labels on the last column because that's how train_test_split likes it
totalDataset[:, 0], totalDataset[:, 1] = totalDataset[:, 1], totalDataset[:, 0].copy()

trainSet, testSet = train_test_split(totalDataset, test_size=0.8)

# Turn this dataset into a list of strings
# Then, split each string into its own list because that's what llda.py wants
#corpus = map(methodcaller("split", " "), list(trainSet[:, 0]))
corpus = list(trainSet[:, 0])

# Thanks http://stackoverflow.com/questions/25346058/removing-list-of-words-from-a-string
# Allows for removing stupid words that do nothing about distinguishing topics from every headline
stopwords = ['the', 'a', 'at', 'of', '...', ':', ',', 'in', 'for', 'with', 'to', 'on', 'and', 'is', ';', '-', '–', '\'', '"', '|', 
             'are', '&', 'this', 'about', 'from', 'be', 'as', 'by', 'up', 'what', 'will', 'how', 'that', 'you', 'it', 'why', 'after']
trimmed_corpus = []
for i in range(len(corpus)):
    temp = [word for word in corpus[i].split() if word.lower() not in stopwords]
    trimmed_corpus.append(temp)

corpus = trimmed_corpus

# Turn the 1.0/0.0 float labels into 1 or 0
# Then turn them into strings
# Then turn each string into a one entry "list" of strings
# Because that's what this llda.py wants I guess
labels = map(list, map(str, map(int, trainSet[:, 1])))

# The labels that we're interested in are '0' for real and '1' for fake
labelset = ['0', '1']

# I don't know what these parameters are, but lets keep them how they are in the llda_nltk example
# Oh, K is the number of topics, so we want that to be 2, not the default of 20
K = 2
# Alpha and Beta are something to do with controlling similarity between topics and words within a topic
# This video explained it and I forgot https://www.youtube.com/watch?v=3mHy4OSyRf0
alpha = 0.0001
beta = 0.0001
iterations = 50

# Instantiate an LLDA object and "set the corpus" which is probably building some kind of internal model
llda = LLDA(K, alpha, beta)
llda.set_corpus(labelset, corpus, labels)

# Then, do iteration-many instances of "inference" -- again, whatever that means. Hopefully it's, like, predicting?
# TODO: Figure out what exactly is happening here

# Actually, I think this is basically like training. Ideally, lower perplexity in each step is good going off of Wikipedia
# Perplexity is a measure of how well a probability distribution predicts a sample, with lower being better
for i in range(iterations):
    sys.stderr.write("-- %d : %.4f\n" % (i, llda.perplexity()))
    llda.inference()
    
    phi = llda.phi()
    for k, label in enumerate(labelset):
        print "\n-- label %d : %s" % (k, label)
        for w in np.argsort(-phi[k])[:20]:
            print "%s: %.4f" % (llda.vocas[w], phi[k,w])
print "perplexity : %.4f" % llda.perplexity()

# And I have no idea what this does right now
# TODO: Figure out what this does
# I think it's just printing out the 20 most common words in each topic
phi = llda.phi()
for k, label in enumerate(labelset):
    print "\n-- label %d : %s" % (k, label)
    for w in np.argsort(-phi[k])[:20]:
        print "%s: %.4f" % (llda.vocas[w], phi[k,w])
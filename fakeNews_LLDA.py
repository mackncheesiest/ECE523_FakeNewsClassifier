# Import the LLDA (labeled latent dirichlet allocation) class that we're so interested in
from llda import LLDA
# Import our code for combining the "real" and "fake" datsets into something usable
from datasetInvestigation import extractDataset
# Numpy is always handy
import numpy as np
# This lets us split the data into training and testing sets
from sklearn.model_selection import train_test_split
# And the dev we're borrowing from  prints to sys.stderr in the nltk example, so for now lets just keep doing that
import sys

# Grab the combined dataset and turn it into a numpy array I guess
totalDataset = np.array(extractDataset());
# Put the class labels on the last column because that's how train_test_split likes it
totalDataset[:, 0], totalDataset[:, 1] = totalDataset[:, 1], totalDataset[:, 0].copy()

trainSet, testSet = train_test_split(totalDataset, test_size=0.2)

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

# K is the number of topics, so we want that to be 2, not the default of 20
K = 2
# Alpha and Beta are something to do with controlling similarity between topics and words within a topic
# This video explained it and I forgot https://www.youtube.com/watch?v=3mHy4OSyRf0
# Lets keep them how they are in the llda_nltk example
alpha = 0.0001
beta = 0.0001
# So, this seems to converge to a top 20 list *really* fast, so lets do like 10 iterations
iterations = 10

# Instantiate an LLDA object and "set the corpus" which is probably building some kind of internal model
llda = LLDA(K, alpha, beta)
llda.set_corpus(labelset, corpus, labels)

# Then, do some kind of training prodecure thing. Inference is better if it's lower or something.
# Ideally, lower perplexity in each step is good going off of Wikipedia
# Perplexity is a measure of how well a probability distribution predicts a sample, with lower being better
for i in range(iterations):
    sys.stderr.write("-- %d : %.4f\n" % (i, llda.perplexity()))
    llda.inference()
    
    #phi = llda.phi()
    #for k, label in enumerate(labelset):
    #    print "\n-- label %d : %s" % (k, label)
    #    for w in np.argsort(-phi[k])[:20]:
    #        print "%s: %.4f" % (llda.vocas[w], phi[k,w])
print "perplexity : %.4f" % llda.perplexity()

# This prints out the 40 most likely words in each topic
#phi = llda.phi()
#for k, label in enumerate(labelset):
#    print "\n-- label %d : %s" % (k, label)
#    for w in np.argsort(-phi[k])[:40]:
#        print "%s: %.4f" % (llda.vocas[w], phi[k,w])
        
testCorpus = list(testSet[:, 0])

trimmedTestCorpus = []
for i in range(len(testCorpus)):
    temp = [word for word in testCorpus[i].split() if word.lower() not in stopwords]
    trimmedTestCorpus.append(temp)
    
phi = llda.phi()
totalPredictionsCorrect = 0
for i in range(len(testSet[:, 0])):
#for i in range(1000):
    # Each column is a one-hot entry indicating the existence of a particular word from the vocab in the dataset
    oneHotVector = np.zeros([1, len(llda.vocas)])
    for j in range(len(trimmedTestCorpus[i][:])):
        if trimmedTestCorpus[i][j] in llda.vocas:
            oneHotVector[0, llda.vocas.index(trimmedTestCorpus[i][j])] = 1
    # Then, dot product that one-hot vector with the phi vector and see what the most likely topic is
    #print("Vector under consideration: " + str(trimmedTestCorpus[i]))
    
    #dotProductCommon = np.dot(oneHotVector[0, :], phi[0, :])
    dotProductReal = np.dot(oneHotVector[0, :], phi[1, :])
    dotProductFake = np.dot(oneHotVector[0, :], phi[2, :])
    
    #print("dot product with topic common: " + str(dotProductCommon))
    #print("dot product with topic real: " + str(dotProductReal))
    #print("dot product with topic fake: " + str(dotProductFake))
    
    if dotProductReal >= dotProductFake:
        #print("Classified as real")
        expected = 0.0
    else:
        #print("Classified as fake")
        expected = 1.0
        
    #print("Actual label: " + str(testSet[i, 1]))
    
    if expected == testSet[i, 1]:
        totalPredictionsCorrect = totalPredictionsCorrect + 1
    
    if i % 10000 == 0 and i > 0:
        print("i = " + str(i) + ", " + str(100.*i/len(testSet[:, 0])) + "% done")
    
    #print("\n\n")

print("Testing Accuracy: " + str((100.*totalPredictionsCorrect)/len(testSet[:, 0])))
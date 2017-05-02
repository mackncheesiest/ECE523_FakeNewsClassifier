# -*- coding: utf-8 -*-

import shelve
import numpy as np
from sklearn.metrics import matthews_corrcoef

def load_llda_model(filename):
    my_shelf = shelve.open(filename=filename)
    #for key in my_shelf:
    llda = my_shelf['llda']
    testSet = my_shelf['testSet']
    trainSet = my_shelf['trainSet']
    #globals()[key] = my_shelf[key]
    my_shelf.close()
    
    return llda, testSet, trainSet

def test_llda_model(llda, testSet):
    
    stopwords = ['the', 'a', 'at', 'of', '...', ':', ',', 'in', 'for', 'with', 'to', 'on', 'and', 'is', ';', '-', '–', '\'', '"', '|', 
            'are', '&', 'this', 'about', 'from', 'be', 'as', 'by', 'up', 'what', 'will', 'how', 'that', 'you', 'it', 'why', 'after']
    
    testCorpus = list(testSet[:, 0])

    trimmedTestCorpus = []
    for i in range(len(testCorpus)):
        temp = [word for word in testCorpus[i].split() if word.lower() not in stopwords]
        trimmedTestCorpus.append(temp)
        
    phi = llda.phi()
    totalPredictionsCorrect = 0
    
    pred_classes = np.zeros([np.size(testSet, axis=0)])
    true_classes = np.float64(np.reshape(testSet[:, 1], [np.size(testSet, axis=0)]))
        
    # True positives (predicted fake and is fake), false positives (pred fake and is real), 
    # and false negatives (pred real and is fake), and fal
    TP = 0
    FP = 0
    FN = 0
    
    for i in range(len(testSet[:, 0])):
    #for i in range(1000):
        # Each column is a one-hot entry indicating the existence of a particular word from the vocab in the dataset
        oneHotVector = np.zeros([1, len(llda.vocas)])
        for j in range(len(trimmedTestCorpus[i][:])):
            if trimmedTestCorpus[i][j] in llda.vocas:
                oneHotVector[0, llda.vocas.index(trimmedTestCorpus[i][j])] = 1

        # Then, dot product that one-hot vector with the phi vector and see what the most likely topic is
        dotProductReal = np.dot(oneHotVector[0, :], phi[1, :])
        dotProductFake = np.dot(oneHotVector[0, :], phi[2, :])
        
        if dotProductReal >= dotProductFake:
            expected = 0.0
        else:
            expected = 1.0
            
        pred_classes[i] = expected        
        if expected == testSet[i, 1]:
            totalPredictionsCorrect = totalPredictionsCorrect + 1
            
        if expected == 1.0 and testSet[i, 1] == 1.0:
            TP += 1
        if expected == 1.0 and testSet[i, 1] == 0.0:
            FP += 1
        if expected == 0.0 and testSet[i, 1] == 1.0:
            FN += 1
        
        if i % 10000 == 0 and i > 0:
            print("i = " + str(i) + ", " + str(100.*i/len(testSet[:, 0])) + "% done")
        
    print("Testing Accuracy: " + str((100.*totalPredictionsCorrect)/len(testSet[:, 0])))
    print("Matthews Correlation Coeff: " + str(matthews_corrcoef(y_true=true_classes, y_pred=pred_classes)))
    #Recall = True Positives / (True Positives + False Negatives)
    print("Recall: " + str(100.*TP/(TP + FN)))
    #Precision = True Positives / (True Positives + False Positives)
    print("Precision: " + str(100.*TP/(TP + FP)))
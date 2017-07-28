## ECE523_FakeNewsClassifier
This repository consists of files used for an exploratory study of binary classification of "real" or "fake" news from a given dataset using a supervised variant of a topic modeling technique known as latent drichlet allocation(LDA).

#Dataset investigation:
In this file, we take our two datasets -- one consisting entirely of [https://www.kaggle.com/mrisdal/fake-news]("fake news") from 2016 and the other a mix of [https://www.kaggle.com/uciml/news-aggregator-dataset](news stories) from 2014 -- and we do preliminary analysis to determine that there are at least websites in common between the two (learning to classify just based on the dataset each came from would be boring). Then, we construct a single unified dataset consisting of each article title, the hostname it came from, and the "real/fake" classification status (with uci news sites presumed real and "fake news" sites presumed fake)

#fakeNews_llda.py
Retrieves the dataset from `datasetInvestigation.py` and utilizes the LLDA library provided by llda.py to build an llda model with specified model parameters. It then splits the dataset into training and testing portions, tests the model, reports on the results, and writes the generated model along with the training and testing datasets to a file.

#reload_llda_model.py
Provides methods for loading an llda model that was saved to disk as well as testing (with more metrics than provided in fakeNews_llda) that reloaded llda model.

#llda.py
Provides the llda library that we use as the basis of our project's implementation

#llda_nltk.py
Provides a working example used as the basis for fakeNews_LLDA.py


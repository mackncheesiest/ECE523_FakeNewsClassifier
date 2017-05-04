## ECE523_FakeNewsClassifier
This repository basically consists of files used for classification of fake news from the dataset using a supervised variant of a topic modeling technique latent drichlet allocation(LDA).

#Dataset investigation-
In this file we have conjoined two datasets our  goal  was  to  introduce  as  little  bias  towards  one  dataset  or  the  otheras possible in the eventual model by minimizing the amount of information that could be used to identify which dataset the data came from.

#fakeNews_llda.py
Retrieves the dataset from datasetInvestigation.py and utilizes the LLDA library provided by llda.py to build an llda model with specified model parameters. It then tests the model, reports on the results, and writes the generated model along with the training and testing datasets to a file.

#reload_llda_model.py
Provides methods for loading an llda model that was saved to disk as well as testing (with more metrics than provided in fakeNews_llda) that reloaded llda model.

#llda.py
Provides the llda library that we use as the basis of our project's implementation

#llda_nltk.py
Provides a working example used as the basis for fakeNews_LLDA.py


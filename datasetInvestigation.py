import pandas as pd
import numpy as np

# I don't even know if I can find all the stack overflow posts I used to help me figure out all this data wrangling
# Pretend I linked them here
def extractDataset():
    fakenews = pd.read_csv('./datasets/fake.csv')
    ucinews = pd.read_csv('./datasets/uci-news-aggregator.csv')
    
    # We want to do the intersection of the hostnames in the fake news
    # First, rename the site_url of fakenews to HOSTNAME so that the column matches with ucinews
    fakenews = fakenews.rename(columns={'site_url': 'HOSTNAME', 'title': 'TITLE'})
    
    # Then, remove www. from the front of all UCI hostnames that don't have them (none of the fakenews sites have them and we want to compare them)
    ucinews.loc[ucinews['HOSTNAME'].str[0:4] == 'www.', 'HOSTNAME'] = ucinews.loc[ucinews['HOSTNAME'].str[0:4] == 'www.', 'HOSTNAME'].str[4:]
    
    # Essentially just build a list of all the hostnames within each dataset so that we can check if we have any in common
    fakenews_subset = fakenews.drop_duplicates(subset=['HOSTNAME'])
    ucinews_subset = ucinews.drop_duplicates(subset=['HOSTNAME'])
    
    # Finally, now that the data is nice and comparable, compare the two to see if there are any sites in common
    intersectionSet_subset = pd.merge(fakenews_subset, ucinews_subset, how='inner', on=['HOSTNAME'])
    
    # And now that we're confident that there are at least sites in common between the two, build one dataset that contains information from both datasets
    # For now, let's just do it based on TITLE and Fake/Not Fake
    print "There are %d hostnames in common between the two datasets" % (intersectionSet_subset.shape[0])
    
    # Add a "FAKE" column to uci dataset and initialize it to zeros
    ucinews['FAKE'] = np.zeros([ucinews.shape[0]])
    # But fill it with 1's for websites that are contained within the fake news dataset
    ucinews.loc[ucinews['HOSTNAME'].isin(intersectionSet_subset['HOSTNAME']), 'FAKE'] = 1
    
    # Then, shrink each of the two DataFrames to contain only the "TITLE" and "FAKE" datapoints
    fakenews = np.array(pd.DataFrame({'TITLE': fakenews['TITLE'], 'FAKE': np.ones([fakenews.shape[0]])}))
    ucinews = np.array(pd.DataFrame({'TITLE': ucinews['TITLE'], 'FAKE': ucinews['FAKE']}))
    
    
    # Conjoin them into one DataFrame, and we have ourselves a dataset to work with
    conjoinedDataset = np.concatenate((ucinews, fakenews))
    conjoinedDataset = pd.DataFrame({'TITLE': conjoinedDataset[:, 1], 'FAKE': conjoinedDataset[:, 0]})
    
    # Some of the titles are turned into "NaN"s, so drop them
    conjoinedDataset = conjoinedDataset.dropna()
    
    #conjoinedDataset['FAKE'] = map(str, conjoinedDataset['FAKE'])
    
    return conjoinedDataset

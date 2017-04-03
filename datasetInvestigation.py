import pandas as pd

# I don't even know if I can find all the stack overflow posts I used to help me figure out all this data wrangling
# Pretend I linked them here

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
# Build a full intersectionSet so that we can see how much data we have in common between the two
intersectionSet = pd.merge(fakenews, ucinews, how='inner', on=['HOSTNAME'])
# And also, intersect the subset sets so that we can see which hostnames appear in each
intersectionSet_subset = pd.merge(fakenews_subset, ucinews_subset, how='inner', on=['HOSTNAME'])

# And now that we're confident that there are at least sites in common between the two, build one dataset that contains information from both datasets
# For now, let's just do it based on TITLE and HOSTNAME

# TODO: Figure out what to include in here for classification labels. 
# Also, we need to merge the type of story from fake news ("bias", "fake", ...) with the labels from uci ("e" = entertainment, "s" = science, ...)
conjoinedDataset = pd.DataFrame({'HOSTNAME': fakenews['HOSTNAME'], 'TITLE': fakenews['TITLE']}).append(
        pd.DataFrame({'HOSTNAME': ucinews['HOSTNAME'], 'TITLE': ucinews['TITLE']})
        )
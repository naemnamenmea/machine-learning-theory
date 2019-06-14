from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

dataset = fetch_20newsgroups(subset='all', shuffle=True, categories=categories)

data = dataset.data
target = dataset.target
filenames = dataset.filenames
DESCR = dataset.DESCR
target_names = dataset.target_names

vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
X = vectorizer.fit_transform(data)
#pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names()).head(10)

svd = TruncatedSVD(n_components=2)

X = svd.fit_transform(X)
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.show()

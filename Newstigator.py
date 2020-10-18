# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Natural Language Processing
# %% [markdown]
# ## Importing the libraries

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]
# ## Importing the dataset

# %%
dataset = pd.read_csv('data.csv')

# %% [markdown]
# ## Cleaning the texts

# %%
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(dataset['title'])):
  review = re.sub('[^a-zA-Z]', ' ', dataset['title'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)


# %%
# print(len(corpus))

# %% [markdown]
# ## Creating the Bag of Words model

# %%
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000)
#RAM tak cukup banyak" feature hehe
#Sweetspot di 1k
cv.fit(corpus)
X = cv.transform(corpus).toarray()
y = dataset.iloc[:, -1].values


# %%
import pickle
pickle.dump(cv, open("vectorizer.pickel", "wb"))


# %%
# X[453]

# %% [markdown]
# ## Splitting the dataset into the Training set and Test set

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# %% [markdown]
# ## Training the K Nearest Neighbor model on the Training set

# %%
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)


# %%
import joblib
joblib.dump(classifier, 'knn_model.pkl')

# %% [markdown]
# ## Predicting the Test set results

# %%
# y_pred = classifier.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# %% [markdown]
# ## Making the Confusion Matrix

# %%
# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# accuracy_score(y_test, y_pred)


# %%



# %%
'''
Feature||Acc
2500||8846
3000||8821
2000||8860
1500||8874
1000||8964
'''


# %%
# aaa = input('input news: ')
# b = pd.DataFrame(aaa)

# y_pred = classifier.predict([aaa])
# print(y_pred)


# %%
# pd.DataFrame("asd")


# %%



# %%



# %%



# %%



# %%



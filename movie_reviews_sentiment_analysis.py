### Movie Reviews Sentiment Analysis | Training ###
"""
Description:
              Movie reviews sentiment analysis is a project which is based on natural language processing, where we use NLP techniques to extract useful words of each review
              and based on these words we can use binary classification to predict the movie sentiment if it's positive or negative
"""

## Importing modules ##
import numpy as np 
import pandas as pd 
import re 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
import joblib
# nltk.download() # Download "punkt" package

## 1 | Data Preprocessing ##
"""Prepare the dataset before training"""

# 1.1 Load dataset
dataset = pd.read_csv('Dataset/IMDB.csv')
print(f"Dataset shape : {dataset.shape}\n")
print(f"Dataset head : \n{dataset.head()}\n")

# 1.2 Output counts
print(f"Dataset output counts:\n{dataset.sentiment.value_counts()}\n")

# 1.3 Encode output column into binary
dataset.sentiment.replace('positive', 1, inplace=True)
dataset.sentiment.replace('negative', 0, inplace=True)
print(f"Dataset head after encoding :\n{dataset.head(10)}\n")

## 2 | Data cleaning ##
"""Clean dataset reviews as following:
1. Remove HTML tags
2. Remove special characters
3. Convert everything to lowercase
4. Remove stopwords
5. Stemming
"""

# 2.1 Remove HTML tags
def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text)

dataset.review = dataset.review.apply(clean)
print(f"Review sample after removing HTML tags : \n{dataset.review[0]}\n")

# 2.2 Remove special characters
def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

dataset.review = dataset.review.apply(is_special)
print(f"Review sample after removing special characters : \n{dataset.review[0]}\n")

# 2.3 Convert everything to lowercase
def to_lower(text):
    return text.lower()

dataset.review = dataset.review.apply(to_lower)
print(f"Review sample after converting everything to lowercase : \n{dataset.review[0]}\n")

#2.4 Remove stopwords
def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

dataset.review = dataset.review.apply(rem_stopwords)
print(f"Review sample after removing stopwords : \n{dataset.review[0]}\n")

# 2.5 Stem the words
def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

dataset.review = dataset.review.apply(stem_txt)
print(f"Review sample after stemming the words : \n{dataset.review[0]}\n")

## 3 | Model Creation ##
"""Create model to fit it to the data"""

# 3.1 Creating Bag Of Words (BOW)
X = np.array(dataset.iloc[:,0].values)
y = np.array(dataset.sentiment.values)
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(dataset.review).toarray()
print(f"=== Bag of words ===\n")
print(f"BOW X shape : {X.shape}")
print(f"BOW y shape : {y.shape}\n")

# 3.2 Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
print(f"Train shapes : X = {X_train.shape}, y = {y_train.shape}")
print(f"Test shapes  : X = {X_test.shape},  y = {y_test.shape}\n")

# 3.3 Defining the models and Training them
gnb, mnb, bnb = GaussianNB(), MultinomialNB(alpha=1.0,fit_prior=True), BernoulliNB(alpha=1.0,fit_prior=True)
gnb.fit(X_train, y_train)
mnb.fit(X_train, y_train)
bnb.fit(X_train, y_train)

# 3.4 Save the three models
joblib.dump(gnb, "Models/MRSA_gnb.pkl")
joblib.dump(mnb, "Models/MRSA_mnb.pkl")
joblib.dump(bnb, "Models/MRSA_bnb.pkl")

# 3.5 Make predictions
ypg = gnb.predict(X_test)
ypm = mnb.predict(X_test)
ypb = bnb.predict(X_test)

## 4 | Model Evaluation ##
"""Evaluate model performance"""
print(f"Gaussian accuracy    =  {round(accuracy_score(y_test, ypg), 2)*100} %")
print(f"Multinomial accuracy =  {round(accuracy_score(y_test, ypm), 2)*100} %")
print(f"Bernoulli accuracy   =  {round(accuracy_score(y_test, ypb), 2)*100} %")


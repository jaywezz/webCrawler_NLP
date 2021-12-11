#warnings :)
import warnings
import os
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from review import Sentiment_Analyzer

from sklearn.metrics import accuracy_score, precision_score, recall_score


Sentiment_Analyzer()
warnings.filterwarnings('ignore')

# Local directory
Reviewdata = pd.read_csv('train.csv')

print(Reviewdata.shape)
print(Reviewdata.head())

# print(Reviewdata.describe().transpose())
 
# Cleaning the data.
count = Reviewdata.isnull().sum().sort_values(ascending=False)
percentage = ((Reviewdata.isnull().sum()/len(Reviewdata)*100)).sort_values(ascending=False)
missing_data = pd.concat([count, percentage], axis=1,
keys=['Count', 'Percentage'])

# print('Count and percentage of missing values for the columns:')

# print(missing_data)

#Removing columns
Reviewdata.drop(columns=['User_ID', 'Browser_Used', 'Device_Used'], inplace=True)
#This function converts to lower-case, removes square bracket, removes numbers and punctuation
def text_clean_1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

cleaned1 = lambda x: text_clean_1(x)
Reviewdata['cleaned_description'] = pd.DataFrame(Reviewdata.Description.apply(cleaned1))
# print(Reviewdata.head(10))

# Apply a second round of cleaning
def text_clean_2(text):
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

cleaned2 = lambda x: text_clean_2(x)
# Let's take a look at the updated text
Reviewdata['cleaned_description_new'] = pd.DataFrame(Reviewdata['cleaned_description'].apply(cleaned2))
# print(Reviewdata.head(10))


Independent_var = Reviewdata.cleaned_description_new
Dependent_var = Reviewdata.Is_Response

IV_train, IV_test, DV_train, DV_test = train_test_split(Independent_var, Dependent_var, test_size = 0.1, random_state = 225)
#
# print('IV_train :', len(IV_train))
# print('IV_test  :', len(IV_test))
# print('DV_train :', len(DV_train))
# print('DV_test  :', len(DV_test))



tvec = TfidfVectorizer()
clf2 = LogisticRegression(solver="lbfgs")

model = Pipeline([('vectorizer', tvec), ('classifier', clf2)])

model.fit(IV_train, DV_train)

predictions = model.predict(IV_test)

print(confusion_matrix(predictions, DV_test))

print("Accuracy : ", accuracy_score(predictions, DV_test))
print("Precision : ", precision_score(predictions, DV_test, average='weighted'))
print("Recall : ", recall_score(predictions, DV_test, average='weighted'))

example = ["this was a waste of my time, i did not like it"]
reviews = pd.read_csv('reviews.csv')
for review in reviews['body']:
    result = model.predict([review])
    print(result)



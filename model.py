import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score, classification_report
import pickle

# Load the csv file
import json
f = open('./complaints.json')
rawdata = json.load(f)

raw_data = rawdata['data']

data = pd.DataFrame(raw_data)
print(data.head())

# Select independent and dependent variable
X = data['text']
y = data['complaint_id']

X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=72)
X_train.shape,y_train.shape,X_test.shape,y_test.shape

# Instantiate the model
text_clf_RF = Pipeline([('tfidf',TfidfVectorizer()),('RF',RandomForestClassifier())])

# Fit the model
text_clf_RF.fit(X_train,y_train)

# Make pickle file of our model
pickle.dump(text_clf_RF, open("model_RF_v2.pkl", "wb"))

import os
os.getcwd()

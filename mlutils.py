from math import pi
from numpy.lib.twodim_base import mask_indices
from sklearn import datasets
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import os
import pickle
from train import train_model,load_model,load_transformer
import json
import pandas as pd

# define a Gaussain NB classifier
#clf = GaussianNB()
classifiers=[GaussianNB(),RandomForestClassifier(n_estimators=200),KNeighborsClassifier(n_neighbors=6,n_jobs=-1),LogisticRegression(max_iter=1000),GradientBoostingClassifier(n_estimators=200,learning_rate=0.001,max_depth=5)]
names=['gaussianNB','randomforestclassifier','KNNClassifier','LogisticRegression','GradientBoostedClassifier']
categorical_features=['Status_of_existing_checking_account', 'Credit_history', 'Purpose',
       'Savings_accountbonds', 'Present_employment_since',
       'Personal_status_and_sex', 'Other_debtors__guarantors', 'Property',
       'Other_installment_plans', 'Housing', 'Job', 'Telephone',
       'foreign_worker']


# define the class encodings and reverse encodings
classes = {0: "Bad Risk", 1: "Good Risk"}
r_classes = {y: x for x, y in classes.items()}

with open('data/columns.json') as f:
    columns=list(json.load(f)['0'].values())

# function to predict the flower using the model
def predict(query_data):
    #clf=pickle.load(open('model.pkl','rb'))
    clf=load_model()
    feature_transformer=load_transformer()
    #x = list(query_data.dict().values())
    print(query_data.dict())
    df=pd.DataFrame([query_data.dict().values()],columns=query_data.dict().keys())
    x=feature_transformer.transform(df)
    prediction = clf.predict(x.reshape(1,-1))[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    #clf=pickle.load(open('model.pkl','rb'))
    clf=load_model()
    feature_transformer=load_transformer()
    #X = [list(d.dict().values())[:-1] for d in data]
    df=pd.DataFrame([data.dict().values()],columns=data.dict().keys())
    #y = [r_classes[d.credit_risk_rating] for d in data]
    y=[r_classes[d] for d in df.loc[:,'credit_risk_rating'].values]
    X=feature_transformer.transform(df.iloc[:,:-1])

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)
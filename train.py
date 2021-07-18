from math import pi
from numpy.lib.twodim_base import mask_indices
from sklearn import datasets
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,ElasticNet
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import os
import pickle

# define a Gaussain NB classifier
#clf = GaussianNB()
classifiers=[GaussianNB(),RandomForestClassifier(n_estimators=200),KNeighborsClassifier(n_neighbors=6,n_jobs=-1),LogisticRegression(max_iter=1000),GradientBoostingClassifier(n_estimators=200,learning_rate=0.001,max_depth=5)]
names=['gaussianNB','randomforestclassifier','KNNClassifier','LogisticRegression','GradientBoostedClassifier']



# define the class encodings and reverse encodings
classes = {0: "Bad Risk", 1: "Good Risk"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def train_model():
    # load the dataset from the official sklearn datasets
    dataset=pd.read_csv('data/cleanedcreditdata.csv')
    X, y = dataset.iloc[:,:-1],dataset.iloc[:,-1]

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model_list=[]
    accuracy=[]
    for clf in classifiers:
        #clf.fit(X_train, y_train)
        clf.fit(X_train,y_train)

        # calculate the print the accuracy score
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"Model {clf} trained with accuracy: {round(acc, 3)}")
        accuracy.append(acc)
        model_list.append(clf)
        acc=0

    
    acc_index=np.argmax(accuracy)

    loaded_model=model_list[acc_index]
    print(f"loading {loaded_model} with accuracy {accuracy[acc_index]}")
    pickle.dump(loaded_model,open('model.pkl','wb'))

def load_model():
    return pickle.load(open('model.pkl','rb'))

def load_transformer():
    return pickle.load(open('columntransformer.pkl','rb'))

if __name__=="__main__":
    #train your model
    train_model()
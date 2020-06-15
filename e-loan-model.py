#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import time


#importing the dataset
dataset=pd.read_csv("Financial_data.csv")

user=dataset["entry_id"]
response=dataset["e_signed"]
dataset=dataset.drop(columns=["entry_id","e_signed"])

#Onehotencoding
dataset=pd.get_dummies(dataset)
dataset.columns
dataset=dataset.drop(columns='pay_schedule_weekly')

#Spltting into training and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataset,response,test_size=0.2,random_state=0)





#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train2=pd.DataFrame(sc_X.fit_transform(X_train))
X_train2.columns=X_train.columns.values
X_train2.index=X_train.index.values
X_test2=pd.DataFrame(sc_X.transform(X_test))
X_test2.columns=X_test.columns.values
X_test2.index=X_test.index.values
X_train=X_train2
X_test=X_test2


#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(penalty="l1")
classifier.fit(X_train,y_train)


y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
acc=accuracy_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
rec=recall_score(y_test,y_pred)

results=pd.DataFrame([["Logistic Regression",acc,f1,prec,rec]],columns=["Model","Accuracy","F1 Score","Precision Score","Recall Score"])


#SVM Linear

from sklearn.svm import SVC
classifier=SVC(kernel="linear")
classifier.fit(X_train,y_train)


y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
acc=accuracy_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
rec=recall_score(y_test,y_pred)

model_results=pd.DataFrame([["SVM(Linear)",acc,f1,prec,rec]],columns=["Model","Accuracy","F1 Score","Precision Score","Recall Score"])
results=results.append(model_results)

#SVM RBF

from sklearn.svm import SVC
classifier=SVC(kernel="rbf")
classifier.fit(X_train,y_train)


y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
acc=accuracy_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
rec=recall_score(y_test,y_pred)

model_results=pd.DataFrame([["SVM(rbf)",acc,f1,prec,rec]],columns=["Model","Accuracy","F1 Score","Precision Score","Recall Score"])
results=results.append(model_results)


#Random Forest

from sklearn.ensemble import RandomForestClassifier as RFC
classifier=RFC(n_estimators=100)
classifier.fit(X_train,y_train)


y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
acc=accuracy_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
rec=recall_score(y_test,y_pred)

model_results=pd.DataFrame([["Random Forest",acc,f1,prec,rec]],columns=["Model","Accuracy","F1 Score","Precision Score","Recall Score"])
results=results.append(model_results)


#Grid Search

from sklearn.model_selection import GridSearchCV
params=[{"n_estimators":[100,400,1000],
         "criterion":["gini",'entropy'],
         "bootstrap":[True,False],
         "verbose":[0,1,2]
         }]
gridsearch=GridSearchCV(estimator=classifier,param_grid=params,scoring="accuracy",cv=10,n_jobs=-1)
gridsearch=gridsearch.fit(X_train,y_train)
best_param=gridsearch.best_params_
best_score=gridsearch.best_score_


cm=confusion_matrix(y_test,y_pred)

#Final Results
final_results=pd.concat([y_test,user],axis=1).dropna()
final_results["prediction"]=y_pred
final_results=final_results[["entry_id","e_signed","prediction"]]
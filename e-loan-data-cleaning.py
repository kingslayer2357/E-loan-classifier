# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:58:06 2020

@author: kingslayer
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#importing the dataset
dataset=pd.read_csv("P39-Financial-Data.csv")

dataset.describe()
dataset.columns
dataset.dtypes

#Histogram
dataset2=dataset.drop(columns="pay_schedule")
plt.suptitle("Histogram")
for i in range(1,dataset2.shape[1]+1):
    plt.figure(figsize=(20,20))
    plt.subplot(5,5,i)
    f=plt.gca()
    f.set_title(dataset2.columns.values[i-1])
    vals=np.size(dataset2.iloc[:,i-1].unique())
    plt.hist(dataset2.iloc[:,i-1],bins=vals,color="green")
    plt.show()
    
    
dataset.isna().any()    
dataset.isna().sum()



#Correlation plot
dataset2.drop(columns=["e_signed",'entry_id']).corrwith(dataset.e_signed).plot.bar(figsize=(20,10),title="Correlation",rot=45,grid=True)


#Correlation matrix
plt.figure(figsize=(50,50))
sns.heatmap(dataset2.corr(),annot=True)


#Feature Engineering
dataset["months_employed"]=dataset["years_employed"]*12 + dataset["months_employed"]
dataset["personal_account_m"]=dataset["personal_account_y"]*12 + dataset["personal_account_m"]
dataset=dataset.drop(columns=["years_employed","personal_account_y"])


dataset.drop(columns=["entry_id","pay_schedule","e_signed"]).corrwith(dataset.e_signed).plot.bar(figsize=(20,10),title="CORRELATION",rot=45,grid=True)


plt.figure(figsize=(30,30))
corr=dataset.drop(columns=["entry_id","pay_schedule","e_signed"]).corr()
mask=np.zeros_like(corr,dtype=bool)
mask[np.triu_indices_from(mask)]=True
sns.heatmap(corr,annot=True,mask=mask)

dataset.to_csv("Financial_data.csv",index=False)
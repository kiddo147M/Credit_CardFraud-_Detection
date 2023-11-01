# importing dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from  sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading dataset
credit_card=pd.read_csv("creditcard.csv")

#first five rows of dataset
credit_card.head()

# lats five rows
credit_card.tail()

# Information about dataset
credit_card.info()

# Null values 
credit_card.isnull().sum()

# Legit and Fraud transactions before preprocessing
credit_card['Class'].value_counts()

# Creating dataframes using column Class
legit=credit_card[credit_card.Class==0]
fraud=credit_card[credit_card.Class==1]

# Basic info of legit and fraud transactions
legit.Amount.describe()
fraud.Amount.describe()

# Analysis of data
credit_card.groupby('Class').mean()

# Under sampling
No of fraud transactions = 73
legit_sample=legit.sample(n=73)

new_dataset=pd.concat([legit_sample,fraud],axis=0)
new_dataset.head()
new_dataset.tail()

# No of fraud and legit transactions after undersampling
new_dataset['Class'].value_counts()

# Grouping data based on column Class
new_dataset.groupby('Class').mean()

# Splitting the data into Features and Target
X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset['Class']

# To split data into training and testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=2)

#Model training
model=LogisticRegression()

# training the Logistic Regression Model with Training data
model.fit(X_train,Y_train)

# Model Evaluation

# Accuracy Score

# accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)

# accuracy on test data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)




'''
while using this as a template for any neural network model,we 
can use this program as such,but if u want to make changes to the data 
set furthur like filling the missing data and encoding categorial data then remove '#' with
lines having '$' 

'''

#data preprocessing

#importing the libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

#importing the dataset   
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

#Taking care of the missing Data
# $ from sklearn.preprocessing import Imputer
# $ imputer = Imputer(missing_values = 'NaN',strategy = 'mean',axis=0)
# $ imputer = imputer.fit(x[:,1:3])
# $ x[:,1:3] = imputer.transform(x[:,1:3])

#Encoding Categorial data
# $ from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# $ labelEncoder_x = LabelEncoder()
# $ x[:,0] = labelEncoder_x.fit_transform(x[:,0])
# $ onehotencoder = OneHotEncoder(categorical_features = [0] )
# $ x = onehotencoder.fit_transform(x).toarray()
# $ labelEncoder_y = LabelEncoder()
# $ y = labelEncoder_y.fit_transform(y) 


#splitting the dataset into the Training set and test set
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


#feature-scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train =sc_x.fit_transform(x_train)
x_test =sc_x.transform(x_test)

'''
Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. 
It is performed during the data pre-processing to handle highly varying magnitudes or values or units. 
If feature scaling is not done, then a machine learning algorithm tends to weigh greater values,
 higher and consider smaller values as the lower values, regardless of the unit of the values

 '''
 
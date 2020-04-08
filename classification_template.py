# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 19:16:57 2020

@author: Quickhotshot612
"""

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

#importing the dataset   
dataset = pd.read_csv('Churn_Modelling_business_dataset.csv')
x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)

#feature-scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train =sc_x.fit_transform(x_train)
x_test =sc_x.transform(x_test)


#Enter your classifier here
#fit your classifier here

#predicting the test results
y_pred = classifier.predict(x_test)

#making the confusion matrix
#here confusion_matrix is not a class,it is a function
#to check for the performance of the model,no of correct and incorrect predictions
from sklearn.metrics import confusion_matrix  
conf_mat = confusion_matrix(y_test,y_pred)

#visualising our final results
from matplotlib.colors import ListedColormap
x_set,y_set= x_train,y_train
x1,x2 = np.meshgrid(np.arange(start =  x_set[:,0].min() - 1,stop = x_set[:,0].max() + 1,step = 0.01),
                    np.arange(start =  x_set[:,0].min() - 1,stop = x_set[:,0].max() + 1,step = 0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                     alpha = 0.75,cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j ,0],x_set[y_set == j,1],
                     c = ListedColormap(('red','green'))(i),label=j)
plt.title('Analysis(Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#now visualising the test results
from sklearn.metrics import confusion_matrix  
conf_mat = confusion_matrix(y_test,y_pred)

#visualising our final results
from matplotlib.colors import ListedColormap
x_set,y_set= x_test,y_test
x1,x2 = np.meshgrid(np.arange(start =  x_set[:,0].min() - 1,stop = x_set[:,0].max() + 1,step = 0.01),
                    np.arange(start =  x_set[:,0].min() - 1,stop = x_set[:,0].max() + 1,step = 0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                     alpha = 0.75,cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j ,0],x_set[y_set == j,1],
                     c = ListedColormap(('red','green'))(i),label=j)
plt.title('Logistic Regression Analysis(Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
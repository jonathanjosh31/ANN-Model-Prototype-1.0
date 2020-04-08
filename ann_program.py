#Installed ALL reuired packages such as Tensorflow,keras and Theano

#Classification template is to classify the test and the training set,to prepare efficiently the model. 
#Data_preprocessing template is to peprocess the data before sending it to the particular model either a ML/DL model


# ------------------data preprocessing---------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset   
dataset = pd.read_csv('Churn_Modelling_business_dataset.csv')
#This dataset also says whether the customer has left the bank or not within six months of period.
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values


#Encoding Categorial data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoder_x_country = LabelEncoder()
x[:,1] = labelEncoder_x_country.fit_transform(x[:,1])
labelEncoder_x_gender = LabelEncoder()
x[:,2] = labelEncoder_x_gender.fit_transform(x[:,2])
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float)
x = x[:,1:]
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y) 



#Splitting into Training set and test set
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#feature-scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train =sc_x.fit_transform(x_train)
x_test =sc_x.transform(x_test)

# ------------------data preprocessing---------------------------


#-------------------------The ANN ---------------------------

#Importing the required keras libraries,packages and modules.

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Initialisation of the ANN
#That is now we are creating a classifier which is going to become our future neural network.We build a classifier since this is a classification problem and also we need to predict the output using the dataset.
ann_classifier = Sequential()  #This is our ANN classifier

#Creating Input and Hidden Layers
ann_classifier.add(Dense(6,kernel_initializer='uniform',activation = 'relu',input_dim=11))  #This is our input layer and the first hidden layer.
ann_classifier.add(Dense(6,kernel_initializer='uniform',activation = 'relu')) #Second hidden layer.Also we didn't specify the input_dim argument here because this layer will know what to expect frm the before hidden layer,whereas the first layer would'nt know because there are no hidden layers before only the input layer.Hence the no of nodes of the input layer are specified. 
ann_classifier.add(Dense(1,kernel_initializer='uniform',activation = 'sigmoid')) #Ouput Layer

#Now compiling the ANN
ann_classifier.compile(optimizer='adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

#Fitting the ANN to teh training set
ann_classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100) 

#-------------------------The ANN ---------------------------

# ------------------------Prediction of Results-----------------------------
#predicting the test results
y_pred = ann_classifier.predict(x_test)
#changing it into either true or false
y_pred = (y_pred > 0.5)
#making the confusion matrix
#here confusion_matrix is not a class,it is a function
#to check for the performance of the model,no of correct and incorrect predictions
from sklearn.metrics import confusion_matrix  
conf_mat = confusion_matrix(y_test,y_pred)

accuracy =(conf_mat[0][0] + conf_mat[1][1])/(conf_mat[0][0]+ conf_mat[0][1] + conf_mat[1][0] + conf_mat[1][1]) * 100

# ------------------------Prediction of Results-----------------------------
'''
#-----------------------Visualisation of Results -----------------------------

#Visualisation of the Training Sets

from matplotlib.colors import ListedColormap
x_set,y_set= x_train,y_train
x1,x2 = np.meshgrid(np.arange(start =  x_set[:,0].min() - 1,stop = x_set[:,0].max() + 1,step = 0.01),
                    np.arange(start =  x_set[:,0].min() - 1,stop = x_set[:,0].max() + 1,step = 0.01))
plt.contourf(x1,x2,ann_classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
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


#Visualisation of the Test Sets
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



#-----------------------Visualisation of Results -----------------------------
'''
# xgboost


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import math as mt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation, tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

def loadData():
    
    dataset = pd.read_csv('onlyx.csv') #loading dataset of parameters
    
    newdata1=dataset.rolling_mean(window = 7)#applying time lags for variouds days say 7 days for example


    data2=pd.read_csv('price.csv') #loading price data


    
    bigdata = dataset.join(data2)#joining the datasets

    #dropping NAN values
    finaldata=bigdata.dropna()

    #splitting x and y
    X= finaldata.iloc[:, [0,1,2,3]].values
    y= finaldata.iloc[:,4].values
    X=X.astype(float)
    y=y.astype(float)

    X=X[~np.isnan(X).any(axis=1)]


#splitting for correlation
x1=finaldata.iloc[:,0].values
x2=finaldata.iloc[:,1].values
x3=finaldata.iloc[:,2].values
x4=finaldata.iloc[:,3].values
x1=x1.astype(float)
x2=x2.astype(float)
x3=x3.astype(float)
x4=x4.astype(float)


#checking correlation
def correlation():
    from scipy.stats import pearsonr
    #for each type of parameter chechking the correlation
    r,p=pearsonr(x1,y)

    print ("r", r)  
    print( "p", p)

    r,p=pearsonr(x2,y)
    print ("r", r)
    print( "p", p)


    r,p=pearsonr(x3,y)
    print ("r", r)
    print( "p", p)

    r,p=pearsonr(x4,y)
    print ("r", r)
    print( "p", p)

def datasplit():
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)



def prediction():
    xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
    traindf, testdf = train_test_split(X_train, test_size = 0.3)
    xgb.fit(X_train,y_train)
    predictions = xgb.predict(X_test)
    print(explained_variance_score(predictions,y_test))    
    



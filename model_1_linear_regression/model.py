import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae, r2_score as r2s
lr = LinearRegression()

df=pd.read_csv('adds.csv')
x_train,x_test,y_train,y_test=train_test_split(df.iloc[:,:-1],df.iloc[:,-1],test_size=0.2,random_state=42)
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
"""
mae=mae(y_test,y_pred)
print(mae)
print(r2s(y_test,y_pred))
def predict(a,b,c):
    feature=np.array([[a,b,c]])
    prediction=lr.predict(feature)
    return prediction
sales=predict(230.1,37.8,69.2)
print(sales)

"""
pickle.dump(lr,open('lr.pkl','wb'))



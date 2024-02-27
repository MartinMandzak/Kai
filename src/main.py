"""Kai v2.0.0"""


#imports
import numpy as np
import pandas as pd
import plotly.express as px

#VARS
DATA = pd.read_csv('telecom_customer_churn.csv')
#data preprocessing / did data_copy.columns
data_copy = DATA.copy()
data_copy.drop(['Customer ID','Zip Code','Longitude', 'Latitude','Churn Category', 'Churn Reason'],
                axis = 'columns', inplace = True)
data_copy.shape




#methods
'''
~data inverts booleans, np.nan (notANumber) np.inf => infinity
any(1) checks for True booleans
'''
def clean_dataset(data):
    assert isinstance(data, pd.DataFrame)
    data.dropna(inplace=True)
    to_be_kept = ~data.isin([np.nan, np.inf, -np.inf]).any(1)
    return data[to_be_kept]



#main
clean_dataset(data_copy)
data_copy = data_copy.interpolate()
data_copy.dropna()

print(data_copy.head())


#graphs
g1 = px.histogram(data_copy, x = 'Age')
#g1.show()

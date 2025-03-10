import pandas as pd
import numpy as np
from datetime import datetime

#Statistical LTV
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data, summary_data_from_transaction_data

#ML Approach to LTV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Evaluation
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns


data= pd.read_excel('./data/Online Retail.xlsx')
data['InvoiceDate'] = pd.to_datetime(data.InvoiceDate, format = '%d/%m/%Y %H:%M')

data['date'] = pd.to_datetime(data.InvoiceDate.dt.date)
data['time'] = data.InvoiceDate.dt.time
data['hour'] = data['time'].apply(lambda x: x.hour)
data['dayofweek'] = data['date'].apply(lambda x: x.dayofweek)
data['total'] = data['Quantity'] * data['UnitPrice']
print(data['date'].max() - data['date'].min())
#Dataset info
print(f'Total Number of Purchases: {data.shape[0]}')
print(f'Total Number of transactions: {data.InvoiceNo.nunique()}')
print(f'Total Unique Days: {data.date.nunique()}')
print(f'Total Unique Customers: {data.CustomerID.nunique()}')
print(f"We are predicting {(data['date'].max() - datetime(2011, 9, 11)).days} days")


#graphs
plt.rcParams['figure.figsize'] = (10,7)
#rev_fig, (ax1, ax2) = plt.subplots(1,2)
#data.groupby('date')['total'].sum().plot(ax=ax1)
#data.groupby('Country')['total'].sum().plot(ax=ax2)

data.groupby('Country')['total'].sum().plot(kind='bar')
plt.show()

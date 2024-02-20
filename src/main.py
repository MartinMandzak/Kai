"""Regression

idk what im even doing"""

#imports

import pandas as pd
import nasdaqdatalink as ndl
ndl.ApiConfig.api_key = "cV_5mV1fxDi_RfMox_wQ"
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, datasets, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#vars   X == features Y == labels
data = ndl.get('LBMA/GOLD', start_date = "1999-12-31")
data['USD (AVG)'] = (data['USD (AM)'] + data['USD (PM)']) / 2
data['GBP (AVG)'] = (data['GBP (AM)'] + data['GBP (PM)']) / 2
data['EURO (AVG)'] = (data['EURO (AM)'] + data['EURO (PM)']) / 2
data = data[['USD (AVG)','GBP (AVG)','EURO (AVG)']]
data.fillna(-1, inplace = True)

k = 0.01
fc_out = int(math.ceil(k*len(data)))
data['EST. EURO'] = data['EURO (AVG)'].shift(-fc_out)
data['EST. GBP'] = data['GBP (AVG)'].shift(-fc_out)
data['EST. USD'] = data['USD (AVG)'].shift(-fc_out)
data.dropna(inplace=True)

X_euro = np.array(data.drop(['EST. EURO'],axis=1))
X_gbp = np.array(data.drop(['EST. GBP'],axis=1))
X_usd = np.array(data.drop(['EST. USD'],axis=1))
Y_euro = np.array(data['EST. EURO'])
Y_gbp = np.array(data['EST. GBP'])
Y_usd = np.array(data['EST. USD'])

X_euro = preprocessing.scale(X_euro)
X_gbp = preprocessing.scale(X_gbp)
X_usd = preprocessing.scale(X_usd)

#Train/test
X_euro_train, X_euro_test, X_gbp_train, X_gbp_test, X_usd_train, X_usd_test, Y_euro_train, Y_euro_test, Y_gbp_train, Y_gbp_test, Y_usd_train, Y_usd_test = train_test_split(X_euro,X_gbp,X_usd, Y_euro,Y_gbp,Y_usd, test_size=0.2, random_state=42)

model_euro = LinearRegression()
model_gbp = LinearRegression()
model_usd = LinearRegression()

model_euro.fit(X_euro_train, Y_euro_train)
model_gbp.fit(X_gbp_train, Y_gbp_train)
model_usd.fit(X_usd_train, Y_usd_train)

#Predictions
Y_euro_prediction = model_euro.predict(X_euro_test)
Y_gbp_prediction = model_gbp.predict(X_gbp_test)
Y_usd_prediction = model_usd.predict(X_usd_test)

#main
mse_euro = mean_squared_error(Y_euro_test, Y_euro_prediction)
r2_euro = r2_score(Y_euro_test, Y_euro_prediction)

mse_gbp = mean_squared_error(Y_gbp_test, Y_gbp_prediction)
r2_gbp = r2_score(Y_gbp_test, Y_gbp_prediction)

mse_usd = mean_squared_error(Y_usd_test, Y_usd_prediction)
r2_usd = r2_score(Y_usd_test, Y_usd_prediction)

print('EST. EURO Model:')
print(f'Mean Squared Error: {mse_euro}')
print(f'R-squared: {r2_euro}')

print('\nEST. GBP Model:')
print(f'Mean Squared Error: {mse_gbp}')
print(f'R-squared: {r2_gbp}')

print('\nEST. USD Model:')
print(f'Mean Squared Error: {mse_usd}')
print(f'R-squared: {r2_usd}')

#plt
data.index = pd.to_datetime(data.index)
fig, (g1,g2,g3) = plt.subplots(3,1,sharex=False,figsize=(12,12))

g1.scatter(X_euro_test[:,2]*1/k, Y_euro_test, color='black', label='Actual EURO')
g1.plot(X_euro_test[:,2]*1/k, Y_euro_prediction, color='blue', linewidth=0.3, label='Predicted EURO')
g1.set_xlabel('Date')
g1.set_ylabel('Value (EURO)')
g1.legend()

g2.scatter(X_gbp_test[:,2]*1/k, Y_gbp_test, color='black', label='Actual GBP')
g2.plot(X_gbp_test[:,2]*1/k, Y_gbp_prediction, color='red', linewidth=0.3, label='Predicted GBP')
g2.set_xlabel('GBP (AVG)')
g2.set_ylabel('EST. GBP')
g2.legend()

g3.scatter(X_usd_test[:,2]*1/k, Y_usd_test, color='black', label='Actual USD')
g3.plot(X_usd_test[:,2]*1/k, Y_usd_prediction, color='green', linewidth=0.3, label='Predicted USD')
g3.set_xlabel('USD (AVG)')
g3.set_ylabel('EST. USD')
g3.legend()

plt.suptitle('LR models')
plt.show()


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
import tensorflow_docs as tfdocs
import modeling_fix as tfdocsFix

#Evaluation
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns

EPOCHS = 30

data= pd.read_excel('./data/Online Retail.xlsx')
data['InvoiceDate'] = pd.to_datetime(data.InvoiceDate, format = '%d/%m/%Y %H:%M')
data['date'] = pd.to_datetime(data.InvoiceDate.dt.date)
data['time'] = data.InvoiceDate.dt.time
data['hour'] = data['time'].apply(lambda x: x.hour)
data['weekend'] = data['date'].apply(lambda x: x.weekday() in [5, 6])
data['dayofweek'] = data['date'].apply(lambda x: x.dayofweek)

#Get revenue column
data['Revenue'] = data['Quantity'] * data['UnitPrice']

#Context data for the revenue (date & customerID)
# Create lookup table with explicit column selection
id_lookup = data[['CustomerID', 'InvoiceNo', 'date']].drop_duplicates()
id_lookup.set_index('InvoiceNo', inplace=True)

# Group and aggregate using named aggregation
transactions_data = (data.groupby('InvoiceNo')
                    .agg(total_revenue=('Revenue', 'sum'))
                    .join(id_lookup))

transactions_data.head()

rfm_train_test = calibration_and_holdout_data(transactions_data, 'CustomerID', 'date',
                                        calibration_period_end='2011-09-10',
                                              monetary_value_col = 'total_revenue')   

#Selecting only customers with positive value in the calibration period (otherwise Gamma-Gamma model doesn't work)
rfm_train_test = rfm_train_test.loc[rfm_train_test['monetary_value_cal'] > 0, :]

print(rfm_train_test.shape)
rfm_train_test.head()
#Train the BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=0.1)
bgf.fit(rfm_train_test['frequency_cal'], rfm_train_test['recency_cal'], rfm_train_test['T_cal'])
rfm_train_test[['monetary_value_cal', 'frequency_cal']].corr()
ggf = GammaGammaFitter(penalizer_coef = 0)
ggf.fit(rfm_train_test['frequency_cal'],
        rfm_train_test['monetary_value_cal'])
#Predict the expected number of transactions in the next 89 days
predicted_bgf = bgf.predict(89,
                        rfm_train_test['frequency_cal'], 
                        rfm_train_test['recency_cal'], 
                        rfm_train_test['T_cal'])
trans_pred = predicted_bgf.fillna(0)

#Predict the average order value
monetary_pred = ggf.conditional_expected_average_profit(rfm_train_test['frequency_cal'],
                                        rfm_train_test['monetary_value_cal'])

#Putting it all together
sales_pred = trans_pred * monetary_pred
actual = rfm_train_test['monetary_value_holdout'] *  rfm_train_test['frequency_holdout']
def evaluate(actual, sales_prediction):
    print(f"Total Sales Actual: {np.round(actual.sum())}")
    print(f"Total Sales Predicted: {np.round(sales_prediction.sum())}")
    print(f"Individual R2 score: {r2_score(actual, sales_prediction)} ")
    print(f"Individual Mean Absolute Error: {mean_absolute_error(actual, sales_prediction)}")
    plt.scatter(sales_prediction, actual)
    plt.xlabel('Prediction')
    plt.ylabel('Actual')      
    plt.show()

def get_features(data, feature_start, feature_end, target_start, target_end):
    """
    Function that outputs the features and targets on the user-level.
    Inputs:
        * data - a dataframe with raw data
        * feature_start - a string start date of feature period
        * feature_end - a  string end date of feature period
        * target_start - a  string start date of target period
        * target_end - a  string end date of target period
    """
    features_data = data.loc[(data.date >= feature_start) & (data.date <= feature_end), :]
    print(f'Using data from {(pd.to_datetime(feature_end) - pd.to_datetime(feature_start)).days} days')
    print(f'To predict {(pd.to_datetime(target_end) - pd.to_datetime(target_start)).days} days')
    
    #Transactions data features
    total_rev = features_data.groupby('CustomerID')['Revenue'].sum().rename('total_revenue')
    recency = (features_data.groupby('CustomerID')['date'].max() - features_data.groupby('CustomerID')['date'].min()).apply(lambda x: x.days).rename('recency')
    frequency = features_data.groupby('CustomerID')['InvoiceNo'].count().rename('frequency')
    t = features_data.groupby('CustomerID')['date'].min().apply(lambda x: (datetime(2011, 6, 11) - x).days).rename('t')
    time_between = (t / frequency).rename('time_between')
    avg_basket_value = (total_rev / frequency).rename('avg_basket_value')
    avg_basket_size = (features_data.groupby('CustomerID')['Quantity'].sum() / frequency).rename('avg_basket_Size')
    returns = features_data.loc[features_data['Revenue'] < 0, :].groupby('CustomerID')['InvoiceNo'].count().rename('num_returns')
    hour = features_data.groupby('CustomerID')['hour'].median().rename('purchase_hour_med')
    dow = features_data.groupby('CustomerID')['dayofweek'].median().rename('purchase_dow_med')
    weekend =  features_data.groupby('CustomerID')['weekend'].mean().rename('purchase_weekend_prop')
    train_data = pd.DataFrame(index = rfm_train_test.index)
    train_data = train_data.join([total_rev, recency, frequency, t, time_between, avg_basket_value, avg_basket_size, returns, hour, dow, weekend])
    train_data = train_data.fillna(0)
    
    #Target data
    target_data = data.loc[(data.date >= target_start) & (data.date <= target_end), :]
    target_quant = target_data.groupby(['CustomerID'])['date'].nunique()
    target_rev = target_data.groupby(['CustomerID'])['Revenue'].sum().rename('target_rev')
    train_data = train_data.join(target_rev).fillna(0)
    
    return train_data.iloc[:, :-1], train_data.iloc[:, -1]

X_train, y_train = get_features(data, '2011-01-01', '2011-06-11', '2011-06-12', '2011-09-09')
X_test, y_test = get_features(data, '2011-04-02', '2011-09-10', '2011-09-11', '2011-12-09')

def build_model():
    model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=[len(X_train.columns), ]),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(loss='mse',
            optimizer=optimizer,
            metrics=['mae', 'mse'])
    
    return model

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)


model = build_model()
early_history = model.fit(X_train, y_train, 
                    epochs=EPOCHS, validation_split = 0.2, verbose=0, 
                    callbacks=[early_stop, tfdocsFix.EpochDots()])

dnn_preds = model.predict(X_test).ravel()
#Putting the actual and predictions into the same datarame for later comparison
compare_df = pd.DataFrame(index=X_test.index)
compare_df['dnn_preds'] = dnn_preds
compare_df = compare_df.join(sales_pred.rename('stat_pred')).fillna(0)
compare_df['actual'] = y_test

evaluate(compare_df['actual'], compare_df['dnn_preds'])
#graphs

#First 98% of data
plt.rcParams['figure.figsize'] = (10,7)
no_out = compare_df.loc[(compare_df['actual'] <= np.quantile(compare_df['actual'], 0.985)), :]

sns.histplot(no_out['actual'])
sns.histplot(no_out['dnn_preds'])
plt.title('Actual vs DNN Predictions')
plt.show()
sns.histplot(no_out['actual'])
sns.histplot(no_out['stat_pred'])
plt.title('Actual vs BG/NBD Predictions')
plt.show()
'''
rev_fig, (ax1, ax2) = plt.subplots(1,2)
data.groupby('date')['Revenue'].sum().plot(ax=ax1)
data.groupby('Country')['Revenue'].sum().plot(ax=ax2)

#data.groupby('Country')['Revenue'].sum().plot(kind='bar')
plt.show()
'''

top_n = int(np.round(compare_df.shape[0] * 0.2))
print(f'Selecting the first {top_n} users')

#Selecting IDs
dnn_ids = compare_df['dnn_preds'].sort_values(ascending=False).index[:top_n].values
stat_ids = compare_df['stat_pred'].sort_values(ascending=False).index[:top_n].values

#Filtering the data
eval_subset = data.loc[data.date >= '2011-09-10', :]

#Sums
dnn_rev = eval_subset.loc[eval_subset.CustomerID.isin(dnn_ids), 'Revenue'].sum() 
stat_rev = eval_subset.loc[eval_subset.CustomerID.isin(stat_ids), 'Revenue'].sum()


print(f'Top 20% selected by DNN have generated {np.round(dnn_rev)}')
print(f'Top 20% selected by BG/NBD and Gamma Gamma have generated {np.round(stat_rev)}')
print(f'Thats {np.round(dnn_rev - stat_rev)} of marginal revenue')


top_n = int(np.round(compare_df.shape[0] * 0.1))
print(f'Selecting the first {top_n} users')

#Selecting IDs
dnn_ids = compare_df['dnn_preds'].sort_values(ascending=False).index[:top_n].values
stat_ids = compare_df['stat_pred'].sort_values(ascending=False).index[:top_n].values

#Filtering the data
eval_subset = data.loc[data.date >= '2011-09-10', :]

#Sums
dnn_rev = eval_subset.loc[eval_subset.CustomerID.isin(dnn_ids), 'Revenue'].sum() 
stat_rev = eval_subset.loc[eval_subset.CustomerID.isin(stat_ids), 'Revenue'].sum()


print(f'Top 20% selected by DNN have generated {np.round(dnn_rev)}')
print(f'Top 20% selected by BG/NBD and Gamma Gamma have generated {np.round(stat_rev)}')
print(f'Thats {np.round(dnn_rev - stat_rev)} of marginal revenue')

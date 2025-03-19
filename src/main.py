import pandas as pd
import numpy as np
from datetime import datetime

# ML Approach to CLV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import modeling_fix as tfdocsFix

# Evaluation
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

EPOCHS = 200

# Load data
data = pd.read_excel('./data/Online Retail.xlsx')
data['InvoiceDate'] = pd.to_datetime(data.InvoiceDate, format='%d/%m/%Y %H:%M')
data['date'] = pd.to_datetime(data.InvoiceDate.dt.date)
data['time'] = data.InvoiceDate.dt.time
data['hour'] = data['time'].apply(lambda x: x.hour)
data['weekend'] = data['date'].apply(lambda x: x.weekday() in [5, 6])
data['dayofweek'] = data['date'].apply(lambda x: x.dayofweek)

# Get revenue column
data['Revenue'] = data['Quantity'] * data['UnitPrice']

def get_features(data, feature_start, feature_end, target_start, target_end):
    features_data = data.loc[(data.date >= feature_start) & (data.date <= feature_end), :]
    print(f'Using data from {(pd.to_datetime(feature_end) - pd.to_datetime(feature_start)).days} days')
    print(f'To predict {(pd.to_datetime(target_end) - pd.to_datetime(target_start)).days} days')
    
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
    weekend = features_data.groupby('CustomerID')['weekend'].mean().rename('purchase_weekend_prop')
    
    train_data = pd.DataFrame(index=total_rev.index)
    train_data = train_data.join([total_rev, recency, frequency, t, time_between, avg_basket_value, avg_basket_size, returns, hour, dow, weekend])
    train_data = train_data.fillna(0)
    
    target_data = data.loc[(data.date >= target_start) & (data.date <= target_end), :]
    target_rev = target_data.groupby(['CustomerID'])['Revenue'].sum().rename('target_rev')
    train_data = train_data.join(target_rev).fillna(0)
    
    # Apply log transformation to stabilize variance
    train_data['target_rev'] = np.log1p(train_data['target_rev'])
    
    return train_data.iloc[:, :-1], train_data.iloc[:, -1]

X_train, y_train = get_features(data, '2011-01-01', '2011-06-11', '2011-06-12', '2011-09-09')
X_test, y_test = get_features(data, '2011-04-02', '2011-09-10', '2011-09-11', '2011-12-09')

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.layers import BatchNormalization, LeakyReLU

def build_model():
    model = keras.Sequential([
        layers.Dense(64, input_shape=[X_train.shape[1]]),
        BatchNormalization(),
        LeakyReLU(),
        layers.Dropout(0.2),

        layers.Dense(64),
        BatchNormalization(),
        LeakyReLU(),
        layers.Dropout(0.2),

        layers.Dense(32),
        BatchNormalization(),
        LeakyReLU(),

        layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

model = build_model()
early_history = model.fit(X_train, y_train, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[early_stop, tfdocsFix.EpochDots()])

dnn_preds = model.predict(X_test).ravel()
dnn_preds = np.expm1(dnn_preds)  # Reverse log transformation
y_test = np.expm1(y_test)

compare_df = pd.DataFrame({'dnn_preds': dnn_preds, 'actual': y_test}, index=range(len(y_test)))

def evaluate(actual, predictions):
    print(f"Total Sales Actual: {np.round(actual.sum())}")
    print(f"Total Sales Predicted: {np.round(predictions.sum())}")
    print(f"Individual R2 score: {r2_score(actual, predictions)}")
    print(f"Individual Mean Absolute Error: {mean_absolute_error(actual, predictions)}")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # KDE Plot
    sns.kdeplot(actual, color='blue', label='Actual', fill=True, ax=axes[0])
    sns.kdeplot(predictions, color='red', label='Predicted', fill=True, ax=axes[0])
    axes[0].set_xlabel('CLV Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title('KDE Distribution of Actual vs Predicted CLV')
    axes[0].legend()
    
    # Scatter plot
    axes[1].scatter(actual, predictions, alpha=0.5)
    axes[1].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r', lw=2)
    axes[1].set_xlabel('Actual CLV')
    axes[1].set_ylabel('Predicted CLV')
    axes[1].set_title('Actual vs Predicted CLV')
    
    plt.tight_layout()
    plt.show()

evaluate(compare_df['actual'], compare_df['dnn_preds'])


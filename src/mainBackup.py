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

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

EPOCHS = 100

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
    
    return train_data.iloc[:, :-1], train_data.iloc[:, -1]

X_train, y_train = get_features(data, '2011-01-01', '2011-06-11', '2011-06-12', '2011-09-09')
X_test, y_test = get_features(data, '2011-04-02', '2011-09-10', '2011-09-11', '2011-12-09')

def build_model():
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=[len(X_train.columns)]),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(0.000069)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

model = build_model()
early_history = model.fit(X_train, y_train, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[early_stop, tfdocsFix.EpochDots()])

dnn_preds = model.predict(X_test).ravel()
compare_df = pd.DataFrame({'dnn_preds': dnn_preds, 'actual': y_test}, index=X_test.index)


def remove_outliers(df, column):
    """Removes outliers from a dataframe based on IQR method for a given column."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers for actual and predicted LTV
filtered_df = remove_outliers(compare_df, 'actual')
filtered_df = remove_outliers(filtered_df, 'dnn_preds')

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

    # Binned Bar Chart
    bins = np.linspace(0, max(max(actual), max(predictions)), 9)  
    actual_binned = np.histogram(actual, bins=bins)[0]
    predicted_binned = np.histogram(predictions, bins=bins)[0]

    bin_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]

    width = 0.4  
    x = np.arange(len(bin_labels))
    
    axes[1].bar(x - width/2, actual_binned, width=width, label='Actual', color='blue', alpha=0.7)
    axes[1].bar(x + width/2, predicted_binned, width=width, label='Predicted', color='red', alpha=0.7)
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bin_labels, rotation=45)
    axes[1].set_xlabel('CLV Value Range')
    axes[1].set_ylabel('Customer Count')
    axes[1].set_title('Binned Bar Chart of Actual vs Predicted CLV')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# Evaluate after removing outliers
evaluate(filtered_df['actual'], filtered_df['dnn_preds'])
#evaluate(compare_df['actual'], compare_df['dnn_preds'])

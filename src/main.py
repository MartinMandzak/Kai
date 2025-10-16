import pandas as pd
import numpy as np
from datetime import datetime, date

# ML Approach to CLV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import modeling_fix as tfdocsFix

# Evaluation
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

EPOCHS = 300

# Load and preprocess data
data = pd.read_excel('./data/AnonymisedData.xlsx')

# Create essential columns
data['Revenue'] = pd.to_numeric(data['Revenue'], errors = 'coerce').fillna(0)
data['Quantity'] = pd.to_numeric(data['Quantity'], errors = 'coerce').fillna(0)
data['InvoiceDate'] = pd.to_datetime(
    data['Extended Day'].str.replace(',', '') + ' ' + data['Year'].astype(str),
    format='%b %d %Y'
)
data['date'] = data['InvoiceDate'].dt.date

def get_features(data, feature_start, feature_end, target_start, target_end):
    features_data = data.loc[(data.date >= pd.to_datetime(feature_start).date()) & 
                            (data.date <= pd.to_datetime(feature_end).date()), :]
    
    if features_data.empty:
        raise ValueError("Feature dataset is empty! Check date ranges.")
    
    print(f'Using data from {(pd.to_datetime(feature_end) - pd.to_datetime(feature_start)).days} days')
    
    # Feature calculations (without time-related features)
    total_rev = features_data.groupby('CustomerID')['Revenue'].sum().rename('total_revenue').astype(float)
    recency = (features_data.groupby('CustomerID')['date'].max() - 
               features_data.groupby('CustomerID')['date'].min()).apply(lambda x: x.days).rename('recency')
    frequency = features_data.groupby('CustomerID')['InvoiceNo'].nunique().rename('frequency').astype(int)
   
    '''
       HERE

    '''
    latest_date = date(2023,6,11)

    t = (
        features_data.groupby('CustomerID')['date'].min()
        .apply(lambda x: (latest_date - x).days)
        .rename('t')
    ) 



    time_between = (t / frequency).rename('time_between')
    avg_basket_value = (total_rev / frequency).rename('avg_basket_value')
    avg_basket_size = (features_data.groupby('CustomerID')['Quantity'].sum() / 
                      frequency).rename('avg_basket_Size')
    
    returns = features_data.loc[features_data['Revenue'] < 0].groupby('CustomerID')['InvoiceNo'].nunique()
    returns = returns.rename('num_returns').reindex(total_rev.index).fillna(0)

    train_data = pd.concat([
        total_rev, recency, frequency, t, time_between,
        avg_basket_value, avg_basket_size, returns
    ], axis=1).fillna(0)

    # Target processing
    target_data = data.loc[(data.date >= pd.to_datetime(target_start).date()) & 
                         (data.date <= pd.to_datetime(target_end).date()), :]
    
    if target_data.empty:
        raise ValueError("Target dataset is empty! Check date ranges.")
    
    target_rev = target_data.groupby('CustomerID')['Revenue'].sum().rename('target_rev')
    return train_data, target_rev.reindex(train_data.index).fillna(0)


def build_model():
    model = keras.Sequential([
        keras.Input(shape=(n_feats,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(0.000069)
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])
    return model

# Build churn prediction model (classification)
def build_churn_model():
    model = keras.Sequential([
        keras.Input(shape=(n_feats,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    optimizer = tf.keras.optimizers.Adam(0.000069)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def remove_outliers(df, column):
    """Removes outliers from a dataframe based on IQR method for a given column."""
    Q1 = df[column].quantile(0.005)
    Q3 = df[column].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def evaluate(actual, predictions):
    if actual.empty or predictions.size == 0:
        raise ValueError("Evaluation dataset is empty!")
    
    MAE = mean_absolute_error(actual, predictions)
    RMSE = mean_squared_error(actual, predictions)
    print(f"Total Sales Actual: {np.round(actual.sum())}")
    print(f"Total Sales Predicted: {np.round(predictions.sum())}")
    print(f"Individual R2 score: {r2_score(actual, predictions)}")
    print(f"Individual Mean Absolute Error: {MAE}")
    print(f"Individual Mean Squared Error: {RMSE}")
    print(f"RMSE/MAE ratio: {RMSE/MAE}")

    accuracy = accuracy_score(y_test_churn, churn_preds_class)
    print("Churn Model Accuracy:", accuracy)

    plt.figure(figsize=(10, 6))
    #plt.plot(churn_history.history['accuracy'], label='Training Accuracy')
    plt.plot(churn_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Churn Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bins = np.linspace(0, np.percentile(actual, 90), 10)
    
    # KDE Plot
    sns.kdeplot(actual, color='blue', label='Actual', fill=True, bw_adjust=0.5, ax=axes[0])
    sns.kdeplot(predictions, color='red', label='Predicted', fill=True, bw_adjust=0.5, ax=axes[0])
    axes[0].set_xlim(bins[0] - 50000, bins[-1])
    axes[0].set_xlabel('CLV Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title('KDE Distribution of Actual vs Predicted CLV')
    axes[0].legend()

    # Binned Bar Chart
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

'''
X_train, y_train = get_features(data,
                                '2021-01-01',
                                '2023-06-01',
                                '2023-06-02',
                                '2023-12-31')

X_test, y_test = get_features(data,
                              '2022-01-01',
                              '2024-06-01',
                              '2024-06-02',
                              '2024-12-31')

'''
X_train, y_train = get_features(data,
                                '2023-01-01',
                                '2023-06-01',
                                '2023-06-02',
                                '2023-12-31')

X_test, y_test = get_features(data,
                              '2024-01-01',
                              '2024-05-26',
                              '2024-05-27',
                              '2024-12-31')

n_feats = len(X_train.columns)
train_index = X_train.index
test_index = X_test.index

train_columns = X_train.columns
test_columns = X_test.columns

#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
y_train_churn = (y_train == 0).astype(int)
y_test_churn = (y_test == 0).astype(int)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
early_stop_churn = keras.callbacks.EarlyStopping(monitor='val_loss', patience=150)

model = build_model()
early_history = model.fit(X_train, y_train,
                          epochs=EPOCHS,
                          validation_split=0.2,
                          verbose=0,
                          callbacks=[early_stop, tfdocsFix.EpochDots()])

churn_model = build_churn_model()

churn_history = churn_model.fit(
    X_train, y_train_churn,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop_churn, tfdocsFix.EpochDots()])


#X_train = pd.DataFrame(X_train, columns=train_columns, index=train_index)
#X_test = pd.DataFrame(X_test, columns=test_columns, index=test_index)

dnn_preds = model.predict(X_test).ravel()
compare_df = pd.DataFrame({'dnn_preds': dnn_preds, 'actual': y_test}, index=test_index)

# Remove outliers for actual and predicted LTV
filtered_df = remove_outliers(compare_df, 'actual')
filtered_df = remove_outliers(filtered_df, 'dnn_preds')

# Predict churn probabilities on test set
churn_preds = churn_model.predict(X_test).ravel()

# Convert probabilities to binary classification (threshold 0.5)
churn_preds_class = (churn_preds > 0.90).astype(int)

evaluate(filtered_df['actual'], filtered_df['dnn_preds'])


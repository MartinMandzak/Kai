import pandas as pd
import numpy as np
from datetime import datetime, date

# ML Approach to CLV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs

# Evaluation
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

class CyclicLR(keras.callbacks.Callback):
    def __init__(self, base_lr=1e-6, max_lr=1e-4, step_size=2000):
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.iterations = 0

    def on_train_batch_begin(self, batch, logs=None):
        cycle = np.floor(1 + self.iterations / (2 * self.step_size))
        x = np.abs(self.iterations / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        self.model.optimizer.learning_rate.assign(lr)
        self.iterations += 1


EPOCHS = 300

# Load and preprocess data
data = pd.read_excel('./data/AnonymisedData.xlsx')

# Create essential columns
data['Revenue'] = pd.to_numeric(data['Revenue'], errors='coerce').fillna(0)
data['Quantity'] = pd.to_numeric(data['Quantity'], errors='coerce').fillna(0)
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
    
    # Feature calculations
    total_rev = features_data.groupby('CustomerID')['Revenue'].sum().rename('total_revenue').astype(float)
    recency = (features_data.groupby('CustomerID')['date'].max() - 
               features_data.groupby('CustomerID')['date'].min()).apply(lambda x: x.days).rename('recency')
    frequency = features_data.groupby('CustomerID')['InvoiceNo'].nunique().rename('frequency').astype(int)
    
    latest_date = features_data['date'].max()  # Fix: Use max date from feature data
    t = (
        features_data.groupby('CustomerID')['date'].min()
        .apply(lambda x: (latest_date - x).days)
        .rename('t')
    )

    time_between = (t / frequency).replace([np.inf, -np.inf], 0).rename('time_between')
    avg_basket_value = (total_rev / frequency).replace([np.inf, -np.inf], 0).rename('avg_basket_value')
    avg_basket_size = (features_data.groupby('CustomerID')['Quantity'].sum() / 
                      frequency).replace([np.inf, -np.inf], 0).rename('avg_basket_Size')
    
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

def build_model(lr=0.001):
    model = keras.Sequential([
        keras.Input(shape=(n_feats,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])
    return model

def build_churn_model(lr=0.001):
    model = keras.Sequential([
        keras.Input(shape=(n_feats,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def remove_outliers(df, column):
    """Removes outliers from a dataframe based on IQR method."""
    Q1 = df[column].quantile(0.05)
    Q3 = df[column].quantile(0.95)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def evaluate(actual, predictions, y_test_churn_filtered, churn_preds_class, churn_history=None):
    if actual.empty or predictions.size == 0:
        raise ValueError("Evaluation dataset is empty!")
    
    # Regression Metrics
    MAE = mean_absolute_error(actual, predictions)
    MSE = mean_squared_error(actual, predictions)
    RMSE = np.sqrt(MSE)
    print(f"Total Sales Actual: {np.round(actual.sum())}")
    print(f"Total Sales Predicted: {np.round(predictions.sum())}")
    print(f"Individual R2 score: {r2_score(actual, predictions)}")
    print(f"Individual Mean Absolute Error: {MAE}")
    print(f"Root Mean Squared Error: {RMSE}")
    print(f"RMSE/MAE ratio: {RMSE/MAE}")
    
    # Classification Metrics
    if len(y_test_churn_filtered) > 0:
        accuracy = accuracy_score(y_test_churn_filtered, churn_preds_class)
        print("Churn Model Accuracy:", accuracy)
    else:
        print("No samples remaining for churn evaluation")
    
    # First plot - Churn Accuracy (standalone)
    plt.figure(figsize=(6, 5))
    if churn_history is not None:
        plt.plot(churn_history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Churn Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Second plot - KDE and Binned Bar Chart together
    plt.figure(figsize=(15, 5))
    
    # KDE Plot
    plt.subplot(1, 2, 1)
    bins = np.linspace(0, np.percentile(actual, 90), 10)
    sns.kdeplot(actual, color='blue', label='Actual', fill=True, bw_adjust=0.5)
    sns.kdeplot(predictions, color='red', label='Predicted', fill=True, bw_adjust=0.5)
    plt.xlim(bins[0] - 50000, bins[-1])
    plt.xlabel('CLV Value')
    plt.ylabel('Density')
    plt.title('KDE Distribution of Actual vs Predicted CLV')
    plt.legend()
    
    # Binned Bar Chart
    plt.subplot(1, 2, 2)
    actual_binned = np.histogram(actual, bins=bins)[0]
    predicted_binned = np.histogram(predictions, bins=bins)[0]
    bin_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
    width = 0.4
    x = np.arange(len(bin_labels))
    
    plt.bar(x - width/2, actual_binned, width=width, label='Actual', color='blue', alpha=0.7)
    plt.bar(x + width/2, predicted_binned, width=width, label='Predicted', color='red', alpha=0.7)
    plt.xticks(x, bin_labels, rotation=45)
    plt.xlabel('CLV Value Range')
    plt.ylabel('Customer Count')
    plt.title('Binned Bar Chart of Actual vs Predicted CLV')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Data loading and feature engineering
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

# Churn labels
y_train_churn = (y_train == 0).astype(int)
y_test_churn = (y_test == 0).astype(int)

# Model training
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
early_stop_churn = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=150, mode='max')

clr =CyclicLR(base_lr = 0.000069, max_lr = 0.00069)

# Rebuild model with optimal learning rate (adjust based on plot)
model = build_model(lr=0.000069)
history = model.fit(X_train, y_train,
                   epochs=EPOCHS,
                   validation_split=0.2,
                   verbose=0,
                   callbacks=[early_stop, clr])

# Churn model training
churn_model = build_churn_model(lr=0.000069)
churn_history = churn_model.fit(
    X_train, y_train_churn,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop_churn, clr])

# Predictions
dnn_preds = model.predict(X_test).ravel()
compare_df = pd.DataFrame({'dnn_preds': dnn_preds, 'actual': y_test}, index=test_index)

# Remove outliers
filtered_df = remove_outliers(compare_df, 'actual')
filtered_df = remove_outliers(filtered_df, 'dnn_preds')

# Filter churn predictions to match filtered_df
mask = filtered_df.index
churn_preds = churn_model.predict(X_test).ravel()
churn_preds_class = (churn_preds > 0.9).astype(int)  # Adjusted threshold
churn_preds_class = churn_preds_class[X_test.index.isin(mask)]
y_test_churn_filtered = y_test_churn[X_test.index.isin(mask)]

# Evaluation
evaluate(filtered_df['actual'], filtered_df['dnn_preds'], y_test_churn_filtered, churn_preds_class, churn_history)

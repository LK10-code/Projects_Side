# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping

# # Generate sample data (replace this with your actual dataset)
# data = pd.read_csv('D:\\BEDEGREE\\side_project\\BMW_Data.csv')
# np.random.seed(42)
# dates = pd.date_range(start='1996-11-08', periods=2000, freq='D')
# data = pd.DataFrame({
#     'Adj_Close': np.random.rand(len(dates)) * 100,
#     'Close': np.random.rand(len(dates)) * 100,
#     'High': np.random.rand(len(dates)) * 100,
#     'Low': np.random.rand(len(dates)) * 100,
#     'Open': np.random.rand(len(dates)) * 100,
#     'Volume': np.random.rand(len(dates)) * 1000
# }, index=dates)

# # 1. Preprocessing
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(data)

# # Create sequences
# def create_sequences(data, seq_length):
#     X, y = [], []
#     for i in range(len(data) - seq_length):
#         X.append(data[i:i + seq_length, :])
#         y.append(data[i + seq_length, :])  # Predict all features
#     return np.array(X), np.array(y)

# SEQ_LENGTH = 30  # Use the last 30 days for predictions
# X, y = create_sequences(scaled_data, SEQ_LENGTH)

# # Split into train and test sets
# train_size = int(0.8 * len(X))
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]

# # 2. Build the Unified LSTM Model
# model = Sequential([
#     LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2])),
#     Dropout(0.2),
#     LSTM(64, return_sequences=False),
#     Dropout(0.2),
#     Dense(32, activation='relu'),
#     Dense(X.shape[2], activation='linear')  # Output layer for 6 features
# ])

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# # Early stopping to prevent overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # 3. Train the Model
# history = model.fit(
#     X_train, y_train,
#     epochs=50,
#     batch_size=32,
#     validation_split=0.2,
#     callbacks=[early_stopping],
#     verbose=1
# )

# # 4. Evaluate the Model
# loss, mae = model.evaluate(X_test, y_test, verbose=1)
# print(f"Test Loss: {loss}, Test MAE: {mae}")

# # 5. Predictions and RMSE
# predictions = model.predict(X_test)
# rmse = np.sqrt(mean_squared_error(y_test.flatten(), predictions.flatten()))
# print(f"RMSE: {rmse:.2e}")

# # 6. Visualization of Predictions vs Actuals
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# plt.plot(y_test[:, 0], label='Actual Adj_Close')  # Replace with specific feature index
# plt.plot(predictions[:, 0], label='Predicted Adj_Close')
# plt.title('Actual vs Predicted (Adj_Close)')
# plt.xlabel('Time')
# plt.ylabel('Normalized Value')
# plt.legend()
# plt.show()





import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# 1. Load and preprocess the dataset
data = pd.read_csv('D:\\BEDEGREE\\side_project\\BMW_Data.csv')

# Parse dates and sort by date
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data.set_index('Date', inplace=True)

# Check for missing values and fill them
data.fillna(method='ffill', inplace=True)

# Normalize features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :])
        y.append(data[i + seq_length, :])
    return np.array(X), np.array(y)

SEQ_LENGTH = 30
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Split into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Updated model
model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2])),
    Dropout(0.3),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(X.shape[2], activation='linear')
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Training with learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# 4. Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# 5. Predictions and RMSE
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test.flatten(), predictions.flatten()))
print(f"RMSE: {rmse:.2e}")

# 6. Visualization
plt.figure(figsize=(10, 6))
plt.plot(y_test[:, 0], label='Actual Adj_Close')  # Replace with specific feature index
plt.plot(predictions[:, 0], label='Predicted Adj_Close')
plt.title('Actual vs Predicted (Adj_Close)')
plt.xlabel('Time')
plt.ylabel('Normalized Value')
plt.legend()
plt.show()

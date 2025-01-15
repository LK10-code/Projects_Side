import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def feature_prediction(data, feature,epo):
    # Ensure the data has a 'Date' column
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Select the single feature
    data = data[[feature]]

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Prepare the dataset for the LSTM
    def prepare_data(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps, 0])
            y.append(data[i + time_steps, 0])
        return np.array(X), np.array(y)

    time_steps = 60
    X, y = prepare_data(data_scaled, time_steps)
    # Reshape X to be [samples, time_steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build the LSTM model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    epoch = epo
    model.fit(X_train, y_train, epochs=epoch, batch_size=32, validation_data=(X_test, y_test))
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions))
    print(f'Root Mean Squared Error: {rmse}')

    # Save the model to a file
    model.save(f'bmw_model\\set_size_60\\stock_price_prediction_model_{feature}_epoch{epoch}_rms{rmse}.keras')

    # Plot the results
    # plt.figure(figsize=(12, 6))
    # plt.plot(data.index[-len(y_test):], y_test_unscaled, label='Actual Price', color='blue')
    # plt.plot(data.index[-len(predictions):], predictions, label='Predicted Price', color='red')
    # plt.title('Stock Price Prediction')
    # plt.xlabel('Date')
    # plt.ylabel('Stock Price')
    # plt.legend()
    # plt.show()

# features = ['Adj_Close', 'Close', 'High', 'Low', 'Open', 'Volume']
features = ['High', 'Low', 'Open', 'Volume']

for feature in features:
    epoch_size = [2,50,100,200,600]
    for item in epoch_size:
        epo = item
        # Load the data
        dataorg = pd.read_csv('D:\\BEDEGREE\\side_project\\BMW_Data.csv')  # Replace with your CSV file

        print(f'feature {feature} epo {epo}')
        feature_prediction(dataorg, feature,epo)

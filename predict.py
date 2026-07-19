import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Download live stock data from Yahoo Finance (Apple Inc. as an example)
ticker = "AAPL" 
data = yf.download(ticker, start="2022-01-01", end="2026-01-01")

# 2. Create the target column (Predicting Next Day's Close Price)
data['Next_Close'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Separate features (X) and target labels (y)
X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
y = data['Next_Close'].values

# 3. Split dataset into 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Machine Learning model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the model performance
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy (R² Score): {accuracy:.2f}")

# 6. Generate predictions and plot them
predictions = model.predict(X_test)

plt.figure(figsize=(10, 5))
plt.plot(y_test[:50], label="Actual Prices", color="blue")
plt.plot(predictions[:50], label="Predicted Prices", color="red", linestyle="--")
plt.title(f"{ticker} Stock Price Prediction Dashboard")
plt.xlabel("Test Data Points")
plt.ylabel("Price ($)")
plt.legend()
plt.show()
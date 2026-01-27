# Stock Price Prediction Using CNN (Conv1D)

## 📌 Overview

This project demonstrates how Convolutional Neural Networks (CNNs) can be applied to time-series forecasting for stock prices. Using historical closing prices of Apple Inc. (AAPL), the model learns patterns in past data to predict future prices and estimate uncertainty using confidence intervals.

The project focuses on:
- Predicting stock prices on unseen test data
- Forecasting the next 7 days of closing prices

---

## 📊 Dataset

- Source: Yahoo Finance (yfinance)
- Stock Symbol: AAPL
- Time Period: 2010-01-01 to 2019-12-31
- Feature Used: Closing Price

Only the closing price is used to simplify the modeling process and focus on price behavior.

---

## 🧠 Model Architecture

The model uses 1D convolutional layers to capture temporal patterns in stock price data.

Input (100 time steps)  
→ Conv1D (64 filters, ReLU)  
→ Batch Normalization  
→ Conv1D (128 filters, ReLU)  
→ MaxPooling1D  
→ Dropout (0.2)  
→ Conv1D (64 filters, ReLU)  
→ MaxPooling1D  
→ Flatten  
→ Dense (50, ReLU)  
→ Dense (1)

- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Epochs: 100
- Batch Size: 16

---

## 🔄 Data Preprocessing

1. Download historical stock data using yfinance
2. Select the Close price column
3. Normalize values using MinMaxScaler
4. Split data into:
   - 80% training data
   - 20% testing data
5. Create sliding windows of 100 days for time-series modeling

---

## 📈 Model Training and Evaluation

- The CNN model is trained on historical price windows
- Performance is evaluated using Root Mean Squared Error (RMSE)
- Predictions are inverse-scaled to obtain actual price values

---

## 🔮 Future Price Prediction

- The trained model predicts the next 7 days of closing prices
- Predictions are generated iteratively using the last available window
- A 95% confidence interval is calculated using the standard deviation of residuals

---

## 📉 Visualizations

The following plots are generated:
1. Training vs Testing Actual Prices
2. Testing Actual vs Predicted Prices
3. Next 7 Days Price Forecast
4. Future Predictions with Confidence Interval

These visualizations help evaluate prediction accuracy and uncertainty.

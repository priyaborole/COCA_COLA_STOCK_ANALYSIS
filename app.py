# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import yfinance as yf
import pickle

# Load trained model
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Model file 'model.pkl' not found. Please train the model first.")
    st.stop()

# App Title
st.title("Coca-Cola Stock Price Prediction")

# Fetch latest Coca-Cola stock data
ticker = "KO"
data = yf.download(ticker, period="30d", interval="1h")

if data.empty:
    st.error("No stock data available. Try again later.")
    st.stop()

# Feature Engineering
data["MA_20"] = data["Close"].rolling(window=20).mean()
data["MA_50"] = data["Close"].rolling(window=50).mean()
data["Daily_Return"] = data["Close"].pct_change()
data["Volatility"] = data["Daily_Return"].rolling(window=20).std()

data.dropna(inplace=True)

if data.empty:
    st.error("Not enough data after processing. Try increasing the data period.")
    st.stop()

# Display latest stock data
st.subheader("Latest Coca-Cola Stock Data")
st.write(data.tail())

# Prepare latest data point for prediction
latest_features = data.iloc[-1][["Open", "High", "Low", "Volume", "MA_20", "MA_50", "Daily_Return", "Volatility"]].values.reshape(1, -1)

# Prediction Button
if st.button("Predict Next Closing Price"):
    predicted_price = model.predict(latest_features)
    st.success(f"Predicted Closing Price: **${predicted_price[0]:.2f}**")

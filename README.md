# Automated Stock Price Prediction using Web Scraping and LSTM

An end-to-end system that automates the process of collecting stock price data via web scraping, processes it, and generates accurate future price predictions using an LSTM (Long Short-Term Memory) neural network.

---

## Overview

This project demonstrates the use of **Python**, **Beautiful Soup**, and **TensorFlow/Keras** to build a real-time stock price prediction system. The system performs the following:

- Automatically **scrapes stock price data** daily from reliable financial websites.
- Preprocesses and feeds the data into an **LSTM model** trained to forecast future stock trends.
- Continuously learns and updates the model to improve predictive accuracy over time.
- Outputs **daily stock price predictions**, empowering users with data-driven investment insights.

---

## Why This Project is Different & Useful

Unlike static models or basic forecast tools, this system is:

### **Truly Automated**
- No need to manually download or update data — scraping is built-in and scheduled.

### **Adaptive Learning**
- The LSTM model **re-trains on new data** daily, improving accuracy and adjusting to market shifts.

### **Focused on Real-World Data**
- Instead of using idealized or API-fed historical data, it scrapes **real live market prices**, making it highly practical.

### **Practical for Traders and Learners**
- Traders can use it to spot trends.


## Features

- **Web Scraping**: Utilizes Beautiful Soup to extract real-time stock data.
- **LSTM Neural Network**: Predicts future prices based on past trends.
- **Automation**: Fully automated pipeline from data collection to prediction.
- **Forecasting**: Helps investors make informed decisions based on trend analysis.

---

## Tech Stack

- **Python** (3.8+)
- **Beautiful Soup** for web scraping
- **Pandas & NumPy** for data manipulation
- **Matplotlib/Seaborn** for visualization
- **TensorFlow / Keras** for LSTM modeling

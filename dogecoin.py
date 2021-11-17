from autots import AutoTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
sns.set()
plt.style.use('seaborn-whitegrid')

import streamlit as st
st.title("Future Price Prediction Model")
df = st.text_input("Let's Predict the Future Prices")

if df == "Dogecoin":
    data = pd.read_csv("C://Dogecoin.csv")
    print(data.head())
    data.dropna()
    model = AutoTS(forecast_length=10, frequency='infer', ensemble='simple', drop_data_older_than_periods=200)
    model = model.fit(data, date_col='Date', value_col='Close', id_col=None)
    prediction = model.predict()
    forecast = prediction.forecast
    st.write(forecast)
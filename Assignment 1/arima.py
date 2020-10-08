import numpy as np
import pandas as pd
# from statsmodels.tsa.arima.model import ARIMA
import pmdarima
from fbprophet import Prophet
import matplotlib.pyplot as plt
import csv

model_name = 'arima'
validation_size = 20

def cal_MAPE_loss():
    pass

def split_country(df):
    data = dict()
    country_list = df['countriesAndTerritories'].drop_duplicates().reset_index(drop=True)
    for country in country_list:
        data[country] = df[df['countriesAndTerritories'] == country][['date', 'cases']][::-1].reset_index(drop=True)
    return country_list, data

def write_csv(predictions):
    output = pd.DataFrame(columns=country_list.values)
    for country in country_list:
        output[country] = prediction[country]
    output.to_csv('106062361_HW1_5.csv')

def plot_validation(data, prediction):    
    x = np.arange(data.shape[0])
    plt.plot(x, data, c='blue')
    plt.plot(x[data.shape[0]-prediction.shape[0]:], prediction, c='green')
    plt.show()    

def predict(model_name, input, predict_size):
    if model_name == 'arima':
        data = input.cases.values
        model = pmdarima.auto_arima(data, seasonal=True, m=15, scoring='mae')
        forecast = model.predict(predict_size)  # predict N steps into the future

    elif model_name == 'prophet':
        data = input.rename(columns={"date": "ds", "cases": "y"})
        model = Prophet()
        model.fit(data)
        future = model.make_future_dataframe(periods=predict_size)
        forecast = model.predict(future).yhat.values

    return forecast

# Read data #
df = pd.read_csv("COVID_data.csv")
df['date'] = pd.to_datetime(df['dateRep'])
df = df[['date', 'cases', 'countriesAndTerritories']]
country_list, data = split_country(df)

# Training and Forecasting #
prediction = dict()
for country in country_list[40:50]:
    print(country)
    # prediction = predict(model_name, data[country][:-validation_size], validation_size)
    # prediction = arima(data[country], validation_size)
    # print(prediction)
    # plot_validation(data[country].cases.values, prediction)
    pred = predict(model_name, data[country], 7)
    prediction[country] = pred
    
write_csv(prediction)


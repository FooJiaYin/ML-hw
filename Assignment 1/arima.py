import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import csv

def split_country(df):
    data = dict()
    country_list = df['countriesAndTerritories'].drop_duplicates()
    print(country_list)
    for country in country_list:
        data[country] = df[df['countriesAndTerritories'] == country]['dateRep', 'cases']
    return country_list, data

df = pd.read_csv("COVID_data.csv")
df = df['dateRep', 'cases', 'countriesAndTerritories']
country_list, data = split_country(df)
print(data['Afghanistan'])

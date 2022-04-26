import plotly

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

SPLIT_TRAIN_RATIO = 0.75
def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename,parse_dates=['Date']).dropna().drop_duplicates()
    df = df[df["Day"].isin(range(1,31))]
    df = df[df["Year"].isin(range(1900,2022))]
    df = df[df["Month"].isin(range(1, 13))]
    df = df[df["Temp"] > -20]
    df = df[df["Temp"] < 60]
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    filered_data= load_data("City_Temperature.csv")
    # Question 2 - Exploring data for specific country
    israel_temp_data = filered_data[filered_data["Country"] == 'Israel']
    israel_temp_data["Year"] = israel_temp_data["Year"].astype(str)
    temp_dayOfYear_fig = px.scatter(israel_temp_data, y="Temp", x="DayOfYear", color="Year",title="Temperature per day of the year")
    plotly.offline.plot(temp_dayOfYear_fig)
    month_std_data =israel_temp_data.groupby('Month').agg({'Temp':'std'})
    month_std_fig = px.bar(month_std_data, x=list(range(1,13)), y='Temp',title= "Standard deviation of the daily temperatures per Month",
                           labels={"x":"Month","Temp":"Temperature standard deviation"})
    plotly.offline.plot(month_std_fig)
    # Question 3 - Exploring differences between countries
    country_month_data =filered_data.groupby(['Country','Month'],as_index=False).agg(mean = ('Temp','mean'),std = ('Temp','std'))
    monthly_temp_per_country_fig = px.line(country_month_data,x='Month',y='mean', error_y='std',color='Country',
                                           title="Average monthly temperature by Country",labels={"mean":" mean temperature"})
    plotly.offline.plot(monthly_temp_per_country_fig)
    # Question 4 - Fitting model for different values of `k`
    temp_vec = israel_temp_data['Temp']
    israel_temp_data = israel_temp_data.drop(columns =['Temp'])
    train_x,train_y,test_x,test_y = split_train_test(israel_temp_data,temp_vec,SPLIT_TRAIN_RATIO)
    loss_per_k = []
    for k in range(1,11):
        poly_fit =PolynomialFitting(k)
        poly_fit.fit(train_x['DayOfYear'].values, train_y.values)
        loss = round(poly_fit.loss(test_x['DayOfYear'].values, test_y.values), 2)
        print("The loss for k = "+str(k)+" is: "+str(loss) )
        loss_per_k.append(loss)
    test_error_k_fig = px.bar(x=list(range(1, 11)), y=loss_per_k, title="Loss value for each Degree",
                           labels={"x": "Degree", "y": "loss :mse"})
    plotly.offline.plot(test_error_k_fig)
    # Question 5 - Evaluating fitted model on different countries
    chosen_k = 5
    poly_fit = PolynomialFitting(chosen_k)
    poly_fit.fit(israel_temp_data['DayOfYear'],temp_vec)
    africa_temp_data = filered_data[filered_data["Country"] == 'South Africa']
    jordan_temp_data = filered_data[filered_data["Country"] == 'Jordan']
    netherlands_temp_data = filered_data[filered_data["Country"] == 'The Netherlands']
    loss_per_country =[]
    loss_per_country.append(poly_fit.loss(africa_temp_data['DayOfYear'].values,africa_temp_data['Temp']))
    loss_per_country.append(poly_fit.loss(jordan_temp_data['DayOfYear'].values,jordan_temp_data['Temp']))
    loss_per_country.append(poly_fit.loss(netherlands_temp_data['DayOfYear'].values,netherlands_temp_data['Temp']))
    loss_per_country_fig = px.bar(x=['South Africa','Jordan','The Netherlands'], y=loss_per_country,
                                  title="Model’s error over each of the countries ",
                           labels={ 'x':"Country",'y': "loss-mse" })
    plotly.offline.plot(loss_per_country_fig)



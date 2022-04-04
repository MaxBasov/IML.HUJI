import math
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
irrelevant_features = ["id", "lat", "long", "date"]
positive_features = ["sqft_living", "sqft_lot", "sqft_above", "yr_built", "sqft_living15", "sqft_lot15"]
not_negative_features = ["bathrooms", "floors", "sqft_basement", "yr_renovated"]
MIN_HOUSE_PRICE = 10000
SPLIT_TRAIN_RATIO = 0.75
def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    #reading the file, dropping all the rows that have missing values and removing duplicate rows.
    df = pd.read_csv(filename).dropna().drop_duplicates()
    # removing features that are irrelevant
    df = df.drop(columns= irrelevant_features)
    #remove not logical values of features
    df = df[df["price"] > MIN_HOUSE_PRICE]
    for feature in positive_features:
        df = df[df[feature]  > 0]
    for feature in not_negative_features:
        df = df[df[feature] >= 0]
    #remove feature values that not in the format
    df = df[df["waterfront"].isin([0, 1])]
    df = df[df["view"].isin(range(5))]
    df = df[df["condition"].isin(range(1,6))]
    df = df[df["grade"].isin(range(1,14))]
    #dealing with zipcode
    df["zipcode"] = df["zipcode"].astype(int)
    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])
    #dealing with year built and year renovation
    df["year_renovated_or_build"] = np.maximum(df["yr_built"],df["yr_renovated"])
    df = df.drop(columns = ["yr_built","yr_renovated"])
    df["decade"] = (df["year_renovated_or_build"] / 10).astype(int)
    df = df.drop(columns=["year_renovated_or_build"])
    df = pd.get_dummies(df, prefix='decade_', columns=['decade'])
    #removing extreme values
    df = df[df["bedrooms"] < 20]
    df = df[df["sqft_lot"] < 1250000]
    df = df[df["sqft_lot15"] < 500000]
    #extracting the price (the response vector)
    price_vec = df["price"]
    df = df.drop(columns=["price"])
    return df,price_vec




def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X = X.drop(columns="intercept")
    for feature in X:
        p_correlation = np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y))
        fig = px.scatter(pd.DataFrame({'x': X[feature], 'y': y}), x="x", y="y", trendline="ols",
                         title=f"Correlation Between {feature} Values and Response <br>Pearson Correlation {p_correlation}",
                         labels={"x": f"{feature} Values", "y": "Response Values"})
        fig.write_image(output_path+"/"+"pearson.correlation.%s.png" % feature)



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("kc_house_data.csv")
    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X,y)
    # Question 3 - Split samples into training- and testing sets.
    train_x,train_y,test_x,test_y = split_train_test(X,y,SPLIT_TRAIN_RATIO)
    linear_regression = LinearRegression()
    test_x.insert(0, 'intercept', 1, True)
    train_x["price"] = train_y
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    average = []
    percentage= []
    sd_lst = []
    for p in range(10,101):
        loss_lst = []
        for i in range(10):
            sampled_train_x = train_x.sample(frac = p/100)
            sampled_train_y = sampled_train_x["price"]
            sampled_train_x = sampled_train_x.drop(columns="price")
            linear_regression.fit(sampled_train_x.values,sampled_train_y.values)
            loss_lst.append(linear_regression.loss(test_x.values,test_y.values))
        average.append(np.mean(loss_lst))
        sd_lst.append(np.std(loss_lst))
        percentage.append(p)
    mean_df = np.array(average)
    sd_df = np.array(sd_lst)
    mse_by_percentage_fig = go.Figure([go.Scatter(x=percentage, y=average, mode="lines+markers",
    name = "mse by percentage",line =dict(dash= "dash"),marker=dict(color="blue")),
    go.Scatter(x = percentage,y=mean_df + 2*sd_df,fill= 'tonexty',mode ="lines",line=dict(color="lightgrey"),showlegend = False),
                       go.Scatter(x=percentage, y=mean_df - 2 * sd_df, fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                                  showlegend=False)])
    mse_by_percentage_fig.update_layout(title= "Loss Evaluation Over Increasing Percentage Of Training Set",
                                        xaxis = {"title": "Percentage of trained data"}, yaxis = {"title": "Mse of predicted test"})
    mse_by_percentage_fig.write_image("mse.over.training.percentage.png")




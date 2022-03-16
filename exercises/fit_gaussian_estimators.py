import math
import matplotlib.pyplot as plt
import plotly.offline
from scipy.stats import stats
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    variance = 1
    size =1000
    sigma = math.sqrt(variance)
    x = np.random.normal(mu,sigma,size=size)
    univar_gauss = UnivariateGaussian()
    univar_gauss.fit(x)
    count, bins, ignored = plt.hist(x, 30, density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    plt.show()
    print("("+ str(univar_gauss.mu_)+', ' + str(univar_gauss.var_)+ ')')




    # Question 2 - Empirically showing sample mean is consistent
    distance_between_exp = []
    ms = np.linspace(10, 1000, 100).astype(np.int)
    for m in ms:
        x = np.random.normal(mu, sigma, size=m)
        univar_gauss.fit(x)
        distance_between_exp.append(math.fabs(univar_gauss.mu_ - mu))


    fig =go.Figure([go.Scatter(x=ms, y=distance_between_exp, mode='markers+lines', name="abs(estimated - real)"),
               go.Scatter(x=ms, y=[0] * len(ms), mode='lines', name="Zero line")],
              layout=go.Layout(title=" Estimation of diffrence between estimated and real Expectation As Function Of Number Of Samples",
                               xaxis_title= "Number of samples", yaxis_title= "abs(estimated - real) Expectation",
                               height=300))
    plotly.offline.plot(fig)
    #raise NotImplementedError()

    # Question 3 - Plotting Empirical PDF of fitted model
    #raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

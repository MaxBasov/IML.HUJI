import math
import plotly.offline
import plotly.graph_objects as go
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"


def univariate_gaussian():

    # Question 1 - Draw samples and print fitted model


    mu = 10
    variance = 1
    size = 1000
    sigma = math.sqrt(variance)
    x = np.random.normal(mu, sigma, size)
    univar_gauss = UnivariateGaussian()
    univar_gauss.fit(x)
    print("Q1 : (" + str(univar_gauss.mu_) + ', ' + str(univar_gauss.var_) + ')')

    """
    #q3 in the quiz
    q3 = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
              -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    univar_gauss.fit(q3)
    print(univar_gauss.log_likelihood(1,1,q3))
    print(univar_gauss.log_likelihood(10,1,q3))
    """


    # Question 2 - Empirically showing sample mean is consistent
    distance_between_exp = []
    ms = np.linspace(10, 1000, 100).astype(np.int)
    for m in ms:
        expected = univar_gauss.fit(x[:m]).mu_
        distance_between_exp.append(abs(expected- mu))

    fig1 = go.Figure([go.Scatter(x=ms, y=distance_between_exp, mode='markers+lines', name="abs(estimated - real)"),
                      go.Scatter(x=ms, y=[0] * len(ms), mode='lines', name="Zero line")],
                     layout=go.Layout(
                         title="Q2: Estimation of difference between estimated and real Expectation As Function Of Number Of Samples",
                         xaxis_title="Number of samples", yaxis_title="abs(estimated - real) Expectation",
                         height=300))
    plotly.offline.plot(fig1)
    

    """
    pdf_arr = (univar_gauss.pdf(x))
    #Question 3 - Plotting Empirical PDF of fitted model
    fig2 = go.Figure([go.Scatter(x=x, y=pdf_arr, mode='markers', name="Empirical PDF of fitted model")],
                     layout=go.Layout(
                         title="Q3: Empirical PDF of fitted gaussian model",
                         xaxis_title="Ordered sample values", yaxis_title="Empirical PDF",
                         height=300))
    plotly.offline.plot(fig2)
    """
def multivariate_gaussian():

    # Question 4 - Draw samples and print fitted model

    mean = np.array([0,0,4,0])
    cov = np.array([[1,0.2,0,0.5],[0.2,2,0,0],[0,0,1,0],[0.5,0,0,1]])
    size = 1000
    x = np.random.multivariate_normal(mean,cov,size)
    multivar_gauss = MultivariateGaussian()
    multivar_gauss.fit(x)
    print("Q4:")
    print("Expectation:")
    print(multivar_gauss.mu_)
    print("Cov:")
    print(multivar_gauss.cov_)

    # Question 5 - Likelihood evaluation

    f1 = f3 = np.linspace(-10,10,200)
    likelihood_arr = np.zeros((200,200))
    for i in range(200):
        for j in range (200):
            cur_mu =np.array([f1[i], 0, f3[j], 0])
            likelihood_arr[i][j] = multivar_gauss.log_likelihood(cur_mu,cov,x)
    fig3 = px.imshow(likelihood_arr,x = f1,y=f3,title= "Q5: Log- likelihood of different expectations",height = 400,
                     labels = dict(x="value of f1",y = "value of f3",color="log- likelihood"))
    plotly.offline.plot(fig3)





    # Question 6 - Maximum likelihood

    i,j= np.unravel_index(np.argmax(likelihood_arr, axis=None), likelihood_arr.shape)
    print("Q6 : Maximum log-likelihood value achieved when f1 = " + str(round(f1[i],3)) + " and f3 = "+ str(round(f3[j],3)))


if __name__ == '__main__':
    np.random.seed(0)
    univariate_gaussian()
    multivariate_gaussian()

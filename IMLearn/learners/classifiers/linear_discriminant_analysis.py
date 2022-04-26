import math
from typing import NoReturn
import numpy as np
from numpy.linalg import det, inv
from IMLearn import BaseEstimator
from IMLearn.metrics import misclassification_error

class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ ,self.b_k= None, None, None, None, None,None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        unique_val = np.unique(y)
        self.classes_ = unique_val
        m, d , k =X.shape[0] , X.shape[1] , self.classes_.shape[0]
        self.cov_= np.zeros((d,d))
        self.mu_ =np.zeros((k,d))
        self.pi_ = np.zeros(k)
        for i,class_val in enumerate (self.classes_):
            self.pi_[i] = (y==class_val).mean()
            self.mu_[i] = X[y==class_val].mean(axis =0 )
            self.cov_ += (X[y== class_val] - self.mu_[i]).T @ (X[y== class_val] - self.mu_[i])
        self.cov_ /= (m-k)
        self.cov_inv = inv(self.cov_)
        self.b_k = - 0.5 * np.diag(self.mu_ @ self.cov_inv @ self.mu_.T) + np.log(self.pi_)
        self.fitted_= True



    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        class_index = np.argmax(X @ self.cov_inv @ self.mu_.T + self.b_k, axis=1)
        res = np.zeros(class_index.size)
        for index,class_val in enumerate (class_index):
            res[index] = self.classes_[class_val]
        return res

    def __one_sample_likelihood(self,x,class_sample_amount,sample_amount):
        x_minus_mu = np.zeros(shape=(self.classes_.size,x.size))
        for i in range (x_minus_mu.shape[0]):
            x_minus_mu[i] = x-self.mu_[i]
        likelihood_per_sample = - 0.5 * np.diag(x_minus_mu @ self.cov_inv @ x_minus_mu.T) \
                              + np.multiply(class_sample_amount,np.log(self.pi_))
        return likelihood_per_sample - (sample_amount*x.size)*math.log(2*math.pi)/2 - sample_amount/2 * math.log(det(self.cov_))

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        class_sample_amount = np.zeros(self.classes_.size)
        for i in range(class_sample_amount.size):
            class_sample_amount[i] = self.pi_[i] * X.shape[0]
        likelihood_matrix = np.zeros(shape=(X.shape[0],self.classes_.size))
        for i,sample in enumerate (X):
            likelihood_matrix[i] = self.__one_sample_likelihood(sample,class_sample_amount,X.shape[0])
        return  likelihood_matrix
    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y,self.predict(X))

if __name__ == '__main__':
    X = np.array([[1, -1], [2, 2], [5, 5], [6, 7],[19,23],[27,28]])
    y = np.array([1, 1, -1, -1,2,2])
    lda = LDA()
    lda.fit(X,y)
    print(lda.predict(X))
    print(lda.likelihood(X))
    print(lda.loss(X,y))
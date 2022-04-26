from typing import NoReturn

from IMLearn import BaseEstimator
import numpy as np
from IMLearn.learners import  MultivariateGaussian
from IMLearn.metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        unique_val = np.unique(y)
        self.classes_ = unique_val
        m, d, k = X.shape[0], X.shape[1], self.classes_.shape[0]
        self.cov_ = np.zeros((d, d))
        self.mu_ = np.zeros((k, d))
        self.pi_ = np.zeros(k)
        self.vars_ = np.zeros((k,d))
        for i, class_val in enumerate(self.classes_):
            self.pi_[i] = (y == class_val).mean()
            self.mu_[i] = X[y==class_val].mean(axis =0)
            self.vars_[i] = X[y==class_val].var(axis =0)

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
        mg = MultivariateGaussian()
        mg.fitted_ = True
        sample_amount = X.shape[0]
        ans = np.zeros(sample_amount)
        for sample_index ,sample  in enumerate(X):
            max_val = float('-inf')
            correct_cls = 0
            for class_index,cls in enumerate (self.classes_):
                likelihood_val = np.log(self.pi_[class_index])+\
                                 mg.log_likelihood(self.mu_[class_index],np.diag(self.vars_[class_index]),sample)
                if max_val < likelihood_val:
                    max_val = likelihood_val
                    correct_cls = cls
            ans[sample_index] = correct_cls
        return ans

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
        likelihood_arr = np.zeros((X.shape[0],self.classes_.size))
        mg = MultivariateGaussian()
        mg.fitted_ = True
        for sample_index, sample in enumerate(X):
            for class_index, cls in enumerate(self.classes_):
                likelihood_arr[sample_index][class_index] = np.log(self.pi_[class_index])+\
                                mg.log_likelihood(self.mu_[class_index],np.diag(self.vars_[class_index]),sample)
        return likelihood_arr
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
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        return misclassification_error(y, self.predict(X))
if __name__ == '__main__':
    X = np.array([0,1,2,3,4,5,6,7])
    y = np.array([2, 2, 3, 3])
    print(y.var())

   # lda = GaussianNaiveBayes()
    #lda.fit(X, y)

import numpy as np
from numpy import *
import scipy
import os

from numpy.linalg import solve


class LinearRegressionOLS:
    def __init__(self, fit_intercept: bool = True,
                 solver: str = "normal_equations",
                 singular: str = "error",
                 store_diagnostic: bool = False):

        self.__fit_intercept = fit_intercept
        self.__solver = solver
        self.__singular = singular
        self.__store_diagnostic = store_diagnostic

        # learned state
        self.beta_=None
        self.fitted_:bool=False


    #private methods
    def __compute_xTx(self,X:np.ndarray):

        self.__xTx=transpose(X)@X
        return self

    def __compute_xTy(self,X,y):
        self.__xTy = transpose(X) @ y
        return self

    def __add_intercept_columns(self,X):
        n=X.shape[0]
        ones=np.ones((n,1),dtype=X.dtype)
        X_aug=np.hstack((ones,X)) #augumented  matrix
        return X_aug

    #public API
    def fit(self,X:np.ndarray,y:np.ndarray):
        X_aug = self.__add_intercept_columns(X) if self.__fit_intercept else X
        self.__compute_xTx(X_aug)
        self.__compute_xTy(X_aug,y)
        self.beta_=solve(self.__xTx,self.__xTy)
        self.fitted_=True







if __name__ == '__main__':
    ols=LinearRegressionOLS()


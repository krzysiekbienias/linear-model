import numpy as np
import pytest
from linear_regression_algebra import LinearRegressionOLS

def test_fit_single_feature_with_intercept_exact():
    # y = 1 + 2x
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    model = LinearRegressionOLS(fit_intercept=True)
    model.fit(X, y)

    assert model.fitted_ is True
    np.testing.assert_allclose(model.beta_, np.array([1.0, 2.0]), rtol=1e-12, atol=1e-12)


def test_fit_single_feature_no_intercept_exact():
    # y = 2x (through origin)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 2.0, 4.0, 6.0])

    model = LinearRegressionOLS(fit_intercept=False)
    model.fit(X, y)

    assert model.fitted_ is True
    np.testing.assert_allclose(model.beta_, np.array([2.0]), rtol=1e-12, atol=1e-12)


def test_fit_two_features_with_intercept_exact():
    # y = 1 + 2x1 + 3x2
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    y = np.array([1.0, 3.0, 4.0, 6.0])

    model = LinearRegressionOLS(fit_intercept=True)
    model.fit(X, y)

    assert model.fitted_ is True
    np.testing.assert_allclose(model.beta_, np.array([1.0, 2.0, 3.0]), rtol=1e-12, atol=1e-12)


def test_refit_overwrites_beta():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y1 = np.array([1.0, 3.0, 5.0, 7.0])    # 1 + 2x
    y2 = np.array([1.0, 4.0, 7.0, 10.0])   # 1 + 3x

    model = LinearRegressionOLS(fit_intercept=True)

    model.fit(X, y1)
    np.testing.assert_allclose(model.beta_, np.array([1.0, 2.0]), rtol=1e-12, atol=1e-12)

    model.fit(X, y2)
    np.testing.assert_allclose(model.beta_, np.array([1.0, 3.0]), rtol=1e-12, atol=1e-12)

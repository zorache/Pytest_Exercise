import pytest 
import NewTrainer 
import numpy as np
import math

# This testing suite works with the class NewTrainer
# We assume that NewTrainer has assertion for catching incorrect data type 

# Generate random data with given parameters 
def test_data_random(x_dim, num, noise=1.0,linear=True):
    X = np.random.rand(num,x_dim)
    if linear:
        coefficients =  np.random.rand(x_dim,1)
        y = X*coefficients+ noise*np.random.normal(size=(num, 1))
    # Can add further options to generate non-linear data
    X = X.tolist()
    y = y.tolist()
    return X, y, coefficients, noise


# Test that the model training and prediction behavior is expected
def test_model_pred():
    x_dim  = 3 
    num = 5
    X,y,coefficient, noise = test_data_random(x_dim,num)
    c1 = NewTrainer()
    c1.train(X,y)
    sample = np.random.rand(x_dim,1)
    pred = c1.pred(sample)
    assert isinstance(pred, float)
    assert len(pred) == 1

# Test that the model raises correct errors for different data type errors
# We assume that the model class raises exception for assertion error
def test_invalid_input():
    x_dim  = 3 
    num = 5
    X,y,coefficient, noise = test_data_random(x_dim,num)
    c1 = NewTrainer()
    X1=X.copy()
    X1[0][0]=None
    try:
        c1.train(X1,y)
    except AssertionError as exc:
        assert "X has None value" in str(exc)
    y1=y.copy()
    y1[0][0]=None
    try:
        c1.train(X,y1)
    except AssertionError as exc:
        assert "Y has None value" in str(exc)
    X2=X.copy()
    X2[0][0]=int(1)
    try:
        c1.train(X2,y)
    except AssertionError as exc:
        assert "X is not a 2D list of float" in str(exc)
    y2=y.copy()
    y2[0]=int(1)
    try:
        c1.train(X,y2)
    except AssertionError as exc:
        assert "Y has None value" in str(exc)
    X3=1
    try:
        c1.train(X3,y)
    except AssertionError as exc:
        assert "X is not a 2D list" in str(exc)
    y3=1
    try:
        c1.train(X,y3)
    except AssertionError as exc:
        assert "X is not a 2D list" in str(exc)
    X4 = [[[[1]]]]
    try:
        c1.train(X4,y)
    except AssertionError as exc:
        assert "X is not a 2D list" in str(exc)
    y4 = [[[[1]]]]
    try:
        c1.train(X,y4)
    except AssertionError as exc:
        assert "Y is not a 1D list" in str(exc)

# Suppose we can access the model coefficients 
# Test that adding constant number to X or y should not change the coefficients 
def coeffient_invariant():
    x_dim  = 3 
    num = 5
    X,y,coefficient, noise = test_data_random(x_dim,num)
    c1 = NewTrainer()
    c1.train(X,y)
    coeff_1 = c1.getcoeff()
    b0_1 = c1.getb0()
    X_1 = X.copy()
    for i in range(num):
        for j in range(x_dim):
            X_1[i][j]=X_1[i][j]+3
    c2 = NewTrainer()
    c2.train(X_1,y)
    coeff_2 = c2.getcoeff()
    b0_2 = c2.getb0()
    assert b0_2!=b0_1
    assert math.dist(coeff_1,coeff_2)<0.01




    

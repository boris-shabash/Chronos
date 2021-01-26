# Copyright (c) Boris Shabash

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from chronos import Chronos
import chronos_utils

import pytest

import pandas as pd
import numpy as np


######################################################################
def test_basic_creation():
    '''
        Test that the class can be created using both MLE and MAP methods
    '''
    for distribution in chronos_utils.SUPPORTED_DISTRIBUTIONS:
        my_chronos = Chronos(distribution=distribution)
        my_chronos = Chronos(method="MAP", distribution=distribution)
        my_chronos = Chronos(method="MLE", distribution=distribution)

######################################################################
def test_bad_method():
    with pytest.raises(ValueError):
        my_chronos = Chronos(method="BOO")

    with pytest.raises(ValueError):
        my_chronos = Chronos(method="map")

    with pytest.raises(ValueError):
        my_chronos = Chronos(method="mlo")

    with pytest.raises(ValueError):
        my_chronos = Chronos(method="mle")

    for value in [1, 5.4, True, False, None]:
        with pytest.raises(TypeError):
            my_chronos = Chronos(method=value)
######################################################################
def test_bad_changepoint_number():
    my_chronos = Chronos(n_changepoints=5)

    with pytest.raises(ValueError):
        my_chronos = Chronos(n_changepoints=-5)

    for value in [5.3, "hello", True, None]:
        with pytest.raises(TypeError):
            my_chronos = Chronos(n_changepoints=value)
######################################################################
def test_bad_yearly_seasonality_order():
    my_chronos = Chronos(year_seasonality_order=5)

    with pytest.raises(ValueError):
        my_chronos = Chronos(year_seasonality_order=-5)

    for value in [5.3, "hello", True, None]:
        with pytest.raises(TypeError):
            my_chronos = Chronos(year_seasonality_order=value)
######################################################################
def test_bad_month_seasonality_order():
    my_chronos = Chronos(month_seasonality_order=5)

    with pytest.raises(ValueError):
        my_chronos = Chronos(month_seasonality_order=-5)

    for value in [5.3, "hello", True, None]:
        with pytest.raises(TypeError):
            my_chronos = Chronos(month_seasonality_order=value)
######################################################################
def test_bad_weekly_seasonality_order():
    my_chronos = Chronos(weekly_seasonality_order=5)

    with pytest.raises(ValueError):
        my_chronos = Chronos(weekly_seasonality_order=-5)

    for value in [5.3, "hello", True, None]:
        with pytest.raises(TypeError):
            my_chronos = Chronos(weekly_seasonality_order=value)
######################################################################
def test_bad_learning_rate():
    my_chronos = Chronos(learning_rate=0.3)

    with pytest.raises(ValueError):
        my_chronos = Chronos(learning_rate=-0.3)

    with pytest.raises(ValueError):
        my_chronos = Chronos(learning_rate=0.0)

    with pytest.raises(ValueError):
        my_chronos = Chronos(learning_rate=-0.0)

    for value in ["hello", True, False, None]:
        with pytest.raises(TypeError):
            my_chronos = Chronos(learning_rate=value)
######################################################################
def test_bad_changepoint_range():
    my_chronos = Chronos(changepoint_range=0.6)

    with pytest.raises(ValueError):
        my_chronos = Chronos(changepoint_range=-0.3)

    with pytest.raises(ValueError):
        my_chronos = Chronos(changepoint_range=1.4)

    for value in ["hello", True, False, None]:
        with pytest.raises(TypeError):
            my_chronos = Chronos(changepoint_range=value)
    
    with pytest.raises(ValueError):
        my_chronos = Chronos(changepoint_range=5)
######################################################################
def test_bad_changepoint_scale():
    my_chronos = Chronos(changepoint_prior_scale=0.6)

    with pytest.raises(ValueError):
        my_chronos = Chronos(changepoint_prior_scale=-0.3)

    with pytest.raises(ValueError):
        my_chronos = Chronos(changepoint_prior_scale=0.0)

    for value in ["hello", True, False, None]:
        with pytest.raises(TypeError):
            my_chronos = Chronos(changepoint_prior_scale=value)
######################################################################
def test_bad_distribution():
    my_chronos = Chronos(distribution="Normal")

    with pytest.raises(ValueError):
        my_chronos = Chronos(distribution="Bla")

    with pytest.raises(ValueError):
        my_chronos = Chronos(distribution="normal")

    with pytest.raises(ValueError):
        my_chronos = Chronos(distribution="student")

    for value in [3, 5.7, True, False, None]:
        with pytest.raises(TypeError):
            my_chronos = Chronos(distribution=value)
######################################################################
def test_bad_seasonality_mode():
    for mode in ["add", "mul"]:
        my_chronos = Chronos(seasonality_mode = mode)

    with pytest.raises(ValueError):
        my_chronos = Chronos(seasonality_mode="moo")

    with pytest.raises(ValueError):
        my_chronos = Chronos(seasonality_mode="additive")

    with pytest.raises(ValueError):
        my_chronos = Chronos(seasonality_mode="multiplicative")

    for value in [3, 5.7, True, False, None]:
        with pytest.raises(TypeError):
            my_chronos = Chronos(seasonality_mode=value)
######################################################################
def test_bad_max_iter():
    my_chronos = Chronos(max_iter=100)

    with pytest.raises(ValueError):
        my_chronos = Chronos(max_iter=-100)

    with pytest.raises(ValueError):
        my_chronos = Chronos(max_iter=0)

    for value in ["hello", 5.7, True, False, None]:
        with pytest.raises(TypeError):
            my_chronos = Chronos(max_iter=value)
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
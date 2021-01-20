# Copyright (c) Boris Shabash

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from chronos import Chronos
import chronos_utils

import pytest

import pandas as pd
import numpy as np


######################################################################
@pytest.fixture
def sample_data():
    '''
        Make some sample data for the tests
    '''

    '''
    dates = pd.date_range(start="2012-01-01", end="2012-01-08")
    measurements = list(range(dates.shape[0]))

    my_dataframe = pd.DataFrame({"ds": dates,
                                 "y": measurements})
    my_dataframe['ds'] = pd.to_datetime(my_dataframe['ds'])'''
    range_limit = 365*4
    x = np.array(range(range_limit))
    my_df = pd.DataFrame({"ds": pd.date_range(start="2016-01-01", periods=range_limit, freq='d'),
                          "y": 0.01 * x + np.sin(x/30)})

    return my_df
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
def test_incorrect_method_specification():
    '''
        Test that when an incorrect method is specified, 
        an error is raised
    '''
    with pytest.raises(ValueError):
        my_chronos = Chronos(method="BLA")

    with pytest.raises(ValueError):
        my_chronos = Chronos(method=3)

    with pytest.raises(ValueError):
        my_chronos = Chronos(method="map")

    with pytest.raises(ValueError):
        my_chronos = Chronos(method=4.3)

    with pytest.raises(ValueError):
        my_chronos = Chronos(method=None)

######################################################################
def test_predictions_with_additional_regressors(sample_data):
    '''
        TODO: update
    '''
    
    my_chronos = Chronos(n_changepoints=0, max_iter=100)
    my_chronos.add_regressors("reg1", "mul")
    my_chronos.add_regressors("reg2", "mul")
    
    # Check not including the regressor throws an error
    with pytest.raises(KeyError):
        my_chronos.fit(sample_data)
    
    # Now check with the regressor
    sample_data['reg1'] = [1] * sample_data.shape[0]    
    sample_data['reg2'] = [1] * sample_data.shape[0]    
    my_chronos.fit(sample_data)

    future_df = my_chronos.make_future_dataframe(include_history=False)

    # Check not including the regressor throws an error
    with pytest.raises(KeyError):
        predictions = my_chronos.predict(future_df)

    # Now check with the regressor
    future_df['reg1'] = [1] * future_df.shape[0]
    future_df['reg2'] = [1] * future_df.shape[0]    
    predictions = my_chronos.predict(future_df)

    predictions.drop('y', axis=1, inplace=True)    

    assert(predictions.isna().sum().sum() == 0)

######################################################################
def test_predictions_not_nan(sample_data):
    '''
        Make sure no nans are returned during the prediction process
    '''
    for method in ["MAP", "MLE"]:
        for distribution in chronos_utils.SUPPORTED_DISTRIBUTIONS:
            my_chronos = Chronos(n_changepoints=0, max_iter=100, distribution=distribution, method=method)
            my_chronos.fit(sample_data)

            future_df = my_chronos.make_future_dataframe(include_history=False)
            predictions = my_chronos.predict(future_df)

            predictions.drop('y', axis=1, inplace=True)    
            #print(predictions)

            assert(predictions.isna().sum().sum() == 0)



######################################################################
def test_prediction_no_seasonality(sample_data):
    '''
        Test that predictions without seasonality
        can be done, and that the trend is close to
        predictions since without seasonality
        only the distribution parameters should
        modify the predictions
    '''
    for distribution in chronos_utils.SUPPORTED_DISTRIBUTIONS:
        if (distribution not in [chronos_utils.Poisson_dist_code, chronos_utils.HalfNormal_dist_code]):
            # Poisson and half-normal produes very strange results here
            my_chronos = Chronos(n_changepoints=6,
                                max_iter=100, 
                                distribution=distribution,
                                year_seasonality_order=0, 
                                month_seasonality_order=0, 
                                weekly_seasonality_order=0)
            my_chronos.fit(sample_data)

            predictions = my_chronos.predict(sample_number=2000, period=30, frequency='D')

            assert(np.mean(np.abs(predictions['yhat'] - predictions['trend'])) <= 0.1)

######################################################################
def test_prediction_no_changepoints(sample_data):
    '''
        Test that predictions work without
        changepoints and that fitting can 
        still be done
    '''
    for distribution in chronos_utils.SUPPORTED_DISTRIBUTIONS:
        my_chronos = Chronos(n_changepoints=0, 
                             distribution=distribution,
                             max_iter=100)
        my_chronos.fit(sample_data)

        predictions = my_chronos.predict(sample_number=2000, period=30, frequency='D')

        assert(my_chronos._Chronos__n_changepoints == 0)

######################################################################

def test_prediction_too_small_for_default_changepoints(sample_data):
    '''
        Test that when the size of the data is too small, the
        number of changepoints gets adjusted
    '''
    for distribution in chronos_utils.SUPPORTED_DISTRIBUTIONS:
        my_chronos = Chronos(max_iter=100, n_changepoints=20, distribution=distribution)

        # This should raise a warning about the size of the data and changepoint number
        with pytest.warns(RuntimeWarning):
            my_chronos.fit(sample_data.iloc[:10])

        future_df = my_chronos.make_future_dataframe()
        predictions = my_chronos.predict(future_df)

        assert(my_chronos._Chronos__n_changepoints < sample_data.shape[0])

######################################################################

def test_prediction_with_easy_extra_regressors(sample_data):
    '''
        Test that when the size of the data is too small, the
        number of changepoints gets adjusted
    '''
    
    z = sample_data.index.values
    y = 0.01 * z + np.sin(z/30)
    sample_data['y'] = y
    
    # should be easy since the target is just dummy1 + dummy2
    dummy1 = 0.01 * z
    dummy2 = np.sin(z/30)
    sample_data['dummy1'] = dummy1
    sample_data['dummy2'] = dummy2

    for distribution in chronos_utils.SUPPORTED_DISTRIBUTIONS:

        # Student t distribution has issues with the long tails
        if (distribution != chronos_utils.StudentT_dist_code):
            my_chronos = Chronos(max_iter=200, distribution=distribution)

            # add dummies
            my_chronos.add_regressors("dummy1")
            my_chronos.add_regressors("dummy2")

            my_chronos.fit(sample_data)


            future_df = my_chronos.make_future_dataframe()

            # Add dummies to future df
            z = future_df.index.values

            y = 0.01 * z + np.sin(z/30)
            future_df['y'] = y
            
            dummy1 = 0.01 * z
            dummy2 = np.sin(z/30)
            future_df['dummy1'] = dummy1
            future_df['dummy2'] = dummy2


            # Make predictions
            predictions = my_chronos.predict(future_df)

            # The predictions should be almost the same as the target
            assert(np.mean(np.abs(predictions['y'] - predictions['yhat'])) < 0.1)


import sys
sys.path.append('../')
sys.path.append('./')

from chronos import Chronos
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def test_basic_creation():
    '''
        Test that the class can be created using both MLE and MAP methods
    '''
    my_chronos = Chronos()
    my_chronos = Chronos(method="MAP")
    my_chronos = Chronos(method="MLE")

######################################################################
@pytest.fixture
def sample_data():
    '''
        Make some sample data for the tests
    '''
    dates = pd.date_range(start="2012-01-01", end="2012-01-08")
    measurements = list(range(dates.shape[0]))

    my_dataframe = pd.DataFrame({"ds": dates,
                                 "y": measurements})
    my_dataframe['ds'] = pd.to_datetime(my_dataframe['ds'])

    return my_dataframe
######################################################################
def test_predictions_not_nan(sample_data):
    '''
        Make sure no nans are returned during the prediction process
    '''

    my_chronos = Chronos(n_changepoints=0, max_iter=100)
    my_chronos.fit(sample_data)

    future_df = my_chronos.make_future_dataframe(include_history=False)
    predictions = my_chronos.predict(future_df)

    predictions.drop('y', axis=1, inplace=True)    

    assert(predictions.isna().sum().sum() == 0)
######################################################################

def test_plotting_no_history(capsys, monkeypatch, sample_data):
    '''
        Test the plotting of data when no history is provided.
        This was an issue raised in issue #2.
    '''
    
    my_chronos = Chronos(n_changepoints=0, max_iter=100)
    my_chronos.fit(sample_data)

    future_df = my_chronos.make_future_dataframe(include_history=False)
    predictions = my_chronos.predict(future_df)

    
    fig = plt.figure(figsize=(15,5))
    gs = gridspec.GridSpec(1,1, figure=fig)
    gs_section = gs[0,0]

    my_chronos.plot_predictions(predictions, fig=fig, gs_section=gs_section)
    plt.savefig("test_prediction_no_history.png")

    std_error = capsys.readouterr().err
    assert(std_error == "")

######################################################################
def test_prediction_no_seasonality(sample_data):
    '''
        Test that predictions without seasonality
        can be done, and that the trend is close to
        predictions since without seasonality
        only the distribution parameters should
        modify the predictions
    '''
    my_chronos = Chronos(n_changepoints=0, 
                         max_iter=100, 
                         year_seasonality_order=0, 
                         month_seasonality_order=0, 
                         weekly_seasonality_order=0)
    my_chronos.fit(sample_data)

    predictions = my_chronos.predict(sample_number=2000, period=30, frequency='D')

    assert(np.mean(np.abs(predictions['yhat'] - predictions['trend'])) <= 0.1)

######################################################################

def test_prediction_too_small_for_default_changepoints(sample_data):
    '''
        Test that when the size of the data is too small, the
        number of changepoints gets adjusted
    '''
    my_chronos = Chronos(max_iter=100, n_changepoints=20)

    # This should raise a warning about the size of the data and changepoint number
    with pytest.warns(RuntimeWarning):
        my_chronos.fit(sample_data)

    future_df = my_chronos.make_future_dataframe()
    predictions = my_chronos.predict(future_df)

    assert(my_chronos.n_changepoints_ < sample_data.shape[0])

######################################################################
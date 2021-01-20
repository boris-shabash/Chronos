# Copyright (c) Boris Shabash

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from chronos import Chronos
import chronos_utils
import chronos_plotting

import pytest


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



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
def test_plotting(capsys, monkeypatch, sample_data):
    '''
        Test the plotting of data when no history is provided.
        This was an issue raised in issue #2.
    '''
    for distribution in chronos_utils.SUPPORTED_DISTRIBUTIONS:
        my_chronos = Chronos(n_changepoints=0, max_iter=100, distribution=distribution)
        my_chronos.fit(sample_data)

        future_df = my_chronos.make_future_dataframe(include_history=True)
        predictions = my_chronos.predict(future_df)


        with pytest.warns(UserWarning):
            chronos_plotting.plot_components(predictions, my_chronos, figure_name="test_prediction.png")

        std_error = capsys.readouterr().err
        assert(std_error == "")

    os.remove("test_prediction.png")

######################################################################
def test_plotting_no_history(capsys, monkeypatch, sample_data):
    '''
        Test the plotting of data when no history is provided.
        This was an issue raised in issue #2.
    '''
    for distribution in chronos_utils.SUPPORTED_DISTRIBUTIONS:
        my_chronos = Chronos(n_changepoints=0, max_iter=100, distribution=distribution)
        my_chronos.fit(sample_data)

        future_df = my_chronos.make_future_dataframe(include_history=False)
        predictions = my_chronos.predict(future_df)

        
        fig = plt.figure(figsize=(15,5))
        gs = gridspec.GridSpec(1,1, figure=fig)
        gs_section = gs[0,0]
        ax = fig.add_subplot(gs_section)

        #chronos_plotting.plot_predictions(predictions, my_chronos, fig=fig, gs_section=gs_section)
        chronos_plotting.plot_predictions(predictions, my_chronos, axs=ax)
        plt.savefig("test_prediction_no_history.png")

        std_error = capsys.readouterr().err
        assert(std_error == "")

    os.remove("test_prediction_no_history.png")
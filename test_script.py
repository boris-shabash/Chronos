import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from chronos import *

my_ts_data = pd.read_csv('data/prophetData.csv')
my_ts_data['ds'] = pd.to_datetime(my_ts_data['ds'])





my_chronos = Chronos(method="MAP", max_iter=1000, learning_rate=0.1)
my_chronos.fit(my_ts_data)
future_df = my_chronos.make_future_dataframe(period=365)

predictions = my_chronos.predict(future_df, sample_number=1000)

MAE = round(np.mean(np.abs(future_df['y'] - predictions['yhat'])), 2)



my_chronos.plot_components(predictions)

#my_chronos.plot_weekly_seasonality_plotly()
#my_chronos.plot_monthly_seasonality_plotly()
#my_chronos.plot_yearly_seasonality_plotly()
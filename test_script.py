import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from chronos import *

my_ts_data = pd.read_csv('data/prophetData.csv')
my_ts_data['ds'] = pd.to_datetime(my_ts_data['ds'])





my_chronos = Chronos(method="MAP", max_iter=100, learning_rate=10.0, n_changepoints=25)

my_chronos.fit(my_ts_data)
#future_df = my_chronos.make_future_dataframe(period=365)

predictions = my_chronos.predict(sample_number=1000, period=365)
print(predictions)
#assert(False)


MAE = round(np.mean(np.abs(predictions['y'] - predictions['yhat'])), 2)

plt.figure(figsize=(15,5))
plt.plot(predictions['ds'], predictions['yhat'], c="green")
plt.fill_between(predictions['ds'], predictions['yhat_upper'], predictions['yhat_lower'], color="green", alpha=0.3)
plt.scatter(predictions['ds'], predictions['y'], c="black")
plt.xlabel("Date", size=16)
plt.ylabel("LOG(page_views)", size=16)
plt.title(f"LOG(page view) for Peyton Manning from 2007-2016.\nMAE={MAE}", size=20)
plt.savefig("Time Series v1 t-distribution.png", dpi=96*4)
plt.show()#'''

#assert(False)

my_chronos.plot_components(predictions, figure_name="Complete plot.png")

#my_chronos.plot_weekly_seasonality_plotly()
#my_chronos.plot_monthly_seasonality_plotly()
#my_chronos.plot_yearly_seasonality_plotly()
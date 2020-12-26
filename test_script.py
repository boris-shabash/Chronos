import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from chronos import *

my_ts_data = pd.read_csv('data/prophetData.csv')
my_ts_data['ds'] = pd.to_datetime(my_ts_data['ds'])
print(my_ts_data.head())





my_chronos = Chronos(method="MAP", max_iter=1000, learning_rate=0.1)
my_chronos.fit(my_ts_data)
future_df = my_chronos.make_future_dataframe(period=365)
print("Future df")
print(future_df)

predictions = my_chronos.predict(future_df, sample_number=1000)

MAE = round(np.mean(np.abs(future_df['y'] - predictions['yhat'])), 2)

'''plt.figure(figsize=(15,5))
plt.plot(future_df['ds'], predictions['yhat'], c="green")
plt.fill_between(future_df['ds'], predictions['yhat_upper'], predictions['yhat_lower'], color="green", alpha=0.3)
plt.scatter(future_df['ds'], future_df['y'], c="black")
plt.xlabel("Date", size=16)
plt.ylabel("LOG(page_views)", size=16)
plt.title(f"LOG(page view) for Peyton Manning from 2007-2016.\n MAE={MAE}", size=20)
plt.savefig("Time Series v1 t-distribution.png", dpi=96*4)
plt.show()'''

my_chronos.plot_weekly_seasonality_plotly()
my_chronos.plot_monthly_seasonality_plotly()
my_chronos.plot_yearly_seasonality_plotly()
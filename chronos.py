'''
There should be a few specs for this method:

* Should have a scikit-learn like interface (fit, predict)
* Should have support for MLE, MAP, MCMC, and SVI
* Parameters should be easy to grab
* Should support arbitrary residual distributions
* Should support censored data

'''


import chronos_utils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import datetime

import plotly
import plotly.graph_objs as go

import torch
import pyro
import pyro.distributions as dist
from torch.optim import SGD, Rprop, Adam
from torch.distributions import constraints


# We will use Markov Chain Monte Carlo (MCMC) methods here, specifically the No U-Turn Sampler (NUTS)
from pyro.infer import MCMC, NUTS
from pyro.optim import ExponentialLR
from pyro.infer import SVI, Trace_ELBO, Predictive, JitTrace_ELBO
from pyro.infer.autoguide import AutoDelta



import warnings
import logging

pyro.enable_validation(False)
pyro.enable_validation(True)

torch.set_default_tensor_type(torch.FloatTensor)
class Chronos:
    
    
    def __init__(self, 
                 method="MLE", 
                 time_col = "ds",
                 target_col="y", 
                 n_changepoints = 6,
                 year_seasonality_order=10,
                 month_seasonality_order=5,
                 weekly_seasonality_order=3,
                 learning_rate=0.01,
                 max_iter=1000):
        
        self.method_ = method
        self.n_iter_ = max_iter
        self.lr_ = learning_rate
        self.n_changepoints_ = n_changepoints
        self.changepoint_range_  = 0.8
        
        
        self.year_seasonality_order_ = year_seasonality_order
        self.weekly_seasonality_order_ = weekly_seasonality_order
        self.month_seasonality_order_ = month_seasonality_order
        
        self.time_col_ = time_col
        self.target_col_ = target_col
        
        
        
    def transform_data_(self, data):
        '''
            A function to grab the raw data with a timestamp and a target column and
            add seasonality to the data using sine and cosine combinations. 
        '''
        # make a copy to avoid side effects of changing the original df
        # and convert datetime column into time in days. This helps convergence
        # BY A LOT
        internal_data = data.copy()
        internal_data['weekday'] = internal_data[self.time_col_].dt.dayofweek
        internal_data['monthday'] = internal_data[self.time_col_].dt.day - 1      # make days start at 0
        internal_data['yearday'] = internal_data[self.time_col_].dt.dayofyear - 1 # make days start at 0
        #display(internal_data)
        
        internal_data[self.time_col_] = internal_data[self.time_col_].values.astype(float)/(1e9*60*60*24)
        
        
        self.seasonality_cols_ = []
        # Yearly seasonality 
        for i in range(1, self.year_seasonality_order_+1):
            cycle_position = i*2*math.pi*internal_data['yearday']/366 # max value will be 365
                                                                      # since values will go from 0-365
            internal_data[f"yearly_sin_{i}"] = np.sin(cycle_position) 
            internal_data[f"yearly_cos_{i}"] = np.cos(cycle_position)
            self.seasonality_cols_.extend([f"yearly_sin_{i}", f"yearly_cos_{i}"])
        
        
        # Monthly seasonality
        for i in range(1, self.month_seasonality_order_+1):
            cycle_position = i*2*math.pi*internal_data['monthday']/31 # max value will be 30 since values
                                                                      # will go from 0 to 30
            internal_data[f"monthly_sin_{i}"] = np.sin(cycle_position) 
            internal_data[f"monthly_cos_{i}"] = np.cos(cycle_position)
            self.seasonality_cols_.extend([f"monthly_sin_{i}", f"monthly_cos_{i}"])
        
        # Weekly seasonality
        for i in range(1, self.weekly_seasonality_order_+1):
            cycle_position = i*2*math.pi*internal_data['weekday']/7 # max value will be 6 since values
                                                                    # will go from 0 to 6
            internal_data[f"weekly_sin_{i}"] = np.sin(cycle_position) 
            internal_data[f"weekly_cos_{i}"] = np.cos(cycle_position) 
            self.seasonality_cols_.extend([f"weekly_sin_{i}", f"weekly_cos_{i}"])
                                                                      
        
        # Add a constant so we can train a simple vector of coefficients
        #internal_data.insert(0, "CONST", 1.0)
        #display(internal_data)
        internal_data = internal_data.drop(['weekday', 'monthday', 'yearday'], axis=1)

        X_trend = torch.tensor(internal_data[self.time_col_].values, dtype=torch.float32)
        X_seasonality = torch.tensor(internal_data[self.seasonality_cols_].values , dtype=torch.float32)

        if (self.target_col_ in internal_data.columns):
            y = torch.tensor(internal_data[self.target_col_].values, dtype=torch.float32)
        else:
            y = None
        
        
        #assert(False)
        return X_trend, X_seasonality, y
        
        
    def find_changepoint_positions(self, X_trend):
        
        changepoint_range = int(self.changepoint_range_ * X_trend.shape[0])

        changepoints =  np.round(np.linspace(0, changepoint_range, self.n_changepoints_+1, dtype=np.float32))
        changepoints = changepoints[1:] # The first entry will always be 0, but we don't
                                        # want a changepoint right in the beginning

        return torch.tensor(changepoints, dtype=torch.float32)
        

    def make_A_matrix(self, point_number, changepoints):
        A = torch.zeros((point_number, self.n_changepoints_))

        for t, row in enumerate(A):
            for j, col in enumerate(row):
                if (changepoints[j] <= t):
                    A[t, j] = 1.0

        return A


    def fit(self, data):
        '''
            Fits the model to the data using the method specified.
        '''
        
        # Make a copy of the history
        self.history = data.copy()
        
        # Transform the data by adding seasonality
        #internal_data = self.transform_data_(data)
        X_trend, X_seasonality, y = self.transform_data_(data)
        
        
        self.changepoints = self.find_changepoint_positions(X_trend)
        A = self.make_A_matrix(X_trend.shape[0], self.changepoints)

        #print(self.changepoints)
        #print(A)
        
        
        #for t, row in enumerate(A):
        #    print(t, row)
        #assert(False)
            
        
        # Split the data into X features (all time related features), and 
        # the target y
        
        #y = torch.tensor(internal_data[self.target_col_].values, dtype=torch.float32)
        
        #internal_data = internal_data.drop(self.target_col_, axis=1)
        #X = torch.tensor(internal_data.values, dtype=torch.float32)
        #self.X_names = internal_data.columns.values
        
        
        
        if (self.method_ in ["MLE", "MAP"]):        # Point estimate methods
            if (self.method_ == "MLE"):
                print("Employing Maximum Likelihood Estimation")
                self.model = self.model_MLE_
                self.guide = self.guide_MLE_
                self.param_name_ = "betas"
                
            elif (self.method_ == "MAP"):
                print("Employing Maximum A Posteriori")
                self.model = self.model_MAP_
                self.guide = AutoDelta(self.model)
                self.param_name_ = "AutoDelta.betas"
                
            # This raises a trace warning so we turn that off. 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.train_point_estimate(self.model, 
                                          self.guide,
                                          X_trend,
                                          X_seasonality,
                                          A,
                                          y)
        elif (self.method == "MCMC"):
            print("Employing Markov Chain Monte Carlo")
        
            
            
    ##################################################################
    def train_MCMC(self, model, X, y, sample_num = 3000):
        pass
    
    ##################################################################
    def train_point_estimate(self, model, guide, X_trend, X_seasonality, A, y):
        
        pyro.clear_param_store()
        
        # Adam, SGD
        optimizer = Rprop
        scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 
                                              'optim_args': {'lr': self.lr_}, 
                                              'gamma': 0.5})
        
        my_loss = JitTrace_ELBO()
        my_loss = Trace_ELBO()
        self.svi_ = SVI(model, 
                        guide, 
                        scheduler, 
                        loss=my_loss)
        
        
        print_interval = max(self.n_iter_//10000, 10)
        
        for step in range(self.n_iter_):
            
            loss = round(self.svi_.step(X_trend, X_seasonality, A, y)/y.shape[0], 4)
            
            if (step % print_interval == 0):
                pct_done = round(100*(step+1)/self.n_iter_, 2)
                
                print(" "*100, end="\r")
                print(f"{pct_done}% - ELBO loss: {loss}", end="\r")
        
        pct_done = round(100*(step+1)/self.n_iter_, 2)
        print(" "*100, end="\r")
        print(f"{pct_done}% - ELBO loss: {loss}")
            
    ##################################################################
    
    def model_MLE_(self, X_trend, X_seasonality, y=None):     

        intercept = pyro.param("intercept", torch.tensor(0.0))
        trend_slope = pyro.param("trend_slope", torch.tensor(0.0))

        trend = trend_slope * X_trend + intercept

        betas = pyro.param(f"betas", torch.zeros(X_seasonality.size(1)))

                                        
        
        seasonality = X_seasonality.matmul(betas)
        

        mu = trend + seasonality

            
        
        sigma = pyro.param("sigma", 
                           torch.tensor(1.0), 
                           constraint = constraints.positive)
         
        
        with pyro.plate("data", X_trend.size(0)):
            pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
            
        return mu
        
        
    
    def guide_MLE_(self, X_trend, X_seasonality, y=None):
        pass
    
    
    ##################################################################
    def get_trend(self, X_trend):
        intercept = pyro.sample("intercept", dist.Normal(0.0, 10.0))
        trend_slope = pyro.sample("trend_slope", dist.Normal(0.0, 10.0))

        trend = trend_slope * X_trend + intercept

        return trend
    ##################################################################
    
    def model_MAP_(self, X_trend, X_seasonality, A, y=None):
        
        
        #print("X_trend", X_trend.shape)
        #print("A", A.shape)
        #trend = self.get_trend(X_trend)
        intercept_init = pyro.sample("intercept", dist.Normal(0.0, 10.0)) # m
        slope_init = pyro.sample("trend_slope", dist.Normal(0.0, 10.0)) # k

        deltas = pyro.sample("delta", dist.Laplace(torch.zeros(self.n_changepoints_), 
                                                   torch.full((self.n_changepoints_, ), 0.05)).to_event(1))

        #deltas = torch.where(torch.abs(deltas) < 0.01, torch.tensor(0.0), deltas)
        #print(deltas)

        #gammas = pyro.param("gamma", torch.zeros(self.n_changepoints_))     
        #gammas = torch.zeros_like(deltas)     
        gammas = -deltas * X_trend[self.changepoints.type(torch.int64)]
        '''for i in range(gammas.shape[0]):
            #one = (self.changepoints[i] - intercept_init - gammas[:i].sum())
            #two = 1 - (slope_init + deltas[:i].sum())/(slope_init + deltas[:i+1].sum())
            one = -self.changepoints[i]
            two = deltas[:i+1].sum() + slope_init
            gammas[i] = one * two
        #print(gammas)#'''

        #gammas = pyro.sample("gamma", dist.Delta(torch.ones(self.n_changepoints_)).to_event(1))


        slope = slope_init + torch.matmul(A, deltas)
        intercept = intercept_init + torch.matmul(A, gammas)
        

        #print("slope", slope.shape)
        '''print("#"*30)
        
        print("slope-init, k", slope_init)
        print('intercept, m', intercept_init)
        print("delta", deltas)
        print("gammas", gammas)
        print("changepoint", self.changepoints)
        print(self.changepoints[0]-2, self.changepoints[0]+1)
        print("t", self.changepoints[0]-2, self.changepoints[0]+2)
        #print("A", )#'''
        
        



        trend = (slope * X_trend) + intercept
        #trend = (slope_init + torch.matmul(A, deltas)) * X_trend + (torch.matmul(A, gammas) + intercept_init)
        #print('trend', trend[self.changepoints[0]-2:self.changepoints[0]+2])

        pyro.deterministic("trend", trend)



        betas = pyro.sample(f"betas", dist.Normal(torch.zeros(X_seasonality.size(1)), 
                                                  torch.full((X_seasonality.size(1),), 10.0)).to_event(1))

                                        
        
        seasonality = X_seasonality.matmul(betas)
        

        mu = trend + seasonality

        
        
        sigma = pyro.sample("sigma", dist.HalfCauchy(1.0))
        df = pyro.sample("df", dist.HalfCauchy(1.0))
         
        
        with pyro.plate("data", X_trend.size(0)):
            #pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
            pyro.sample("obs", dist.StudentT(df, mu, sigma), obs=y)
            
        return mu
    
    
    def model_MAP_gamma(self, X, y=None):
        
        betas = pyro.sample(f"betas", dist.Normal(torch.zeros(X.size(1)), 
                                                  torch.full((X.size(1),), 10.0)).to_event(1))
        
        linear_combo = X.matmul(betas)
        
        mu = linear_combo.clamp(min=torch.finfo(torch.float32).eps)
            
        
        
        rate = pyro.sample("rate", dist.HalfCauchy(1.0)).clamp(min=torch.finfo(torch.float32).eps)
        shape = mu * rate
         
        #print(shape, rate)
        
        
        with pyro.plate("data", X.size(0)):
            pyro.sample('obs', dist.Gamma(shape, rate), obs=y)
            
        return mu
        
    
    ##################################################################
    def predict(self, data, sample_number=100, ci_interval=0.95):
        #internal_data = self.transform_data_(data)
        #display(internal_data)

        X_trend, X_seasonality, y = self.transform_data_(data)
        
        
        #y = torch.tensor(internal_data[self.target_col_].values, dtype=torch.float32)
        
        #internal_data = internal_data.drop(self.target_col_, axis=1)
        #X = torch.tensor(internal_data.values, dtype=torch.float32)

        
        if (self.method_ in ["MAP", "MLE"]):
            # https://pyro.ai/examples/bayesian_regression.html#Model-Evaluation
            predictive = Predictive(model=self.model,
                                    guide=self.guide,
                                    num_samples=sample_number,
                                    return_sites=("obs", "trend")) 

            A = self.make_A_matrix(X_trend.shape[0], self.changepoints)
            samples = predictive(X_trend, X_seasonality, A)
            
            
            space_on_each_side = (1.0 - ci_interval)/2.0
            lower_ntile = int(len(samples['obs']) * space_on_each_side)
            upper_ntile = int(len(samples['obs']) * (1.0 - space_on_each_side))
            

            trend = samples['trend'].squeeze()[0, :]
            print(trend.shape)
            predictions = pd.DataFrame({"yhat": torch.mean(samples['obs'], 0).detach().numpy(),
                                        "yhat_lower": samples['obs'].kthvalue(lower_ntile, dim=0)[0].detach().numpy(),
                                        "yhat_upper": samples['obs'].kthvalue(upper_ntile, dim=0)[0].detach().numpy(),
                                        "trend": trend.detach().numpy()})
            
            predictions[self.target_col_] = y.detach().numpy()
            predictions[self.time_col_] = data[self.time_col_]
            return predictions[[self.time_col_, self.target_col_, 'yhat', 'yhat_upper', 'yhat_lower', 'trend']]
        else:
            raise NotImplementedError(f"Did not implement .predict for {self.method_}")

        
        
        
    ##################################################################
    def make_future_dataframe(self, period=30, frequency="D", include_history=True):
        '''
            Creates a future dataframe based on the specified parameters.
            
            Parameters:
            ------------
            [period] - int : The number of rows of future data to predict
            
        '''
        max_date_observed = self.history[self.time_col_].max()
        
        date_range = pd.date_range(start=str(max_date_observed), periods=period+1, freq=frequency)
        date_range = date_range[1:]
        
        future_df = pd.DataFrame({self.time_col_: date_range,
                                  self.target_col_: [np.nan] * date_range.shape[0]})
        
        past_df = self.history.copy()
        future_df = pd.concat([past_df, future_df], axis=0).reset_index(drop=True)

        return future_df
    
    
    ##################################################################
    def compute_seasonality(self, param_pairs, numeric_values, cycle_period):
        seasonality = np.zeros_like(numeric_values, dtype=np.float64)


        for i, pair in enumerate(param_pairs):
            cycle_order = i+1
            sin_coef = pair[0]
            cosin_coef = pair[1]
            
            cycle_pos = cycle_order * 2 * math.pi * numeric_values/cycle_period
            seasonality += (sin_coef * np.sin(cycle_pos)) + (cosin_coef * np.cos(cycle_pos))

        return seasonality
    ##################################################################
    def get_weekly_seasonality(self):
        if (self.method_ == "MAP"):
            return self.get_weekly_seasonality_point('AutoDelta.betas')
        elif (self.method_ == "MLE"):
            return self.get_weekly_seasonality_point('betas')
        else:
            raise NotImplementedError("Did not implement weekly seasonality for non MAP non MLE")
    ##################################################################
    def get_seasonal_params(self, param_name, seasonality_name):
        seasonal_params = []
        for i, param in enumerate(pyro.param(param_name)):
            if (seasonality_name in self.seasonality_cols_[i]):
                seasonal_params.append(param.item())
                
        seasonal_params = np.array(seasonal_params).reshape(-1, 2)

        return seasonal_params
    ##################################################################
    def get_seasonal_plotly_figure(self, seasonality_df):

        data_trace = go.Scatter(x=seasonality_df['X'], 
                                y=seasonality_df['Y'], 
                                mode="lines+markers",
                                marker=dict(color='Green', symbol="diamond"),
                                line=dict(width=3))

        trace_0 = go.Scatter(x=seasonality_df['X'],
                             y=[0]*seasonality_df.shape[0],
                             mode="lines",
                             line=dict(width=3, color="Black"))
        fig = go.Figure(
            data=[data_trace, trace_0])

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        return fig
    ##################################################################
    def get_weekly_seasonality_point(self, param_name):
        '''
            0 = Monday
        '''

        weekly_params = self.get_seasonal_params(param_name, "weekly")
        
        weekdays_numeric = np.arange(0, 7, 1)
        weekdays = chronos_utils.weekday_names_
        seasonality = self.compute_seasonality(weekly_params, weekdays_numeric, 7)
            
            
        weekly_seasonality = pd.DataFrame({"X": weekdays_numeric,
                                           "Label": weekdays,
                                           "Y": seasonality})
        
        return weekly_seasonality
    
    ##################################################################
    def plot_components(self, predictions, residuals=False, changepoint_threshold = 0.0):
        
        fig, axs = plt.subplots(5, 1, figsize=(15, 15))

        const = pyro.param('AutoDelta.intercept').detach()
        growth = pyro.param('AutoDelta.trend_slope').detach()
        #const, growth = pyro.param(self.param_name_).detach().numpy()[:2]
        trend_X = predictions[self.time_col_]
        trend_Y = predictions['trend'] #growth * trend_X.values.astype(float)/(1e9*60*60*24) + const
        predictions_start = predictions[predictions[self.target_col_].isna()][self.time_col_].min()
        axs[0].plot(trend_X, trend_Y, linewidth=3, c="green")
        for index, changepoint in enumerate(self.changepoints):
            
            changepoint_as_index = int(changepoint.item())
            changepoint_value = self.changepoints[index]
            changepoint_date_value = trend_X.iloc[changepoint_as_index]

            if (abs(changepoint_value) > changepoint_threshold):
                axs[0].axvline(changepoint_date_value, c="black", linestyle="dotted")
        axs[0].axvline(datetime.date(predictions_start.year, predictions_start.month, predictions_start.day), c="black", linestyle="--")
        axs[0].set_xlabel('Date', size=18)
        axs[0].set_ylabel('Growth', size=18)


        weekly_seasonality = self.get_weekly_seasonality()
        axs[1].plot(weekly_seasonality['X'], weekly_seasonality['Y'], linewidth=3, c="green")
        axs[1].axhline(0.0, c="black", linestyle="--")
        axs[1].set_xticks(weekly_seasonality['X'].values)
        axs[1].set_xticklabels(weekly_seasonality['Label'].values)
        axs[1].set_xlim(-0.1, 6.1)
        axs[1].set_xlabel('Weekday', size=18)
        axs[1].set_ylabel('Seasonality', size=18)

        monthly_seasonality = self.get_monthly_seasonality()
        axs[2].plot(monthly_seasonality['X'], monthly_seasonality['Y'], linewidth=3, c="green")
        axs[2].axhline(0.0, c="black", linestyle="--")
        axs[2].set_xticks(monthly_seasonality['X'].values[::9])
        axs[2].set_xticklabels(monthly_seasonality['Label'].values[::9])
        axs[2].set_xlim(-0.2, 30.2)
        axs[2].set_xlabel('Day of Month', size=18)
        axs[2].set_ylabel('Seasonality', size=18)

        yearly_seasonality = self.get_yearly_seasonality()
        axs[3].plot(yearly_seasonality['X'], yearly_seasonality['Y'], linewidth=3, c="green")
        axs[3].axhline(0.0, c="black", linestyle="--")
        axs[3].set_xlim(datetime.date(2019, 12, 31), datetime.date(2021, 1, 2))
        axs[3].set_xlabel('Day of Year', size=18)
        axs[3].set_ylabel('Seasonality', size=18)

        
        X = predictions[self.time_col_]
        Y = predictions[self.target_col_]
        Y_hat = predictions['yhat']
        Y_diff = Y - Y_hat
        
        axs[4].scatter(trend_X, Y_diff, c="green", s=4, alpha=0.2)
        axs[4].set_xlabel('Date', size=18)
        axs[4].set_ylabel('Residuals', size=18)

        plt.subplots_adjust(hspace=0.5)
        plt.savefig("Seasonality Plots.png", dpi=96*4)
        plt.show()
    ##################################################################
    def plot_weekly_seasonality(self):
        weekly_seasonality = self.get_weekly_seasonality()
        plt.figure(figsize=(15,5))
        plt.plot(weekly_seasonality['Weekday'], weekly_seasonality['Value'])
        plt.axhline(0.0, c="black")
        plt.xlim(0.0, 6.0)
        plt.show();
                
    ##################################################################
    def plot_weekly_seasonality_plotly(self):
        weekly_seasonality = self.get_weekly_seasonality()

        fig = self.get_seasonal_plotly_figure(weekly_seasonality)

        fig.update_layout(xaxis=dict(tickmode = 'array', 
                                     tickvals = weekly_seasonality['X'],
                                     ticktext = weekly_seasonality['Label']),
                          title=dict(text="Weekly Seasonality", x = 0.5))
        fig.show()

    ##################################################################
    def get_monthly_seasonality(self):
        if (self.method_ == "MAP"):
            return self.get_monthly_seasonality_point('AutoDelta.betas')
        elif (self.method_ == "MLE"):
            return self.get_monthly_seasonality_point('betas')
        else:
            raise NotImplementedError("Did not implement monthly seasonality for non MAP non MLE")
    ##################################################################
    def get_monthly_seasonality_point(self, param_name):

        monthly_params = self.get_seasonal_params(param_name, "monthly")
        
        monthdays_numeric = np.arange(0, 31, 1)
        monthday_names = chronos_utils.monthday_names_
        seasonality = self.compute_seasonality(monthly_params, monthdays_numeric, 31)
            
        monthly_seasonality = pd.DataFrame({"X": monthdays_numeric,
                                            "Label": monthday_names,
                                            "Y": seasonality})
        
        return monthly_seasonality
    ##################################################################
    def plot_monthly_seasonality(self):
        monthly_seasonality = self.get_monthly_seasonality()
        plt.figure(figsize=(15,5))
        plt.plot(monthly_seasonality['Monthday'], monthly_seasonality['Value'])
        plt.axhline(0.0, c="black")
        plt.xlim(1.0, 30.0)
        plt.show();
    ##################################################################
    def plot_monthly_seasonality_plotly(self):
        monthly_seasonality = self.get_monthly_seasonality()
        
        fig = self.get_seasonal_plotly_figure(monthly_seasonality)

        fig.update_layout(xaxis_range=[-0.2, 30.2])
        fig.update_layout(xaxis=dict(tickmode = 'array', 
                                     tickvals = monthly_seasonality['X'],
                                     ticktext = monthly_seasonality['Label']),
                          title=dict(text="Monthly Seasonality", x = 0.5))    
        fig.show()
    ##################################################################
    def get_yearly_seasonality(self):
        if (self.method_ == "MAP"):
            return self.get_yearly_seasonality_point('AutoDelta.betas')
        elif (self.method_ == "MLE"):
            return self.get_yearly_seasonality_point('betas')
        else:
            raise NotImplementedError("Did not implement yearly seasonality for non MAP non MLE")
    ##################################################################
    def get_yearly_seasonality_point(self, param_name):

        yearly_params = self.get_seasonal_params(param_name, "yearly")
        
        yeardays_numeric = np.arange(0, 366, 1)
        yearly_dates = pd.date_range(start="01-01-2020", periods=366) # Use a leap year to include Feb 29th
        seasonality = self.compute_seasonality(yearly_params, yeardays_numeric, 366)
            
            
        yearly_seasonality = pd.DataFrame({"X": yearly_dates,
                                           "Label": yearly_dates,
                                           "Y": seasonality})
        
        return yearly_seasonality
    
    ##################################################################
    def plot_yearly_seasonality(self):
        yearly_seasonality = self.get_yearly_seasonality()
        plt.figure(figsize=(15,5))
        plt.plot(yearly_seasonality['X'], yearly_seasonality['Y'])
        plt.axhline(0.0, c="black")
        plt.xlim(1.0, 366.0)
        plt.show()
        
    ##################################################################
    def plot_yearly_seasonality_plotly(self):
        yearly_seasonality = self.get_yearly_seasonality()

        
        fig = self.get_seasonal_plotly_figure(yearly_seasonality)
        
        fig.update_xaxes(dtick="M1", tickformat="%b", showgrid=False)
        # https://github.com/facebook/prophet/blob/20f590b7263b540eb5e7a116e03360066c58de4d/python/fbprophet/plot.py#L933        
        fig.update_layout(xaxis=go.layout.XAxis(tickformat = "%B %e", 
                                                type='date'), 
                          xaxis_range=["2019-12-30", "2021-01-02"],
                          title=dict(text="Yearly Seasonality", x = 0.5))
        
        fig.show()
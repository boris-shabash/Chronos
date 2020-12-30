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
from pyro.infer.autoguide.initialization import init_to_sample, init_to_feasible



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
                 n_changepoints = 20,
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


        self.ms_to_day_ratio_ = (1e9*60*60*24)
        
        
        
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
        
        internal_data[self.time_col_] = internal_data[self.time_col_].values.astype(float)/self.ms_to_day_ratio_
        
        
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
        
        
    def find_changepoint_positions(self, X_trend, changepoint_num, drop_first = True):
        
        #changepoint_range = int(self.changepoint_range_ * X_trend.max())

        min_value = X_trend.min().item()
        max_value_in_data = X_trend.max().item() #* data_prop

        max_distance = (max_value_in_data - min_value) * self.changepoint_range_
        max_value = min_value + max_distance

        if (drop_first):
            changepoints =  np.round(np.linspace(min_value, max_value, changepoint_num+1, dtype=np.float32))
            changepoints = changepoints[1:] # The first entry will always be 0, but we don't
                                            # want a changepoint right in the beginning
        else:
            changepoints =  np.round(np.linspace(min_value, max_value, changepoint_num, dtype=np.float32))

        changepoints = torch.tensor(changepoints, dtype=torch.float32)

        return changepoints
        

    def make_A_matrix(self, X_trend, changepoints):
        A = torch.zeros((X_trend.shape[0], self.n_changepoints_))

        #print(X_trend)

        for t, row in enumerate(A):
            for j, col in enumerate(row):
                if (changepoints[j] <= X_trend[t]):
                    A[t, j] = 1.0

        #print(A)
        #assert(False)
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

        self.changepoint_proportion = self.n_changepoints_/(X_trend.max() - X_trend.min())
        self.max_train_time = X_trend.max().item()
        
        
        
        self.changepoints = self.find_changepoint_positions(X_trend, self.n_changepoints_)
        A = self.make_A_matrix(X_trend, self.changepoints)

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
                self.param_prefix_ = ""
                
            elif (self.method_ == "MAP"):
                print("Employing Maximum A Posteriori")
                self.model = self.model_MAP_
                self.guide = AutoDelta(self.model, init_loc_fn=init_to_feasible)
                self.param_prefix_  = "AutoDelta."
                
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
                                              'gamma': 0.9})
        
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
    
    def model_MLE_(self, X_trend, X_seasonality, A, y=None):     
        
        intercept_init = pyro.param("intercept", torch.tensor(0.0))
        slope_init = pyro.param("trend_slope", torch.tensor(0.0))

        deltas = pyro.param("delta", torch.zeros(self.n_changepoints_))
        gammas = -deltas * self.changepoints

        slope = slope_init + torch.matmul(A, deltas)
        intercept = intercept_init + torch.matmul(A, gammas)



        trend = slope * X_trend + intercept

        pyro.deterministic('trend', trend)

        betas = pyro.param(f"betas", torch.zeros(X_seasonality.size(1)))

        seasonality = X_seasonality.matmul(betas)
        

        linear_combo = trend + seasonality
        mu = linear_combo

            
        
        sigma = pyro.param("sigma", 
                           torch.tensor(1.0), 
                           constraint = constraints.positive)

        df = pyro.param("df", 
                        torch.tensor(1.0), 
                        constraint = constraints.positive)
         
        
        with pyro.plate("data", X_trend.size(0)):
            #pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
            pyro.sample("obs", dist.StudentT(df, mu, sigma), obs=y)
            
        return mu
        
        
    
    def guide_MLE_(self, X_trend, X_seasonality, A, y=None):
        pass
    
    
    ##################################################################
    def get_trend(self, X_trend):
        intercept = pyro.sample("intercept", dist.Normal(0.0, 10.0))
        trend_slope = pyro.sample("trend_slope", dist.Normal(0.0, 10.0))

        trend = trend_slope * X_trend + intercept

        return trend
    ##################################################################
    
    def model_MAP_(self, X_trend, X_seasonality, A, y=None):
        
        
        intercept_init = pyro.sample("intercept", dist.Normal(0.0, 10.0))
        slope_init = pyro.sample("trend_slope", dist.Normal(0.0, 10.0))

        deltas = pyro.sample("delta", dist.Laplace(torch.zeros(self.n_changepoints_), 
                                                   torch.full((self.n_changepoints_, ), 0.05)).to_event(1))

        #deltas = torch.where(torch.abs(deltas) < 0.01, torch.tensor(0.0), deltas)
        
        gammas = -deltas * self.changepoints


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

        pyro.deterministic("trend", trend)

        betas = pyro.sample(f"betas", dist.Normal(torch.zeros(X_seasonality.size(1)), 
                                                  torch.full((X_seasonality.size(1),), 10.0)).to_event(1))

                                        
        seasonality = X_seasonality.matmul(betas)
        

        linear_combo = trend + seasonality
        mu = linear_combo
        #mu = torch.nn.Softplus()(linear_combo)+torch.finfo(torch.float32).eps
        #if (mu < 0.0):
        #    mu = -mu
        #rate = pyro.sample("rate", dist.HalfCauchy(1.0))#.clamp(min=torch.finfo(torch.float32).eps)
        #shape = (mu * rate)#.clamp(min=torch.finfo(torch.float32).eps)
        #print(shape, rate)
        
        sigma = pyro.sample("sigma", dist.HalfCauchy(1.0))
        df = pyro.sample("df", dist.HalfCauchy(1.0))
         
        
        with pyro.plate("data", X_trend.size(0)):
            #pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
            pyro.sample("obs", dist.StudentT(df, mu, sigma), obs=y)
            #pyro.sample('obs', dist.Gamma(shape, rate), obs=y)
            
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
        future_changepoint_number = self.changepoint_proportion * (X_trend.max() - self.max_train_time)
        future_changepoint_number = round(future_changepoint_number.item())
        print(future_changepoint_number)


        #assert(False)
        
        
        #y = torch.tensor(internal_data[self.target_col_].values, dtype=torch.float32)
        
        #internal_data = internal_data.drop(self.target_col_, axis=1)
        #X = torch.tensor(internal_data.values, dtype=torch.float32)

        
        if (self.method_ in ["MAP", "MLE"]):
            # https://pyro.ai/examples/bayesian_regression.html#Model-Evaluation
            predictive = Predictive(model=self.model,
                                    guide=self.guide,
                                    num_samples=sample_number,
                                    return_sites=("obs", "trend")) 

            A = self.make_A_matrix(X_trend, self.changepoints)
            samples = predictive(X_trend, X_seasonality, A)
            
            
            space_on_each_side = (1.0 - ci_interval)/2.0
            lower_ntile = int(len(samples['obs']) * space_on_each_side)
            upper_ntile = int(len(samples['obs']) * (1.0 - space_on_each_side))
            

            trend = samples['trend'].squeeze()[0, :]
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
            return self.get_weekly_seasonality_point(f'{self.param_prefix_}betas')
        elif (self.method_ == "MLE"):
            return self.get_weekly_seasonality_point(f'{self.param_prefix_}betas')
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
    def plot_components(self, predictions, residuals=False, changepoint_threshold = 0.0, figure_name = None):
        
        plot_num = 3
        if self.weekly_seasonality_order_ > 0:
            plot_num += 1
        if self.month_seasonality_order_ > 0:
            plot_num += 1
        if (self.year_seasonality_order_ > 0):
            plot_num += 1



        
        fig, axs = plt.subplots(plot_num, 1, figsize=(15, 15))

        current_axs = 0
        axs[0].plot(predictions[self.time_col_], predictions['yhat'], c="green", label="predictions")
        axs[0].fill_between(predictions[self.time_col_], predictions['yhat_upper'], predictions['yhat_lower'], color="green", alpha=0.3)
        axs[0].scatter(predictions[self.time_col_], predictions['y'], c="black", label="observed")
        axs[0].set_xlabel("Date", size=16)
        axs[0].set_ylabel("Values", size=16)
        current_axs += 1


        const = pyro.param(f'{self.param_prefix_}intercept').detach()
        growth = pyro.param(f'{self.param_prefix_}trend_slope').detach()
        #const, growth = pyro.param(self.param_name_).detach().numpy()[:2]
        trend_X = predictions[self.time_col_]
        trend_Y = predictions['trend']
        predictions_start = predictions[predictions[self.target_col_].isna()][self.time_col_].min()
        axs[current_axs].plot(trend_X, trend_Y, linewidth=3, c="green")
        for index, changepoint in enumerate(self.changepoints):
            
            changepoint_value = int(self.changepoints[index].item() * self.ms_to_day_ratio_/1e9)
            changepoint_date_value = datetime.datetime.fromtimestamp(changepoint_value)

            if (abs(changepoint_value) >= changepoint_threshold):
                axs[current_axs].axvline(changepoint_date_value, c="black", linestyle="dotted")#'''
        axs[current_axs].axvline(datetime.date(predictions_start.year, predictions_start.month, predictions_start.day), c="black", linestyle="--")
        axs[current_axs].set_xlabel('Date', size=18)
        axs[current_axs].set_ylabel('Growth', size=18)
        current_axs += 1

        if (self.weekly_seasonality_order_ > 0):
            weekly_seasonality = self.get_weekly_seasonality()
            axs[current_axs].plot(weekly_seasonality['X'], weekly_seasonality['Y'], linewidth=3, c="green")
            axs[current_axs].axhline(0.0, c="black", linestyle="--")
            axs[current_axs].set_xticks(weekly_seasonality['X'].values)
            axs[current_axs].set_xticklabels(weekly_seasonality['Label'].values)
            axs[current_axs].set_xlim(-0.1, 6.1)
            axs[current_axs].set_xlabel('Weekday', size=18)
            axs[current_axs].set_ylabel('Seasonality', size=18)
            current_axs += 1

        if (self.month_seasonality_order_ > 0):
            monthly_seasonality = self.get_monthly_seasonality()
            axs[current_axs].plot(monthly_seasonality['X'], monthly_seasonality['Y'], linewidth=3, c="green")
            axs[current_axs].axhline(0.0, c="black", linestyle="--")
            axs[current_axs].set_xticks(monthly_seasonality['X'].values[::9])
            axs[current_axs].set_xticklabels(monthly_seasonality['Label'].values[::9])
            axs[current_axs].set_xlim(-0.2, 30.2)
            axs[current_axs].set_xlabel('Day of Month', size=18)
            axs[current_axs].set_ylabel('Seasonality', size=18)
            current_axs += 1

        if (self.year_seasonality_order_ > 0):
            yearly_seasonality = self.get_yearly_seasonality()
            axs[current_axs].plot(yearly_seasonality['X'], yearly_seasonality['Y'], linewidth=3, c="green")
            axs[current_axs].axhline(0.0, c="black", linestyle="--")
            axs[current_axs].set_xlim(datetime.date(2019, 12, 31), datetime.date(2021, 1, 2))
            axs[current_axs].set_xlabel('Day of Year', size=18)
            axs[current_axs].set_ylabel('Seasonality', size=18)
            current_axs += 1

        
        X = predictions[self.time_col_]
        Y = predictions[self.target_col_]
        Y_hat = predictions['yhat']
        Y_diff = Y - Y_hat
        
        axs[current_axs].scatter(trend_X, Y_diff, c="green", s=4, alpha=0.2)
        axs[current_axs].set_xlabel('Date', size=18)
        axs[current_axs].set_ylabel('Residuals', size=18)

        plt.subplots_adjust(hspace=0.5)
        if (figure_name is not None):
            plt.savefig(figure_name, dpi=96*4)
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
            return self.get_monthly_seasonality_point(f'{self.param_prefix_}betas')
        elif (self.method_ == "MLE"):
            return self.get_monthly_seasonality_point(f'{self.param_prefix_}betas')
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
            return self.get_yearly_seasonality_point(f'{self.param_prefix_}betas')
        elif (self.method_ == "MLE"):
            return self.get_yearly_seasonality_point(f'{self.param_prefix_}betas')
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
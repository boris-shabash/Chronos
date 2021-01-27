# Copyright (c) Boris Shabash

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import chronos_utils
import pandas as pd
import numpy as np

import torch
from torch.optim import Rprop
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.optim import ExponentialLR
from pyro.infer import SVI, Trace_ELBO, Predictive, JitTrace_ELBO
from pyro.infer.autoguide import AutoDelta, AutoNormal
from pyro.infer.autoguide.initialization import init_to_feasible



import warnings


pyro.enable_validation(True)

class Chronos:
    '''


        Parameters:
        ------------
        method="MAP" -              [str] The estimation method used. Currently only 
                                    supports one of "MAP" (Maximum A Posteriori), or "MLE"
                                    (Maximum Likelihood Estimation). If "MLE" is chosen, 
                                    'changepoint_prior_scale' is ignored. 

                                    Default is "MAP"

        n_changepoints -            [int] The number of changepoints the model considers
                                    when fitting to the data, must be 0 or larger. 
                                    Changepoints are points in time when the slope of the
                                    trend can change. More changepoints will allow for a 
                                    better fit, but will also increase uncertainty when 
                                    predicting into the future. 

                                    Default is 20

        year_seasonality_order -    [int] The fourier order used to predict yearly 
                                    seasonality. Must be 0 or larger. Larger values will 
                                    allow for a better fit for yearly seasonality but 
                                    increase the odds of overfitting, as well as fitting
                                    time. Setting this value to 0 implies there is no 
                                    yearly seasonality.

                                    Default is 10

        month_seasonality_order -   [int] The fourier order used to predict monthly
                                    seasonality. Must be  0 or larger. Larger values
                                    will allow for a better fit for monthly seasonality
                                    but increase the odds of overfitting, as well as 
                                    fitting time. Setting this value to 0 implies there
                                    is no monthly seasonality.

                                    Default is 5

        weekly_seasonality_order -  [int] The fourier order used to predict weekly
                                    seasonality. Must be 0 or larger. Larger values will
                                    allow for a better fit for weekly seasonality but
                                    increase the odds of overfitting, as well as fitting 
                                    time. Setting this value to 0 implies there is no 
                                    weekly seasonality.

                                    Default is 3

        learning_rate -             [float] The learning rate used for optimization when
                                    the optimization method is "MAP" or "MLE". Most be 
                                    larger than 0. Larger values make the algorithm learn
                                    faster but might produce worse solutions. Smaller
                                    values allow for better convergence but will require
                                    more iterations to be specified at [max_iter].

                                    Default is 0.01

        changepoint_range -         [float] The range of the historical data to apply
                                    changepoints to. Must be between 0.0 - 1.0. 0.8 
                                    would mean only considering changepoints for the 
                                    first 80% of historical data. Larger values would 
                                    provide better fit, but would also be more 
                                    sensitive to recent changes which may or may not 
                                    indicate trends.

                                    Default is 0.8

        changepoint_prior_scale -   [float] the scale for the changepoint value
                                    prior distribution. Must be larger than 0.0. The 
                                    changepoints are assumed to come from a Laplace 
                                    distribution which is centered at 0 and is specified
                                    by the scale value. Larger values for the scale 
                                    allow for more changepoints with larger changes, and
                                    may increase fit, but will also increase the 
                                    uncertainty in future predictions.
                                    
                                    Default is 0.05

        distribution -              [string] The distribution which describes the 
                                    behaviour of the data (mainly of the residuals) 
                                    at each timestamp. Supported distributions are:

                                    "Normal"    - The normal (Guassian) distribution
                                    "StudentT"  - Student's t-distribution. Unlike the 
                                                  normal distribution it has fatter 
                                                  tails, so can be more resistent to 
                                                  outliers
                                    "Gamma"     - The gamma distribution. Only has 
                                                  support for positive values.

                                    Default is "Normal"

        seasonality_mode -          [string] Whether seasonality is an additive quantity
                                    ("add") or a multiplicative one ("mul"). If 
                                    seasonality is specified as "add", the seasonal 
                                    components are added to the trend. For example, if 
                                    every Saturday you see 5 additiona clients, that 
                                    seasonal component is additive. If the seasonality 
                                    is specified as "mul", the seasonal components are 
                                    multiplied by the trend. For example, if every 
                                    Saturday you see a 50% increase in clients, that 
                                    seasonal component is multiplicative as it depends 
                                    on the number of clients already expected.

                                    Default is "add"

        max_iter -                  [int] The maximum number of iterations the algorithm 
                                    is allowed to run for. Must be larger than 0. Chronos 
                                    employes an optimization based approach for the "MAP" 
                                    and "MLE" method, and this parameter determines how 
                                    long the optimization algorithm can run for. Larger 
                                    values will increase run-time, but will lead to better
                                    results. Smaller learning_rate values require larger 
                                    max_iter values.

                                    Default is 1000.



        Example Usage:
        ----------------
        
        # Create sample data
        >>> import chronos_plotting
        >>> from chronos import Chronos

        >>> x = np.array(range(365*4))
        >>> my_df = pd.DataFrame({"ds": pd.date_range(start="2016-01-01", 
                                                      periods=365*4, 
                                                      freq='d'),
                                  "y": 0.01 * x + np.sin(x/30)})
        >>> print(my_df.head())
                  ds         y
        0 2016-01-01  0.000000
        1 2016-01-02  0.043327
        2 2016-01-03  0.086617
        3 2016-01-04  0.129833
        4 2016-01-05  0.172939
        
        >>> my_chronos = Chronos()
        >>> my_chronos.fit(my_df)
        Employing Maximum A Posteriori
        100.0% - ELBO loss: -2.4531 | Mean Absolute Error: 0.2296   

        >>> predictions = my_chronos.predict(period=31)
        Prediction no: 1000
        >>> chronos_plotting.plot_components(predictions, my_chronos)
        ... plot appears
    '''

    
    
    def __init__(self, 
                 method="MAP", 
                 n_changepoints = 20,
                 year_seasonality_order=10,
                 month_seasonality_order=5,
                 weekly_seasonality_order=3,
                 learning_rate=0.01,
                 changepoint_range = 0.8,
                 changepoint_prior_scale = 0.05,
                 distribution = "Normal",
                 seasonality_mode = "add",
                 max_iter=1000):

        '''
            The initialization function. See class docstring for an in-depth explanation
            of all parameters
        '''

        
        self.__method = self.__check_is_string_supported(method, 
                                                         "method", 
                                                         chronos_utils.SUPPORTED_METHODS)
        self.__seasonality_mode = self.__check_is_string_supported(seasonality_mode, 
                                                                   "seasonality_mode", 
                                                                   ["add", "mul"])
        self.__y_likelihood_distribution = self.__check_is_string_supported(distribution, 
                                                                            "distribution", 
                                                                            chronos_utils.SUPPORTED_DISTRIBUTIONS)
        

        self.__number_of_changepoints = self.__check_is_supported_integer(n_changepoints, 
                                                                          "n_changepoints", 
                                                                          positive=False)        
        self.__year_seasonality_fourier_order = self.__check_is_supported_integer(year_seasonality_order, 
                                                                                  "year_seasonality_order", 
                                                                                  positive=False)
        self.__weekly_seasonality_fourier_order = self.__check_is_supported_integer(weekly_seasonality_order, 
                                                                                    "weekly_seasonality_order", 
                                                                                    positive=False)
        self.__month_seasonality_fourier_order = self.__check_is_supported_integer(month_seasonality_order, 
                                                                                   "month_seasonality_order", 
                                                                                   False)
        self.__number_of_iterations  = self.__check_is_supported_integer(max_iter, 
                                                                        "max_iter", 
                                                                        positive=True)


        self.__learning_rate = self.__check_is_supported_float(learning_rate, 
                                                               "learning_rate", 
                                                               positive=True)
        self.__changepoint_prior_scale = self.__check_is_supported_float(changepoint_prior_scale, 
                                                                         "changepoint_prior_scale", 
                                                                         positive=True)
        self.__proportion_of_data_subject_to_changepoints = self.__check_is_supported_float(changepoint_range, 
                                                                                            "changepoint_range", 
                                                                                            positive=False)

        if (self.__proportion_of_data_subject_to_changepoints > 1.0):
            raise ValueError("changepoint_range must be less than, or equal to, 1.0")


        
        self.__y_max = None
        self.__history_min_time_seconds = None
        self.__history_max_time_seconds = None
        self.__prediction_verbose = False

        
        self.__multiplicative_seasonalities = []
        self.__multiplicative_additional_regressors = []

        self.__additive_seasonalities = []
        self.__additive_additional_regressors = []


        self.__reserved_names = [f"trend{suffix}" for suffix in ["", "_upper", "_lower"]]
        self.__reserved_names.extend([f"yhat{suffix}" for suffix in ["", "_upper", "_lower"]])


        self.__add_default_seasonalities()

        # Some distributions will require extra work to ensure the 
        # parameters fed in are positive
        if (self.__y_likelihood_distribution in chronos_utils.POSITIVE_DISTRIBUTIONS):
            self.__make_likelihood_mean_positive = True
        else:
            self.__make_likelihood_mean_positive = False
        
    ########################################################################################################################
    def __check_is_string_supported(self, variable, variable_name, options):
        '''
            Checks if the string supplied is one of the options, and throws
            a value error if it is not

            Parameters:
            ------------
            variable -      [str] A string which is the value

            variable_name - [str] A string which is the variable name. Is included
                            for output purposes

            options -       [list] A list of strings which contains the available
                            options for the given variable

            Returns:
            ------------
            variable -      [str] The inputted variable, unmodified, if no errors
                            occured
        '''
        if (not isinstance(variable, str)):
            raise TypeError(f"A value of {variable} for {variable_name} is not supported. Supported values are: {options}")
        if (variable not in options):
            raise ValueError(f"A value of {variable} for {variable_name} is not supported. Supported values are: {options}")
        else:
            return variable
    ########################################################################################################################
    def __check_is_supported_float(self, variable, variable_name, positive=True):
        '''
            Checks if the float provided is supported, and throws an error if it
            is not

            Parameters:
            ------------
            variable -      [float] An integer which is the value

            variable_name - [str] A string which is the variable name. Is included
                            for output purposes

            positive -      [bool] A flag denoting if only positive values are
                            supported or all non-negatives are

            Returns:
            ------------
            variable -      [float] The inputted variable, unmodified, if no errors
                            occured
        '''

        if (positive == True):
            error_message = f"{variable_name} must be a positive float"
        else:
            error_message = f"{variable_name} must be a non-negative float"

        if (not isinstance(variable, float)):
            if (not isinstance(variable, int)):
                raise TypeError(error_message)
            elif (isinstance(variable, bool)):
                raise TypeError(error_message)
        elif (positive == True):
            if (variable <= 0.0):
                raise ValueError(error_message)
        elif (positive == False):
            if (variable < 0.0):
                raise ValueError(error_message)
        
        return variable
    ########################################################################################################################
    def __check_is_supported_integer(self, variable, variable_name, positive=True):
        '''
            Checks if the integer provided is supported, and throws an error if it
            is not

            Parameters:
            ------------
            variable -      [int] An integer which is the value

            variable_name - [str] A string which is the variable name. Is included
                            for output purposes

            positive -      [bool] A flag denoting if only positive values are
                            supported or all non-negatives are

            Returns:
            ------------
            variable -      [int] The inputted variable, unmodified, if no errors
                            occured
        '''
        if (positive == True):
            error_message = f"{variable_name} must be a positive integer"
        else:
            error_message = f"{variable_name} must be a non-negative integer"

        if (not isinstance(variable, int)):
            raise TypeError(error_message)
        elif (isinstance(variable, bool)):
            raise TypeError(error_message)
        elif (positive == True):
            if (variable <= 0):
                raise ValueError(error_message)
        elif (positive == False):
            if (variable < 0):
                raise ValueError(error_message)
        
        return variable
    ########################################################################################################################
    def __add_default_seasonalities(self):
        '''
            A function to add the built-in default seasonalities

            Parameters:
            ------------
            None

            Returns:
            ------------
            None
        '''
        self.add_seasonality("yearly", 
                             self.__year_seasonality_fourier_order, 
                             self.__yearly_cycle_extraction_function,
                             self.__seasonality_mode)

        self.add_seasonality("monthly", 
                             self.__month_seasonality_fourier_order, 
                             self.__monthly_cycle_extraction_function,
                             self.__seasonality_mode)

        self.add_seasonality("weekly", 
                             self.__weekly_seasonality_fourier_order, 
                             self.__weekly_cycle_extraction_function,
                             self.__seasonality_mode)
    ########################################################################################################################
    def __is_regressor_name_available(self, regressor_name):
        '''
            A function which checks to see if the regressor name is available.

            Parameters:
            ------------
            regressor_name -    [str] A string which is the regressor name 

            Returns:
            ------------
            is_available -      [bool] True if the name is available,
                                False otherwise
        '''
        if (regressor_name in self.__additive_additional_regressors) or \
            (regressor_name in self.__multiplicative_additional_regressors) or \
            (regressor_name in self.__reserved_names):
            return False
        else:
            return True
    ######################################################################################################################## 
    def add_regressors(self, regressor_name, regressor_method="add"):
        '''
            A function which adds the name of a regressor that should
            be considered when fitting the model. The regressor can be
            either additive or multiplicative

            Parameters:
            ------------
            regressor_name -    [str] A string denoting the name of
                                the regressor. Cannot be one of the
                                names 'yhat', 'yhat_upper', 'yhat_lower',
                                'trend', 'trend_upper', or 'trend_lower'.
                                Also cannot be the name of a previously
                                added regressor, i.e. two regressors 
                                cannot have the same name

            regressor_method -  [str] either additive "add" or multiplicative "mul".
                                Specifies the mode of regressor incorporation

            
            Returns:
            ------------
            None
        '''

        # First check the name is available
        if (not self.__is_regressor_name_available(regressor_name)):
            raise ValueError(f"Name {regressor_name} is already in use")


        # Now add it to the appropriate bucket
        if (regressor_method == "add"):
            self.__additive_additional_regressors.append(regressor_name)
        elif (regressor_method == "mul"):
            self.__multiplicative_additional_regressors.append(regressor_name)
        else:
            raise ValueError(f"method {regressor_method} is not supported, supported methods are 'add' or 'mul'")

    ######################################################################################################################## 
    def __weekly_cycle_extraction_function(self, date_column):
        '''
            A function which extracts the weekly cycle from the date column provided
            to it.

            Parameters:
            ------------
            date_column -       [pd.Series] A pandas series of date type which supports
                                datetime functions.

            Returns:
            ------------
            normalized_data -   [pd.Series] A pandas series with values in the half
                                open interval [0, 1.0) where 1.0 would correspond
                                to a full cycle, and each value correspond to the
                                position in the cycle.
        '''

        day_of_week = date_column.dt.dayofweek      # days already start at 0
        if (self.__trained_on_weekend):
            normalized_data = day_of_week/7             # normalize data to be between 0.0 and 1.0
        else:
            normalized_data = day_of_week/5             # normalize data to be between 0.0 and 1.0

        return normalized_data
    ######################################################################################################################## 
    def __monthly_cycle_extraction_function(self, date_column):
        '''
            A function which extracts the monthly cycle from the date column provided
            to it.

            Parameters:
            ------------
            date_column -       [pd.Series] A pandas series of date type which supports
                                datetime functions.

            Returns:
            ------------
            normalized_data -   [pd.Series] A pandas series with values in the half
                                open interval [0, 1.0) where 1.0 would correspond
                                to a full cycle, and each value correspond to the
                                position in the cycle.
        '''

        day_of_month = date_column.dt.day - 1      # make days start at 0
        normalized_data = day_of_month/31          # normalize data to be between 0.0 and 1.0

        return normalized_data
    ######################################################################################################################## 
    def __yearly_cycle_extraction_function(self, date_column):
        '''
            A function which extracts the yearly cycle from the date column provided
            to it.

            Parameters:
            ------------
            date_column -       [pd.Series] A pandas series of date type which supports
                                datetime functions.

            Returns:
            ------------
            normalized_data -   [pd.Series] A pandas series with values in the half
                                open interval [0, 1.0) where 1.0 would correspond
                                to a full cycle, and each value correspond to the
                                position in the cycle.
        '''

        day_of_year = date_column.dt.dayofyear - 1  # make days start at 0
        normalized_data = day_of_year/366           # normalize data to be between 0.0 and 1.0

        return normalized_data

    ######################################################################################################################## 
    def add_seasonality(self, seasonality_name, fourier_order, cycle_extraction_function, seasonality_mode="add"):
        '''
            A function to add a requested seasonality to the inference process. The seasonality is
            added to a list of seasonalities to construct, but is not constructed here. It is
            constructed during the .fit method.


            Parameters:
            ------------
            seasonality_name -          [str] The name of the seasonality. Must be
                                        a unique name and cannot be one of the
                                        reserved names. e.g. 'yearly', 'monthly',
                                        or 'daily'.

            fourier_order -             [int] The fourier order of the seasonality.
                                        Must be a positive integer. Higher values allow
                                        for better fitting, but might also overfit.

            cycle_extraction_function - [callable] A function which implements cycle
                                        extraction from a date column. It should
                                        accept a pandas series of datetime and
                                        return a pandas series of the same size with
                                        values from the half open interval [0.0, 1.0)

            seasonality_mode -          [str] The mode of this seasonality. Must be
                                        either "add" or "mul".


            Returns:
            ------------
            None
        '''
        if (not self.__is_regressor_name_available(seasonality_name)):
            raise ValueError(f"Name {seasonality_name} is already in use")

        seasonality_information = {"name": seasonality_name,
                                   "order": fourier_order,
                                   "extraction_function":cycle_extraction_function}

        

        if (seasonality_mode == "add"):            
            self.__additive_seasonalities.append(seasonality_information)
        elif (seasonality_mode == "mul"):
            self.__multiplicative_seasonalities.append(seasonality_information)
        else:
            raise ValueError(f"Seasonality mode {seasonality_mode} is unsupported. Must be one of 'add' or 'mul'")

        self.__reserved_names.append(seasonality_name)

        
    ######################################################################################################################## 
    def __transform_data(self, data):
        '''
            A function which takes the raw data containing the timestamp and
            target column, and returns tensors for the trend, seasonality, and additional
            components

            Parameters:
            ------------
            data -          [DataFrame] The dataframe with the raw data. 
                            Must contain at least one column with the 
                            timestamp (with dtype np.datetime64). It 
                            can optionally also contain the 
                            target column

            
            Returns:
            ------------
            X_time -        [tensor] A tensor of shape (n_samples, ), 
                            where n_samples is the number of samples in 
                            data

            X_dataframe -   [pd.DataFrame] - A pandas dataframe of all
                            columns except the target column if it is present

            y -             [tensor] A tensor of shape (n_samples, ), 
                            where n_samples is the number of samples in
                            data, or None if there is no target column in
                            the original data
        '''

        # make a copy to avoid side effects of changing the original df
        internal_data = data.copy()

        for regressor_list in [self.__additive_additional_regressors, self.__multiplicative_additional_regressors]:
            for regressor_name in regressor_list:
                if regressor_name not in internal_data.columns.values:
                    raise KeyError(f"Could not find regressor '{regressor_name}' in data provided")

        if (self.__target_col in internal_data.columns):
            X_dataframe = internal_data.drop(self.__target_col, axis=1)
        else:
            X_dataframe = internal_data


        # Convert ms values to seconds
        internal_data[self.__time_col] = internal_data[self.__time_col].values.astype(float)/1e9

        if (self.__history_min_time_seconds is None):
            self.__history_min_time_seconds = internal_data[self.__time_col].min()
            self.__history_max_time_seconds = internal_data[self.__time_col].max()

        # Make time column go from 0 to 1
        internal_data[self.__time_col] = internal_data[self.__time_col] - self.__history_min_time_seconds
        internal_data[self.__time_col] = internal_data[self.__time_col]/(self.__history_max_time_seconds - self.__history_min_time_seconds)

        X_time = torch.tensor(internal_data[self.__time_col].values, dtype=torch.float32)
        
        


        # we only want to define y_max once
        if (self.__y_max is None):
            self.__y_max = internal_data[self.__target_col].max()


        # If we don't have a target column (i.e. we're predicting), don't try and grab it
        if (self.__target_col in internal_data.columns):

            # Possion distribution requires counts, so we don't want to scale for it
            # TODO: make sure this matches the code in chronos utils if we make a Poisson distribution            
            if (self.__y_likelihood_distribution not in ["Poisson"]):
                y_values = internal_data[self.__target_col].values/self.__y_max
            else:
                y_values = internal_data[self.__target_col].values
            
            y = torch.tensor(y_values, dtype=torch.float32)

        else:
            y = None
        

        
        return X_time, X_dataframe, y
        
    ########################################################################################################################
    def __find_changepoint_positions(self, X_time, changepoint_num, changepoint_range, min_value = None, drop_first = True):
        '''
            A function which takes a tensor of the time, expressed in days, and the
            number oc changepoints to find, and finds the desired number of changepoints.


            Parameters:
            ------------
            X_time -            [tensor] A tensor of the time, expressed in seconds. The 
                                seconds need not be consecutive, or evenly spaced.

            changepoint_num -   [int] The number of changepoints to find

            changepoint_range - [float] The range of the available times to consider. A 
                                value between 0.0 and 1.0. 0.8 means only the first 80% 
                                of the range is considered

            min_value -         [int] The timepoint which describes the beginning of the
                                range where changepoints can be found. Default is None, 
                                which means the first measurement sets the beginning of 
                                the range

            drop_first -        [bool] Whether to drop the first measurement found. 
                                When True, this prevents from the first measurement of
                                being considered as a changepoint (we don't want the 
                                first second to be a changepoint usually)

            Returns:
            ------------
            changepoints -      [tensor] A tensor of shape (changepoint_num, ) where each 
                                entry is a day where a changepoint can happen. The
                                changepoints are chosen to be evenly spaced based on the
                                DATE RANGE, not the number of samples, in case samples
                                are unevenly spaced.
        '''
        
        # Set the minimum value in case it is None
        if (min_value is None):
            min_value = X_time.min().item()
        
        # Find the maximum value available in the data
        max_value_in_data = X_time.max().item() 

        # We usually don't want to consider the entire range, so we only
        # consider a certain section, dictated by changepoint_range
        max_distance = (max_value_in_data - min_value) * changepoint_range
        max_value = min_value + max_distance

        # When fitting, we don't want the first day to be a changepoint candidate
        # However, when predicting the future, it is very much possible our first
        # prediction day is a changepoint
        if (drop_first):
            changepoints =  np.linspace(min_value, max_value, changepoint_num+1, dtype=np.float32)
            changepoints = changepoints[1:] # The first entry will always be 0, but we don't
                                            # want a changepoint right in the beginning
        else:
            changepoints =  np.linspace(min_value, max_value, changepoint_num, dtype=np.float32)


        changepoints = torch.tensor(changepoints, dtype=torch.float32)

        return changepoints
        
    ########################################################################################################################
    def __make_A_matrix(self, X_time, changepoints):
        '''
            A function which takes in the time tensor, and the changepoints
            chosen, and produces a matrix A which specifies when to add the
            effect of the changepoints

            Parameters:
            ------------
            X_time -        [tensor] A tensor of the time, in seconds

            changepoints -  [tensor] A tensor of changepoints where each element
                            is a second when a changepoint can happen

            Returns:
            ------------
            A -             [tensor] A tensor of shape (n_samples, S), where n_samples
                            is the number of samples in X_time, and S
                            is the number of changepoints
        '''

        
        A = torch.zeros((X_time.shape[0], len(changepoints)))

        # For each row t and column j,
        # A(t, j) = 1 if X_time[t] >= changepoints[j]. i.e. if the current time 
        # denoted by that row is greater or equal to the time of the most recent
        # changepoint

        for j in range(A.shape[1]):
            row_mask = (X_time >= changepoints[j])
            A[row_mask, j] = 1.0
            
        
        return A

    ########################################################################################################################
    def __check_incoming_data_for_nulls(self, data, predictions=False):
        '''
            Checks incoming data for null values and advises the user what to
            do

            Parameters:
            ------------
            data -          [pd.DataFrame] - A pandas dataframe which contains
                            the incoming data

            predictions -   [bool] A flag which determines if we are in prediction
                            mode. If we are, we don't care about the target column
                            containing null values since we won't be using it

            Returns:
            ------------
            None
        '''

        if (data[self.__target_col].isna().sum() > 0):
            if (predictions == False):
                raise ValueError(f"{self.__target_col} contains null values, which Chronos cannot process. Consider either removing the null values, or setting them as 0, whichever makes more sense for your data")
        elif (data[self.__time_col].isna().sum() > 0):
            raise ValueError(f"{self.__time_col} contains null values, which Chronos cannot process. Consider removing these values")
        else:
            for col in data.columns:
                if (col not in [self.__time_col, self.__target_col]):
                    if (data[col].isna().sum() > 0):
                        raise ValueError(f"{col} contains null values, which Chronos cannot process. Consider removing these values")
    ########################################################################################################################
    def fit(self, data, time_col = "ds", target_col="y"):
        '''
            A function which performs fitting of the required method on the data provided,
            and thus estimates the parameters of this model.


            Parameters:
            ------------
            data -          [DataFrame] A pandas dataframe with at least two columns. One
                            specifying the timestamp, and one specifying the target value 
                            (the time series observations). The default expected column 
                            names are 'ds' and 'y' but can be set to other names.

            time_col -      [str] A string denoting the name of the timestamp column.
                            
                            Default is 'ds'

            target_col -    [str] A string denoting the name of the time series 
                            observation column.
                            
                            Default is 'y'

            
            Returns:
            ------------
            self -          [Chronos] A fitted Chronos model
            
        '''
        # Record the time-series named columns. We will use them a lot
        self.__time_col = time_col
        self.__target_col = target_col
        
        self.__check_incoming_data_for_nulls(data, predictions=False)

        # Make a copy of the history
        self.history = data.copy()
        if (self.history[self.__time_col].dt.day_name().isin(["Sunday", "Saturday"]).any() == False):
            print("No weekends found in training data, will only consider Monday - Friday")
            self.__trained_on_weekend = False
        else:
            self.__trained_on_weekend = True

        

        
        X_time, X_dataframe, y = self.__transform_data(data)
        
        
        

        number_of_valid_changepoints = int(X_time.shape[0] * self.__proportion_of_data_subject_to_changepoints)
        if (number_of_valid_changepoints < self.__number_of_changepoints):
            warnings.warn(f"Number of datapoints in range, {number_of_valid_changepoints}, is smaller than number of changepoints, {self.__number_of_changepoints}. Using {number_of_valid_changepoints} instead", RuntimeWarning)
            self.__number_of_changepoints = number_of_valid_changepoints

        # Compute the changepoint frequency in changepoints/seconds
        self.__changepoint_frequency = self.__number_of_changepoints/(self.__history_max_time_seconds - self.__history_min_time_seconds)
        
        
        
        # Find a set of evenly spaced changepoints in the training data, and 
        # buid a matrix describing the effect of the changepoints on each timepoint
        self.__changepoints = self.__find_changepoint_positions(X_time, 
                                                                self.__number_of_changepoints, 
                                                                self.__proportion_of_data_subject_to_changepoints)
        
        
        self.__trend_components = {}
        self.__multiplicative_components = {}
        self.__additive_components = {}
        
        
        if (self.__method in chronos_utils.SUPPORTED_METHODS):  
            if (self.__method == "MLE"):
                print("Employing Maximum Likelihood Estimation")
                self.__model = self.__model_function
                self.__guide = self.__guide_MLE
                self.__param_prefix = ""
                
            elif (self.__method == "MAP"):
                print("Employing Maximum A Posteriori")
                self.__model = self.__model_function
                self.__guide = AutoDelta(self.__model, init_loc_fn=init_to_feasible)
                self.__param_prefix  = "AutoDelta."
                
            elif (self.__method == "SVI"):
                print("Employing Stochastic Variational Inference")
                self.__model = self.__model_function
                self.__guide = AutoNormal(self.__model, init_loc_fn=init_to_feasible)
                self.__param_prefix = "AutoNormal."

            self.__train_point_estimate(X_time,
                                        X_dataframe,
                                        y)
        elif (self.__method == "MCMC"):
            print("Employing Markov Chain Monte Carlo")
            raise NotImplementedError("Did not implement MCMC methods")
        

        return self
            
    
    ########################################################################################################################
    def __train_point_estimate(self, X_time, X_dataframe, y):
        '''
            A function which takes in the model and guide to use for
            the training of point estimates of the parameters, as well as
            the regressor tensors, the changepoint matrix, and the target,
            and performs optimization on the model parameters

            Parameters:
            ------------
            X_time -        [tensor] The time tensor specifying the 
                            time regressor

            X_dataframe -   [pd.DataFrame] A pandas dataframe containing all
                            columns except for the target. This dataframe is
                            included if additional regressors are requested

            y -             [tensor] The target to predict, i.e. the time
                            series measurements

            Returns:
            ------------
            None
        '''
        
        # Make sure we are working with a fresh param store 
        # TODO: see if there is a way to protect this
        pyro.clear_param_store()
        
        # Use a decaying optimizer which starts with a given learning
        # rate, but then slowly drops it to take smaller and smaller
        # steps
        optimizer = Rprop
        scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 
                                              'optim_args': {'lr': self.__learning_rate}, 
                                              'gamma': 0.9})
        
        # Use the ELBO (evidence lower bound) loss function
        # and Stochastic Variational Inference optimzation
        # technique
        my_loss = Trace_ELBO()
        self.svi_ = SVI(self.__model, 
                        self.__guide, 
                        scheduler, 
                        loss=my_loss)
        
        # Calculate when to print output
        print_interval = max(self.__number_of_iterations//10000, 10)


        # Keep track of this for MAE metric
        y_true = y.detach().numpy().copy()
        if (self.__y_max is not None):
            y_true *= self.__y_max
        
        # Create a predictive object to predict for us for
        # metric reporting purposes
        predictive = Predictive(model=self.__model,
                                guide=self.__guide,
                                num_samples=1,
                                return_sites=("_RETURN",)) 
        
        # Iterate through the optimization
        for step in range(self.__number_of_iterations):
            
            loss = self.svi_.step(X_time, 
                                  X_dataframe, 
                                  y)

            # After calculating the loss, normalize by the 
            # number of points
            loss = round(loss/y.shape[0], 4)

            
            
            # If required, print out the results
            if (step % print_interval == 0):
                pct_done = round(100*(step+1)/self.__number_of_iterations, 2)

                # If we're reporting, grab samples for the predictions
                samples = predictive(X_time, 
                                     X_dataframe)
                
                y_pred = samples["_RETURN"].detach().numpy()[0]
                if (self.__y_max is not None):
                    y_pred *= self.__y_max
                
                
                # Calculate mean absolute error and format it nicely
                mean_absolute_error_loss = "{:.4f}".format(np.mean(np.abs(y_true - y_pred)))
                
                
                print(" "*100, end="\r")
                print(f"{pct_done}% - ELBO loss: {loss} | Mean Absolute Error: {mean_absolute_error_loss}", end="\r")
        
        # Always have a final printout
        pct_done = 100.0
        print(" "*100, end="\r")
        print(f"{pct_done}% - ELBO loss: {loss} | Mean Absolute Error: {mean_absolute_error_loss}")
            
    ########################################################################################################################
    def __add_future_changepoints(self, past_deltas, future_trend_period):
        '''
            A function which accepts a changepoint matrix, and a changepoint rate change 
            tensor and adds compares their sizes. If the matrix A specifies more 
            changepoints than deltas, new changepoint values are added to deltas. 
            Otherwise deltas is unchanged

            The additions to deltas are randomly drawn from a Laplace distribution since 
            they are simulations of potential future changepoints, and thus are not fixed.
            Each run of this function is designed to be a single possible future.

            Parameters:
            ------------

            past_deltas -           [tensor] A 1D tensor specifying the increase, or 
                                    decrease, in slope at each changepoint. The size is 
                                    (S, ) where S is the number of changepoints

            future_trend_period -   [int] The duration of the future trend, in seconds. 
                                    This is the number of seconds the future trend spans, 
                                    not the number  of observations (for example, there 
                                    can be two  observations, 15 seconds apart, so the 
                                    period will be 15 seconds)
            
            Returns:
            ------------
            deltas -                [tensor] A new 1D tensor which contains the increase, 
                                    or decrease, in slope for both past and future 
                                    changepoints. 
            
        '''
        # Find the number of future changepoints in this simulation
        extra_changepoint_num = np.random.binomial(n=future_trend_period, p = self.__changepoint_frequency)

        

        # Infer future changepoint scale
        future_laplace_scale = torch.abs(past_deltas).mean()


        if (future_laplace_scale > 0.0):
            changepoint_dist = torch.distributions.Laplace(0, future_laplace_scale)
        
            # The future changepoints can be any value from the
            # inferred Laplace distribution
            future_deltas = changepoint_dist.sample((extra_changepoint_num,))
        else:
        
            future_deltas = torch.zeros(extra_changepoint_num)

        # Combine the past change rates as 
        # well as future ones
        deltas = torch.cat([past_deltas, future_deltas])

        
        return deltas

    ########################################################################################################################
    def __simulate_potential_future(self, X_time, past_deltas):
        '''
            A function which simulates a potential future to account for future
            changepoints over X_time. The future may or may not contain changepoints
            and so a single run of this function simulates a potential future
            where additional changepoints are added. 
            X_time can contain the time tensor for both past and future


            Parameters:
            ------------
            
            X_time -                [tensor] The time tensor specifying the time regressor


            past_deltas -           [tensor] A 1D tensor specifying the increase, or decrease,
                                    in slope at each changepoint. The size is (S, ) where S 
                                    is the number of changepoints in the past, and not for 
                                    the entire duration of X_time
            
            Returns:
            ------------
            A tuple of (deltas, combined_changepoints, A)

            deltas -                [tensor] A 1D tensor of the rate adjustments for the
                                    entire time of X_time.
            
            future_changepoints -   [tensor] A 1D Tensor specifing the times, where each 
                                    changepoint occurs. May be the same size as 
                                    past_changepoints if no new changepoints have been 
                                    simulated
            
        '''  
        # Simulate potential changepoint generation We've scaled the history so the last timestamp
        # is 1.0, so we need to find, proportionally, how much the future period is 
        # bigger
        
        
        future_raw_value = (X_time.max() - 1.0)
        future_seconds_number = int(future_raw_value * (self.__history_max_time_seconds - self.__history_min_time_seconds))
        
        
        deltas = self.__add_future_changepoints(past_deltas, future_seconds_number)

        
        # Count the number of future changepoints simulated
        future_changepoint_number = int(deltas.shape[0] - self.__number_of_changepoints)
        
        
        # If we need to simulate a certain number of future changepoints,
        # we will randomly draw their positions and create a new A
        # matrix to be used to correct the trend.
        # Otherwise, we can continue as usual
        if (future_changepoint_number > 0):

            # The values start at 1.0, and torch doesn't have a random number generator
            # for random floats, only the values from 0.0 - 1.0, so need to do some magic
            first_future_trend_value = 1.0#X_time[-future_changepoint_number].item()
            last_future_trend_value = X_time[-1].item()

            
            # Make some random values
            random_values = torch.rand(size = (future_changepoint_number, )).type(torch.float32)
            
            # Employ inverse scaling to get the values from 0.0 - 1.0 to first_future_trend_value - last_future_trend_value
            future_changepoints = random_values * (last_future_trend_value - first_future_trend_value) + first_future_trend_value
        else:
            future_changepoints = torch.tensor([])
        
        return deltas, future_changepoints

    ########################################################################################################################
    def __sample_initial_slope_and_intercept(self, method):
        '''
            A function which samples the initial values for the slope and
            intercept. Depending on the method these could be distributions
            or single parameters

            Parameters:
            ------------
            method -    [str] A method describing which sampling method
                        to employ. If MAP, the samples are from a prior
                        distribution. If MLE, the samples are point parameters.

            Returns:
            ------------
            return_tuple -  [tuple] A tuple of an intercept and slope;
                            intercept_init -    [float] The value of the initial 
                                                intercept
                            slope_init -        [float] The value of the initial
                                                slope 
        '''
        if (method in ["MAP", "SVI"]):
            # Define paramters for the constant and slope
            intercept_init = pyro.sample("intercept", 
                                         dist.Normal(0.0, 10.0))
            slope_init = pyro.sample("trend_slope", 
                                     dist.Normal(0.0, 10.0))
        elif (method == "MLE"):
            intercept_init = pyro.param("intercept", 
                                        torch.tensor(0.0))
            slope_init = pyro.param("trend_slope", 
                                    torch.tensor(0.0))
        
        return intercept_init, slope_init
    ########################################################################################################################
    def __sample_past_slope_changes(self, method):
        '''
            A function which samples the values for the slope changes.
            Depending on the method these could be distributions
            or single parameters

            Parameters:
            ------------
            method -        [str] A method describing which sampling method
                            to employ. If MAP, the samples are from a prior
                            distribution. If MLE, the samples are point parameters.

            Returns:
            ------------
            past_deltas -   [tensor] A tensor of the slope changes for the
                            trend component.
        '''
        if (method in ["MAP", "SVI"]):
            # sample from a Laplace distribution
            means = torch.zeros(self.__number_of_changepoints)
            scales = torch.full((self.__number_of_changepoints, ), self.__changepoint_prior_scale)

            slope_change_laplace_distribution = dist.Laplace(means, scales).to_event(1)

            past_deltas = pyro.sample("delta", slope_change_laplace_distribution)
        elif (method == "MLE"):            
            past_deltas = pyro.param("delta", 
                                     torch.zeros(self.__number_of_changepoints))
        
        return past_deltas
    ########################################################################################################################
    def __sample_seasonalities_coefficients(self, method, seasonality_component, seasonality_name):
        '''
            A function which samples the values for the seasonality coefficients.
            Depending on the method these could be distributions or single parameters

            Parameters:
            ------------
            method -                [str] A method describing which sampling method
                                    to employ. If MAP, the samples are from a prior
                                    distribution. If MLE, the samples are point 
                                    parameters.

            seasonality_component - [tensor] The matrix of seasonality fourier
                                    sine-cosine pairs. This matrix has an even number
                                    of rows. Each row pair is a sine-cosine pair of
                                    the seasonality fourier series

            seasonality_name -      [str] The name of the seasonality          

            Returns:
            ------------
            betas_seasonality - [tensor] A tensor of the coefficients for the seasonality
                                components. For a single seasonality these are
                                coefficients for a pair of sine-cosine fourier terms
        '''
        if (method in ["MAP", "SVI"]):
            # sample from a normal distribution
            means = torch.zeros(seasonality_component.size(1))
            standard_deviations = torch.full((seasonality_component.size(1),), 10.0)

            seasonality_coefficient_normal_distribution = dist.Normal(means, standard_deviations).to_event(1)

            betas_seasonality = pyro.sample(f"betas_{seasonality_name}", 
                                            seasonality_coefficient_normal_distribution)
        elif (method == "MLE"):
            betas_seasonality = pyro.param(f"betas_{seasonality_name}", 
                                            torch.zeros(seasonality_component.size(1)))

        return betas_seasonality
        
    ########################################################################################################################
    def __sample_additional_regressors_coefficients(self, method, regressor_compoents, regressor_modes):
        '''
            A function which samples the values for the additional regressor
            coefficients. Depending on the method these could be distributions
            or single parameters

            Parameters:
            ------------
            method -                [str] A method describing which sampling method
                                    to employ. If MAP, the samples are from a prior
                                    distribution. If MLE, the samples are point parameters

            regressor_compoents -   [tensor] A matrix of the additional regressors
                                    for which coefficients are requested
            
            regressor_modes -       [str] The mode of regressor incorporation. 
                                    provided for naming purposes but should be
                                    either "mul" or "add"

            Returns:
            ------------
            betas_regressors -      [tensor] A tensor of the coefficients for the 
                                    additional regressor components. 
        '''
        if (method in ["MAP", "SVI"]):
            means = torch.zeros(regressor_compoents.size(1))
            standard_deviations = torch.full((regressor_compoents.size(1),), 10.0)

            additional_regressor_normal_distribution = dist.Normal(means, standard_deviations).to_event(1)

            # Compute coefficients for the regressors
            betas_regressors = pyro.sample(f"betas_{regressor_modes}_regressors", 
                                           additional_regressor_normal_distribution)
        elif (method == "MLE"):
            betas_regressors = pyro.param(f"betas_{regressor_modes}_regressors", 
                                          torch.zeros(regressor_compoents.size(1)))

        return betas_regressors

    ########################################################################################################################
    ########################################################################################################################    
    ########################################################################################################################
    def __guide_MLE(self, X_time, X_dataframe, y=None):
        '''
            A function which specifies a special guide which does nothing.
            This guide is used in MLE optimization since there is no
            relationship between parameters and prior distributions.

            Parameters:
            ------------
            
            X_time -        [tensor] The time tensor specifying the time
                            regressor

            X_dataframe -   [pd.DataFrame] A pandas dataframe containing all
                            columns except for the target. This dataframe is
                            included if additional regressors are requested

            y -             [tensor] The target to predict, i.e. the time
                            series measurements. If None, the model 
                            generates these observations instead.

            
            Returns:
            ------------
            None
            
        '''
        pass
    
    ########################################################################################################################
    ########################################################################################################################
    def __predict_normal_likelihood(self, method, mu, y):
        '''
            A function which takes up the expected values and employs them
            to specify a normal distribution, conditioned on the 
            observed data.
            An additional sigma (standard deviation) value is registered as 
            either a distribution or a parameter based on the method used

            Parameters:
            ------------

            method -    [str] Which method is used. Either MAP
                        or MLE
            
            mu -        [tensor] The expected values computed from
                        the model paramters

            y -         [tensor] The observed values
            
            Returns:
            ------------
            None
        '''

        
        # Define additional paramters specifying the likelihood
        # distribution. 
        if (method in ["MAP", "SVI"]):
            sigma = pyro.sample("sigma", dist.HalfCauchy(1.0))
        elif (method == "MLE"):
            sigma = pyro.param("sigma", 
                               torch.tensor(1.0), 
                               constraint = constraints.positive)
        
         
        # Finally sample from the likelihood distribution and
        # optionally condition on the observed values. 
        # If y is None, this simply samples from the distribution
        with pyro.plate("data", mu.size(0)):
            pyro.sample("obs", dist.Normal(mu, sigma), obs=y)

        
    ########################################################################################################################
    def __predict_studentT_likelihood(self, method, mu, y):
        '''
            A function which takes up the expected values and employs them
            to specify a Student t-distribution, conditioned on the 
            observed data.
            Additional sigma (standard deviation), and df (degrees of freedom) 
            values are registered as either distributions or parameters 
            based on the method used

            Parameters:
            ------------

            method -    [str] Which method is used. Either MAP
                        or MLE
            
            mu -        [tensor] The expected values computed from
                        the model paramters

            y -         [tensor] The observed values
            
            Returns:
            ------------
            None
        '''
        

        # Define additional paramters specifying the likelihood
        # distribution. 
        if (method in ["MAP", "SVI"]):
            sigma = pyro.sample("sigma", dist.HalfCauchy(1.0))
            df = pyro.sample("df", dist.HalfCauchy(1.0))
        elif (method == "MLE"):
            sigma = pyro.param("sigma", 
                               torch.tensor(1.0), 
                               constraint = constraints.positive)

            df = pyro.param("df", 
                            torch.tensor(1.0), 
                            constraint = constraints.positive)
         
        # Finally sample from the likelihood distribution and
        # optionally condition on the observed values. 
        # If y is None, this simply samples from the distribution
        with pyro.plate("data", mu.size(0)):
            pyro.sample("obs", dist.StudentT(df, mu, sigma), obs=y)

        
    ########################################################################################################################
    def __predict_gamma_likelihood(self, method, mu, y):
        '''
            A function which takes up the expected values and employs them
            to specify a gamma distribution, conditioned on the 
            observed data.
            An additional rate value is registered as either a distribution 
            or a parameter based on the method used, and a shape tensor
            is computed based on the rate and mu.

            Parameters:
            ------------

            method -    [str] Which method is used. Either MAP
                        or MLE
            
            mu -        [tensor] The expected values computed from
                        the model paramters

            y -         [tensor] The observed values
            
            Returns:
            ------------
            None
        '''

        # Define additional paramters specifying the likelihood
        # distribution. 
        if (method in ["MAP", "SVI"]):
            rate = pyro.sample("rate", dist.HalfCauchy(1.0)).clamp(min=torch.finfo(torch.float32).eps)
        elif (method == "MLE"):
            rate = pyro.param("rate", 
                              torch.tensor(1.0), 
                              constraint = constraints.positive)

        shape = rate * mu
         
        if (y is not None):
            y_obs = y + torch.finfo(torch.float32).eps
        else:
            y_obs = y
        # Finally sample from the likelihood distribution and
        # optionally condition on the observed values. 
        # If y is None, this simply samples from the distribution
        with pyro.plate("data", mu.size(0)):
            pyro.sample("obs", dist.Gamma(concentration=shape, rate=rate), obs=y_obs)

        
    ########################################################################################################################
    ########################################################################################################################    
    def __predict_likelihood(self, method, distribution, mu, y):
        '''
            A function which takes up the trend, seasonalities, and additional regressors
            and combines them to form the expected values, then conditions them
            on the observed data based on the distribution requested

            Parameters:
            ------------

            method -        [str] Which method is used

            distribution -  [str] Which distribution to use. Must be one
                            of the distributions specified in chronos_utils
            
            mu -            [tensor] The expected values computed from
                            the model paramters

            y -             [tensor] The observed values
            
            Returns:
            ------------
            None
        '''

        if (distribution == chronos_utils.Normal_dist_code):
            self.__predict_normal_likelihood(method, mu, y)
        elif (distribution == chronos_utils.StudentT_dist_code):
            self.__predict_studentT_likelihood(method, mu, y)
        elif (distribution == chronos_utils.Gamma_dist_code):
            self.__predict_gamma_likelihood(method, mu, y)
        
    ########################################################################################################################
    def __compute_changepoint_positions_and_values(self, X_time, y):
        '''
            A function which finds the position of changepoints in the time
            frame provided, and samples their values. If unobserved time
            frame is provided (prediction), also simulated future
            changepoints.

            Parameters:
            ------------
            X_time -            [tensor] The tensor of timesteps, in seconds

            y -                 [tensor] The tensor of observed values. Or None if
                                we are in prediction mode

            Returns:
            ------------
            changepoint_tuple - [tuple] A tuple of:
                                deltas -                [tensor] The changepoint values
                                combined_changepoints - [tensor] the changepoint positions
        '''
        past_deltas = self.__sample_past_slope_changes(self.__method)

        # If no observations are given, we assume we are in
        # prediction mode. Therefore, we have to generate possible scenarios
        # for the future changepoints as a simulation
        if (y is None):
            deltas, future_changepoints = self.__simulate_potential_future(X_time, past_deltas)            
            combined_changepoints = torch.cat([self.__changepoints, future_changepoints])            
        else:
            # If we are not in prediction mode, we only care about learning the past
            deltas = past_deltas
            combined_changepoints = self.__changepoints

        return deltas, combined_changepoints
    ########################################################################################################################
    def __compute_trend(self, X_time, y):
        '''
            A function which computes the trend component T(t). The function
            computes the initial slope, intercept, and changepoints for the
            time series and then combines them.


            Parameters:
            ------------
            X_time -    [tensor] The tensor of timesteps, in seconds

            y -         [tensor] The tensor of observed values. Or None if
                        we are in prediction mode

            Returns:
            ------------
            trend -     [tensor] The trend tensor which describes the growth
                        excluding seasonalities and additional regressors
        '''


        intercept_init, slope_init = self.__sample_initial_slope_and_intercept(self.__method)

        deltas, combined_changepoints = self.__compute_changepoint_positions_and_values(X_time, y)
        A = self.__make_A_matrix(X_time, combined_changepoints)


        # To adjust the rates we also need to adjust the displacement during each rate change
        intercept_adjustments = -deltas * combined_changepoints

        # There is a unique slope value and intercept value for each
        # timepoint to create a piece-wise function
        slope = slope_init + torch.matmul(A, deltas)
        intercept = intercept_init + torch.matmul(A, intercept_adjustments)

        # Finally compute the trend component and record it in the global
        # parameter store using the pyro.deterministic command
        trend = slope * X_time + intercept
        if (self.__make_likelihood_mean_positive == True):
            trend = torch.nn.functional.softplus(trend, beta=100)

        pyro.deterministic('trend', trend)


        return trend
    ########################################################################################################################
    def __create_seasonality_from_specs(self, X_date, seasonality_specs, seasonality_memoization_dictionary):
        '''
            A function which takes up a series of dates, and specs on how to construct
            seasonality from the dates, and returns the constructed seasonality.
            If the seasonality fourier terms have been calculated once they are
            loaded from the seasonality_memoization_dictionary. Otherwise, they are
            stored there after being computed

            Parameters:
            ------------
            X_date -                                [pd.Series] A pandas series of dates
                                                    as dtype pd.datetime64
            
            seasonality_specs -                     [dict] A dictionary with the
                                                    seasonality specifications
            
            seasonality_memoization_dictionary -    [dict] A memoization dictionary which
                                                    stores the fourier components of the
                                                    seasonality after it is calculated 
                                                    once

            Returns:
            ------------
            X_seasonality -                         [tensor] A tensor containing the
                                                    values of the computed seasonality.
                                                    The tensor is the same length as
                                                    the date series inputted.
        '''
        seasonality_name = seasonality_specs['name']
        seasonality_order = seasonality_specs['order']
        seasonality_extraction_function = seasonality_specs['extraction_function']

        if (seasonality_name in seasonality_memoization_dictionary):
            seasonality_tensor = seasonality_memoization_dictionary[seasonality_name]
        else:
            cycle = torch.tensor(2 * np.pi * seasonality_extraction_function(X_date).values)
            seasonality_tensor = torch.empty(X_date.shape[0], seasonality_order*2)

            index = 0
            for f in range(seasonality_order):
                fourier_term = (f+1) * cycle
                seasonality_tensor[:, index] = np.sin(fourier_term)
                seasonality_tensor[:, index+1] = np.cos(fourier_term)

                index += 2
            
            seasonality_memoization_dictionary[seasonality_name] = seasonality_tensor

        betas_seasonality = self.__sample_seasonalities_coefficients(self.__method, seasonality_tensor, seasonality_name)
        X_seasonality = seasonality_tensor.matmul(betas_seasonality)

        return X_seasonality
    ########################################################################################################################
    def __compute_multiplicative_seasonalities_product(self, X_date):
        '''
            A function which accepts a pandas date series and computes all
            multiplicative seasonalities. Then combines the seasonalities
            by computing their product.
            e.g. if seasonalities s1, s2, and s3 are requested, the resulting
            tensor is (s1 * s2 * s3).
            The seasonalities have 1.0 added to them so that positive values
            amplify the results, and negative values dampen the results.

            Parameters:
            ------------
            X_date -                        [pd.Series] A pandas series of dates as 
                                            dtype pd.datetime64

            Returns:
            ------------
            total_seasonalities_product -   [tensor] A tensor of all multiplicative
                                            seasonalities, multiplied together.
        '''
        total_seasonalities_product = torch.ones(X_date.shape[0], )

        for multiplicative_seasonality in self.__multiplicative_seasonalities:
            X_seasonality = self.__create_seasonality_from_specs(X_date, 
                                                                 multiplicative_seasonality, 
                                                                 self.__multiplicative_components)

            X_seasonality = 1.0 + X_seasonality

            total_seasonalities_product = total_seasonalities_product * X_seasonality


        return total_seasonalities_product
    ########################################################################################################################
    def __compute_multiplicative_regressors_product(self, X_dataframe):
        '''
            A function which accepts a pandas dataframe and computes all
            multiplicative regressors' coefficients and effects. 
            Then combines the regressors by computing the individual
            regressors' product.
            e.g. if regressor effects r1, r2, and r3 are requested, the resulting
            tensor is (r1 * r2 * r3).
            The regressors have 1.0 added to them so that positive values
            amplify the results, and negative values dampen the results.

            Parameters:
            ------------
            X_dataframe -                       [pd.DataFrame] A pandas dataframe which
                                                contains the regressor values.

            Returns:
            ------------
            multiplicative_regressors_product - [tensor] A tensor of all multiplicative
                                                seasonalities, multiplied together.
        '''
        for additional_regressor in self.__multiplicative_additional_regressors:
            if (additional_regressor not in X_dataframe.columns):
                raise KeyError(f"Regressor '{additional_regressor}' not found in provided dataframe")

        if ("regressors" not in self.__multiplicative_components):
            X_multiplicative_regressors = torch.tensor(X_dataframe[self.__multiplicative_additional_regressors].values)
            self.__multiplicative_components["regressors"] = X_multiplicative_regressors
        else:
            X_multiplicative_regressors = self.__multiplicative_components["regressors"]

        betas_multiplicative_regressors = self.__sample_additional_regressors_coefficients(self.__method, 
                                                                                           X_multiplicative_regressors, 
                                                                                           "mul")

        # Notice we don't do a matrix-vector product here, but rather an 
        # row-wise multipication. The reason is we want to then find a cumulative
        # product of all multiplicative regressors.
        # i.e. if I have regressors reg1, reg2, and reg3, I want to multiply the trend
        # by (reg1 * reg2 * reg3) rather than by (reg1 + reg2 + reg3)
        multiplicative_regressors = X_multiplicative_regressors * betas_multiplicative_regressors

        # The multiplicative regressors have to be adjusted so that they are larger than 1
        # when positive, and between 0.0 and 1.0 when negative so that their multiplicative
        # effect either dampens, or amplifies, the results, but never flips the sign
        multiplicative_regressors = (1 + multiplicative_regressors) 

        multiplicative_regressors_product = torch.prod(multiplicative_regressors, dim=1)

        return multiplicative_regressors_product

    ########################################################################################################################    
    def __compute_multiplicative_component(self, X_dataframe):
        '''
            A function which looks at the data and computes all components that were
            labeled as multiplicative. The function computes the product of all 
            multiplicative seasonalities, then the product of all multiplicative
            regressors, and then produces their product.

            Parameters:
            ------------
            X_dataframe -               [pd.DataFrame] A pandas dataframe which contains 
                                        all columns of the data except the target

            Returns:
            ------------
            multiplicative_component -  [tenspr] A tensor containing the data for the 
                                        multiplicative component of the model
        '''

        X_date = X_dataframe[self.__time_col]

        multiplicative_seasonalities_product = self.__compute_multiplicative_seasonalities_product(X_date)

        multiplicative_regressors_product = self.__compute_multiplicative_regressors_product(X_dataframe)

        multiplicative_component = multiplicative_seasonalities_product * multiplicative_regressors_product

        return multiplicative_component

    ########################################################################################################################
    def __compute_additive_seasonalities_sum(self, X_date):
        '''
            A function which accepts a pandas date series and computes all
            additive seasonalities. Then combines the seasonalities
            by computing their sum.
            e.g. if seasonalities s1, s2, and s3 are requested, the resulting
            tensor is (s1 + s2 + s3).

            Parameters:
            ------------
            X_date -                    [pd.Series] A pandas series of dates as 
                                        dtype pd.datetime64

            Returns:
            ------------
            total_seasonalities_sum -   [tensor] A tensor of all additive
                                        seasonalities, summed together.
        '''
        total_seasonalities_sum = torch.zeros(X_date.shape[0], )

        for additive_seasonality in self.__additive_seasonalities:
            X_seasonality = self.__create_seasonality_from_specs(X_date, 
                                                                 additive_seasonality, 
                                                                 self.__additive_components)

            total_seasonalities_sum = total_seasonalities_sum + X_seasonality


        return total_seasonalities_sum
    ########################################################################################################################
    def __compute_additive_regressors_sum(self, X_dataframe):
        '''
            A function which accepts a pandas dataframe and computes all
            additive regressors' coefficients and effects. 
            Then combines the regressors by computing the individual
            regressors' sum.
            e.g. if regressor effects r1, r2, and r3 are requested, the resulting
            tensor is (r1 + r2 + r3).

            Parameters:
            ------------
            X_dataframe -               [pd.DataFrame] A pandas dataframe which
                                        contains the regressor values.

            Returns:
            ------------
            additive_regressors_sum -   [tensor] A tensor of all additive
                                        seasonalities, summed together.
        '''
        for additional_regressor in self.__additive_additional_regressors:
            if (additional_regressor not in X_dataframe.columns):
                raise KeyError(f"Regressor '{additional_regressor}' not found in provided dataframe")

        if ("regressors" not in self.__additive_components):
            X_additive_regressors = torch.tensor(X_dataframe[self.__additive_additional_regressors].values, dtype=torch.float32)
            self.__additive_components["regressors"] = X_additive_regressors
        else:
            X_additive_regressors = self.__additive_components["regressors"]

        betas_additive_regressors = self.__sample_additional_regressors_coefficients(self.__method, X_additive_regressors, "add")


        additive_regressors_sum = X_additive_regressors.matmul(betas_additive_regressors)
        

        return additive_regressors_sum
    ########################################################################################################################
    def __compute_additive_component(self, X_dataframe):
        '''
            A function which looks at the data and computes all components that were
            labeled as additive. The function computes the sum of all 
            additive seasonalities, then the sum of all additive
            regressors, and then produces their sum.

            Parameters:
            ------------
            X_dataframe -           [pd.DataFrame] A pandas dataframe which contains 
                                    all columns of the data except the target

            Returns:
            ------------
            additive_component -    [tenspr] A tensor containing the data for the 
                                    additive component of the model
        '''

        X_date = X_dataframe[self.__time_col]

        additive_seasonalities_sum = self.__compute_additive_seasonalities_sum(X_date)

        additive_regressors_sum = self.__compute_additive_regressors_sum(X_dataframe)

        additive_component = additive_seasonalities_sum + additive_regressors_sum

        return additive_component
    ########################################################################################################################    
    def __model_function(self, X_time, X_dataframe, y=None):
        '''
            The major powerhouse function of this object, performs the modeling of
            the generative process which generates y from X_dataframe and X_time.
            Returns the expected values given the regressors and the learned 
            parameters.


            Parameters:
            ------------
            X_time -        [tensor] A tensor of time values, normalized. If
                            we are training, this will contain values between 0.0
                            and 1.0. Otherwise, it will contain values normalized to
                            reflect the range of the training data (i.e. from 1.0 unward
                            if it contains only future times, and from 0.0 to 1.0 and 
                            unward if it contains both past and future times).

            X_dataframe -   [pd.DataFrame] A pandas dataframe which contains the
                            raw data passed into this Chronos object. Used to
                            compute seasonality and additional regressors

            y -             [tesnor] A tensor of the observaed values, normalized to
                            the range 0.0 - 1.0, if we are training, or a None object
                            if we are predicting.

            Returns:
            ------------
            mu -            [tensor] A tensor of the expected values given X_dataframe
                            and the learned parameters. 
        '''

        prediction_mode = y is None
        if (prediction_mode):
            # Poor man's verbose printing of prediction number
            if (self.__prediction_verbose == True):
                self.predict_counter_ += 1
                if (self.predict_counter_ > 0):
                    print(f"Prediction no: {self.predict_counter_}", end="\r")


        trend = self.__compute_trend(X_time, y)

        multiplicative_component = self.__compute_multiplicative_component(X_dataframe)

        additive_component = self.__compute_additive_component(X_dataframe)



        mu = (trend * multiplicative_component) + additive_component

        if (self.__make_likelihood_mean_positive == True):
            mu = torch.nn.functional.softplus(mu, beta=100)
            mu = mu + torch.finfo(torch.float32).eps


        # Sample observations based on the appropriate distribution
        self.__predict_likelihood(self.__method, 
                                  self.__y_likelihood_distribution, 
                                  mu,
                                  y)

        return mu


    ########################################################################################################################
    def predict(self, 
                future_df=None, 
                sample_number=1000, 
                ci_interval=0.95, 
                period=30, 
                frequency='D', 
                include_history=True,
                verbose=True):
        '''
            A function which accepts a dataframe with at least one column, the timestamp
            and employes the learned parameters to predict observations as well as 
            credibility intervals and uncertainty intervals.
            Alternatively, the function can accept the parameters accepted by 
            .make_future_dataframe and produce the future dataframe internally.
            Returns a dataframe with predictions for observations, upper and lower limits
            for credibility intervals, trend, and upper and lower limits on trend 
            uncertainty.

            Parameters:
            ------------
            future_df -         [DataFrame] The dataframe. Must at least have a single 
                                column of timestamp with the same name as the training 
                                dataframe. If data is not provided, period, frequency, 
                                and include_history must be provided. If data is 
                                provided, period, frequency, and include_history are 
                                ignored

                                Default is None

            sample_number -     [int] The number of posterior samples to generate in 
                                order to draw the uncertainty intervals and credibility
                                intervals. Larger values give more accurate results, 
                                but also take longer to run. 
                                
                                Default 1000

            ci_interval -       [float] The credibility interval range to generate. 
                                Must be between 0.0 and 1.0, 0.95 generates a range 
                                such that 95%  of all observations fall within this 
                                range. 
                                
                                Default is 0.95.

            period -            [int] The number of future observations based on
                                frequency. The default is 30, and the default
                                for frequency id 'D' which means 30 days ahead.

                                Default is 30
            
            frequency -         [str] The frequency of the period. See 
                                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
                                for a list of supported freuqncies.
                                The default is 'D' which stands for calendar 
                                day.

                                Default is 'D'

            incldue_history -   [bool] A boolean describing whether to include 
                                history observations or not used by the fit method.

                                Default is True
            

            
            Returns:
            ------------
            predictions -       [DataFrame] A dataframe with:
                                [time_col] -    The time column fed into this method
                                [target_col] -  The name for the original target 
                                                column if history is included in the dataframe
                                yhat -          The predicted value for observations
                                yhat_upper -    The upper value for uncertainty 
                                                + credibility interval
                                yhat_lower -    The lower value for uncertainty 
                                                + credibility interval
                                trend -         The predicted value for the trend, 
                                                excluding seasonality
                                trend_upper -   The upper value for trend uncertainty
                                trend_lower -   The lower value for trend uncertainty

                                Seasonality is not returned in the dataframe, but is 
                                incorporated when computing yhat.
            
        '''
        
        

        self.__prediction_verbose = verbose
        # Make a future dataframe if one is not provided
        if (future_df is None):            
            future_df = self.make_future_dataframe(period=period, 
                                                   frequency=frequency, 
                                                   include_history=include_history)
                                                   
        self.__check_incoming_data_for_nulls(future_df, predictions=True)

        # Transform data into trend and seasonality as before
        X_time, X_dataframe, y = self.__transform_data(future_df)
        

        # For some reason the predictive method runs for 2 extra runs
        # so we need to set the counter to -2 to tell the predictive method
        # when to start printing out output
        self.predict_counter_ = -2

        self.__trend_components = {}
        self.__multiplicative_components = {}
        self.__additive_components = {}

        # For point estimates, use the predictive interface
        if (self.__method in chronos_utils.SUPPORTED_METHODS):
            # https://pyro.ai/examples/bayesian_regression.html#Model-Evaluation
            predictive = Predictive(model=self.__model,
                                    guide=self.__guide,
                                    num_samples=sample_number,
                                    return_sites=("obs", "trend")) 

            
            samples = predictive(X_time, 
                                 X_dataframe)
            
            
            
            # Calculate ntiles based on the CI provided. Each side should have
            # CI/2 credibility
            space_on_each_side = (1.0 - ci_interval)/2.0
            lower_ntile = int(len(samples['obs']) * space_on_each_side)
            upper_ntile = int(len(samples['obs']) * (1.0 - space_on_each_side))
            
            # The resulting tensor returns with (sample_number, 1, n_samples) shape
            trend_array = samples['trend'].squeeze()

            # Calculate uncertainty
            trend = trend_array.mean(dim=0)
            trend_upper = trend_array.max(dim=0).values
            trend_lower = trend_array.min(dim=0).values

            # Build the output dataframe
            predictions = pd.DataFrame({"yhat": torch.mean(samples['obs'], 0).detach().numpy(),
                                        "yhat_lower": samples['obs'].kthvalue(lower_ntile, dim=0)[0].detach().numpy(),
                                        "yhat_upper": samples['obs'].kthvalue(upper_ntile, dim=0)[0].detach().numpy(),
                                        "trend": trend.detach().numpy(),
                                        "trend_lower": trend_lower.detach().numpy(),
                                        "trend_upper": trend_upper.detach().numpy()})
            
            
            # Incorporate the original values, and build the column order to return
            columns_to_return = []
            
            columns_to_return.append(self.__time_col)
            predictions[self.__time_col] = future_df[self.__time_col]

            if (y is not None):
                predictions[self.__target_col] = y.detach().numpy()
                columns_to_return.append(self.__target_col)

            columns_to_return.extend(['yhat', 'yhat_upper', 'yhat_lower', 
                                      'trend', 'trend_upper', 'trend_lower'])


            predictions = predictions[columns_to_return]
            numeric_columns = columns_to_return[1:]

            if (self.__y_max is not None):
                predictions[numeric_columns] *= self.__y_max
            return predictions
        else:
            raise NotImplementedError(f"Did not implement .predict for {self.__method}")

        
        
        
    ########################################################################################################################
    def make_future_dataframe(self, period=30, frequency="D", include_history=True):
        '''
            A function which takes in a future range specified by the period and the
            frequency and returns a dataframe which can be used by the predict method.
            By default, the history is included as well for easy diagnostics via the 
            plotting  methods.

            NOTE: You should only use this method if you plan on adding additional 
            custom regressors. If there are no custom regressors involved you can 
            use the .predict method directly

            Parameters:
            ------------
            period -            [int] The number of future observations based on
                                frequency. The default is 30, and the default
                                for frequency id 'D' which means 30 days ahead.

                                Default is 30
            
            frequency -         [str] The frequency of the period. See 
                                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
                                for a list of supported freuqncies.
                                The default is 'D' which stands for calendar 
                                day.

                                Default is 'D'

            incldue_history -   [bool] A boolean describing whether to include history
                                observations or not used by the fit method.

                                Default is True

            
            Returns:
            ------------
            future_df -         [DataFrame] A dataframe with a datestamp column, and a
                                target column ready to be used by the 
                                predict method. The datestamp and target
                                column names are the same as the ones
                                used in the fitting dataframe.
            
        '''
        # Find the highest timestamp observed, and build a new series of timestamps starting
        # at that timestamp, then remove the first of that series. The resulting date range
        # will begin one [frequency] ahead of the last datestamp in history (for example, 
        # one day after, or one hour after)
        max_date_observed = self.history[self.__time_col].max()

        
        date_range = pd.date_range(start=str(max_date_observed), periods=period+1, freq=frequency)
        date_range = date_range[1:]
        
        # Package everything into a dataframe
        future_df = pd.DataFrame({self.__time_col: date_range,
                                  self.__target_col: [np.nan] * date_range.shape[0]})
        
        # Optionally add the history
        if (include_history == True):
            past_df = self.history.copy()
            future_df = pd.concat([past_df, future_df], axis=0).reset_index(drop=True)

        return future_df
    
    
    ########################################################################################################################
    def __compute_seasonality(self, param_pairs, numeric_values, cycle_period):
        '''
            A function which accepts a tensor of coefficients in param pairs, the
            time values and the cycle period, and returns a seasonal (cyclical)
            tensor of the required seasonality using pairs of sin and cos 
            calculations.


            Parameters:
            ------------
            param_pairs -       [tensor] A tensor of parameter pairs. The tensor should
                                always be of shape (N, 2) where N is the order
                                of the given seasonality.

            numeric_values -    [tensor] A tensor of the time values to be calculated. 
                                This can be weekdays, which will range from 0-6, 
                                month days which will range from 0-30 etc' 

            cycle_periods -     [tensor] The period of the cycle. For weekdays, the cycle
                                repeats every 7 days, so the period will be
                                7. For months, it will be 31. For years, 366, etc'
            

            
            Returns:
            ------------
            seasonality -       [tensor] The seasonal component specified by
                                the input parameters. e.g. weekly seasonality,
                                or yearly seasonality.
        '''

        seasonality = np.zeros_like(numeric_values, dtype=np.float32)

        # Go through each parameter pair and apply it to
        # a pair of sin and cos functions defining the
        # seasonality.
        for i, pair in enumerate(param_pairs):
            cycle_order = i+1

            sin_coef = pair[0]
            cosin_coef = pair[1]
            
            cycle_pos = cycle_order * 2 * np.pi * numeric_values/cycle_period
            seasonality += (sin_coef * np.sin(cycle_pos)) + (cosin_coef * np.cos(cycle_pos))

        return seasonality

    ########################################################################################################################
    def get_seasonality(self, seasonality_name):
        '''
            A function which returns a tensor denoting the seasonality requested. If a 
            method not in [MAP, MLE] is requested, the function throws an error

            Parameters:
            ------------
            seasonality_name -  [str] A string denoting the name of the seasonality
                                requested
            
            
            Returns:
            ------------
            seasonality -       [DataFrame] A pandas dataframe of the requested 
                                seasonality
            
        '''
        if (self.__method in chronos_utils.POINT_ESTIMATE_METHODS):
            if (seasonality_name == "weekly"):
                seasonality = self.__get_weekly_seasonality_point(f'{self.__param_prefix}betas')
            elif (seasonality_name == "monthly"):
                seasonality = self.__get_monthly_seasonality_point(f'{self.__param_prefix}betas')
            elif (seasonality_name == "yearly"):
                seasonality = self.__get_yearly_seasonality_point(f'{self.__param_prefix}betas')

            if (self.__seasonality_mode == "add"):
                if (self.__y_max is not None):
                    seasonality['Y'] *= self.__y_max
            elif (self.__seasonality_mode == "mul"):
                seasonality['Y'] = 100*(1.0 + seasonality['Y'])
            return seasonality
        elif (self.__method in chronos_utils.DISTRIBUTION_ESTIMATE_METHODS):
            if (seasonality_name == "weekly"):
                seasonality = self.__get_weekly_seasonality_dist(f'{self.__param_prefix}')
            elif (seasonality_name == "monthly"):
                seasonality = self.__get_monthly_seasonality_dist(f'{self.__param_prefix}')
            elif (seasonality_name == "yearly"):
                seasonality = self.__get_yearly_seasonality_dist(f'{self.__param_prefix}')
            

            if (self.__seasonality_mode == "add"):
                if (self.__y_max is not None):
                    seasonality['Y'] *= self.__y_max
            elif (self.__seasonality_mode == "mul"):
                seasonality['Y'] = 100*(1.0 + seasonality['Y'])
            return seasonality
            
        else:
            raise NotImplementedError("Did not implement weekly seasonality for non MAP non MLE")
    ########################################################################################################################
    def __get_seasonal_params(self, param_name):
        '''
            A function which accepts the name of the parameter store where
            seasonality coefficients are stored, and the seasonality name,
            and returns the coefficients corresponding to the requested
            seasonality.

            Parameters:
            ------------
            param_name -            [str] The name of the global param store where
                                    the specific seasonality is stored.  
            
            Returns:
            ------------
            seasonality_params -    [tensor] A tensor of shape (N,2) where N
                                    is the order of the requested 
                                    seasonality. This tensor contains
                                    the seasonality coefficients.
            
        '''

        seasonal_params = []

        for param in pyro.param(param_name):
            seasonal_params.append(param.item())

        # Reshape to have two columns. The parameters are assumed to be
        # in order (sin_param1, cos_param1, sin_param2, cos_param2, ...) 
        seasonal_params = np.array(seasonal_params).reshape(-1, 2)

        return seasonal_params
    
    ########################################################################################################################
    def __get_weekly_seasonality_point(self, param_name):
        '''
            A function which accepts the name of the parameter where point estimates
            of seasonalities are stored and returns a pandas dataframe containing
            the data for the weekly seasonality as well axis labels

            Parameters:
            ------------
            param_name -            [str] The name of the pyro parameter store where the 
                                    point estimates are stored

            
            Returns:
            ------------
            weekly_seasonality -    [DataFrame] A pandas dataframe containing three 
                                    columns:
                                    
                                    X -     The values for the weekly seasonality (0-6)
                                    Label - The labels for the days ("Monday" - "Sunday")
                                    Y -     The seasonal response for each day
        '''

        # Get the parameter pairs of coefficients
        weekly_params = self.__get_seasonal_params(param_name+"_weekly")
        
        # Monday is assumed to be 0
        weekdays_numeric = np.arange(0, 7, 1)
        weekdays = chronos_utils.weekday_names_
        if (self.__trained_on_weekend == False):
            weekdays_numeric = weekdays_numeric[:-2]
            weekdays = weekdays[:-2]

        # Compute seasonal response
        seasonality = self.__compute_seasonality(weekly_params, weekdays_numeric, weekdays_numeric.shape[0])
            
        # Package everything nicely into a df
        weekly_seasonality = pd.DataFrame({"X": weekdays_numeric,
                                           "Label": weekdays,
                                           "Y": seasonality})
        
        return weekly_seasonality
    ########################################################################################################################
    def __get_weekly_seasonality_dist(self, param_name):
        '''
            TODO: update description
            A function which accepts the name of the parameter where point estimates
            of seasonalities are stored and returns a pandas dataframe containing
            the data for the weekly seasonality as well axis labels

            Parameters:
            ------------
            param_name -            [str] The name of the pyro parameter store where the 
                                    point estimates are stored

            
            Returns:
            ------------
            weekly_seasonality -    [DataFrame] A pandas dataframe containing three 
                                    columns:
                                    
                                    X -     The values for the weekly seasonality (0-6)
                                    Label - The labels for the days ("Monday" - "Sunday")
                                    Y -     The seasonal response for each day
        '''

        
        # Get the parameter pairs of coefficients
        weekly_params_loc = torch.tensor(self.__get_seasonal_params(param_name+"locs.betas_weekly"))
        weekly_params_scales = torch.tensor(self.__get_seasonal_params(param_name+"scales.betas_weekly"))
        
        #print(weekly_params_loc)
        #print(weekly_params_scales)

        weekly_param_sampled_values = dist.Normal(weekly_params_loc, 
                                                  weekly_params_scales).sample((1000,)).detach().numpy()
        

        
        
        # Monday is assumed to be 0
        weekdays_numeric = np.arange(0, 7, 1)
        weekdays = chronos_utils.weekday_names_
        if (self.__trained_on_weekend == False):
            weekdays_numeric = weekdays_numeric[:-2]
            weekdays = weekdays[:-2]

        # Compute seasonal response
        seasonality_array = np.zeros((weekly_param_sampled_values.shape[0], 
                                      weekdays_numeric.shape[0]))

        for i in range(weekly_param_sampled_values.shape[0]):
            seasonality_array[i, :] = self.__compute_seasonality(weekly_param_sampled_values[i], 
                                                                 weekdays_numeric, 
                                                                 weekdays_numeric.shape[0])
        seasonality = seasonality_array.mean(axis=0)
        seasonality_max = seasonality_array.max(axis=0)
        seasonality_min = seasonality_array.min(axis=0)

        #print(seasonality)
        # Package everything nicely into a df
        weekly_seasonality = pd.DataFrame({"X": weekdays_numeric,
                                           "Label": weekdays,
                                           "Y": seasonality})
        
        return weekly_seasonality
    ########################################################################################################################

    def __get_monthly_seasonality_point(self, param_name):
        '''
            A function which accepts the name of the parameter where point estimates
            of seasonalities are stored and returns a pandas dataframe containing
            the data for the monthly seasonality as well axis labels

            Parameters:
            ------------
            param_name -            [str] The name of the pyro parameter store where the 
                                    point estimates are stored

            
            Returns:
            ------------
            weekly_seasonality -    [DataFrame] A pandas dataframe containing three 
                                    columns:
                                    
                                    X -     The values for the monthly seasonality (0-30)
                                    Label - The labels for the days ("1st" - "31st")
                                    Y -     The seasonal response for each day
        '''

        # Get the parameter pairs of coefficients
        monthly_params = self.__get_seasonal_params(param_name+"_monthly")
        
        monthdays_numeric = np.arange(0, 31, 1)
        monthday_names = chronos_utils.monthday_names_

        # Compute seasonal response
        seasonality = self.__compute_seasonality(monthly_params, monthdays_numeric, 31)
            
        # Package everything nicely into a df
        monthly_seasonality = pd.DataFrame({"X": monthdays_numeric,
                                            "Label": monthday_names,
                                            "Y": seasonality})
        
        return monthly_seasonality
        ########################################################################################################################

    def __get_monthly_seasonality_dist(self, param_name):
        '''
            TODO: update
            A function which accepts the name of the parameter where point estimates
            of seasonalities are stored and returns a pandas dataframe containing
            the data for the monthly seasonality as well axis labels

            Parameters:
            ------------
            param_name -            [str] The name of the pyro parameter store where the 
                                    point estimates are stored

            
            Returns:
            ------------
            weekly_seasonality -    [DataFrame] A pandas dataframe containing three 
                                    columns:
                                    
                                    X -     The values for the monthly seasonality (0-30)
                                    Label - The labels for the days ("1st" - "31st")
                                    Y -     The seasonal response for each day
        '''

        # Get the parameter pairs of coefficients
        monthly_params_loc = torch.tensor(self.__get_seasonal_params(param_name+"locs.betas_monthly"))
        monthly_params_scales = torch.tensor(self.__get_seasonal_params(param_name+"scales.betas_monthly"))


        monthly_param_sampled_values = dist.Normal(monthly_params_loc, 
                                                   monthly_params_scales).sample((1000,)).detach().numpy()

        
        monthdays_numeric = np.arange(0, 31, 1)
        monthday_names = chronos_utils.monthday_names_

        # Compute seasonal response
        seasonality_array = np.zeros((monthly_param_sampled_values.shape[0], 
                                      monthdays_numeric.shape[0]))

        for i in range(monthly_param_sampled_values.shape[0]):
            seasonality_array[i, :] = self.__compute_seasonality(monthly_param_sampled_values[i], 
                                                                 monthdays_numeric, 
                                                                 monthdays_numeric.shape[0])

        seasonality = seasonality_array.mean(axis=0)
        seasonality_max = seasonality_array.max(axis=0)
        seasonality_min = seasonality_array.min(axis=0)
            
        # Package everything nicely into a df
        monthly_seasonality = pd.DataFrame({"X": monthdays_numeric,
                                            "Label": monthday_names,
                                            "Y": seasonality})
        
        return monthly_seasonality

    ########################################################################################################################
    def __get_yearly_seasonality_point(self, param_name):
        '''
            A function which accepts the name of the parameter where point estimates
            of seasonalities are stored and returns a pandas dataframe containing
            the data for the yearly seasonality as well axis labels

            Parameters:
            ------------
            param_name -            [str] The name of the pyro parameter store where the 
                                    point estimates are stored

            
            Returns:
            ------------
            weekly_seasonality -    [DataFrame] A pandas dataframe containing three 
                                    columns:
                                    
                                    X -     The values for the yearly seasonality 
                                            days (0-366)
                                    Label - The labels for the days (the individual dates)
                                    Y -     The seasonal response for each day
        '''
        
        # Get the parameter pairs of coefficients
        yearly_params = self.__get_seasonal_params(param_name+"_yearly")
        
        yeardays_numeric = np.arange(0, 366, 1)
        yearly_dates = pd.date_range(start="01-01-2020", periods=366) # Use a leap year to include Feb 29th

        # Compute seasonal response
        seasonality = self.__compute_seasonality(yearly_params, yeardays_numeric, 366)
            
        # Package everything nicely into a df
        yearly_seasonality = pd.DataFrame({"X": yearly_dates,
                                           "Label": yearly_dates,
                                           "Y": seasonality})
        
        return yearly_seasonality
    
    ########################################################################################################################
    @property
    def changepoints_values(self):
        '''
            A property which return the changepoint values
            of the history
        '''
        
        # data is in AutoNormal.locs.delta and AutoNormal.scales.data
        if (self.__method in chronos_utils.POINT_ESTIMATE_METHODS):
            past_deltas = pyro.param(f"{self.__param_prefix}delta")
        else:
            changepoint_values_distribution  = dist.Normal(pyro.param(f"{self.__param_prefix}locs.delta"), 
                                                           pyro.param(f"{self.__param_prefix}scales.delta"))
            past_deltas = changepoint_values_distribution.sample((1000, )).mean(dim=0)
            #print(dir(self.__guide))
            
        return past_deltas
    ########################################################################################################################
    @property
    def changepoints_positions(self):
        '''
            A property which return the changepoint positions (timestamps)
            of the history
        '''
        positions = (self.__changepoints * (self.__history_max_time_seconds - self.__history_min_time_seconds) + self.__history_min_time_seconds)
        return positions
    ########################################################################################################################
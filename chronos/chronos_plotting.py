# Copyright (c) Boris Shabash

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import datetime



#import plotly
#import plotly.graph_objs as go


history_color_ = "black"
prediction_color_ = "blue"
uncertainty_color_ = "blue"
underperdiction_color_ = "darkblue"
overprediction_color_ = "lightblue"

########################################################################################################################
def plot_components(predictions, chronos_object=None, changepoint_threshold = 0.0, figure_name = None, figsize=(15,15)):
    '''
        TODO: update
        A function to plot all components of the predictions into a matplotlib figure. 
        The resulting figure will have the raw predictions, the trend, and the residuals
        (the error terms) from the prediction process. Seasonalities are plottted as well
        if chronos_object is provided.


        Parameters:
        ------------
        predictions -           [DataFrame] A pandas dataframe returned from the .predict 
                                method of  this Chronos object. The dataframe is expected
                                to have the same columns as the dataframe used for fitting
                                as well as the following columns: 
                                yhat, yhat_upper, yhat_lower, trend, trend_upper, 
                                trend_lower

        chronos_object -        [Chronos] A fitted chronos object that was used to 
                                generate the predictions dataframe. If this object 
                                is not provided seasonalities are not plotted

                                Default is None

        changepoint_threshold - [float] The threshold for the changepoints to be marked on
                                the plot. Must be a non-negative

                                Default is 0.0

        figure_name -           [str] An optional parameter for the figure name for it to
                                be saved. E.g. "myfig.png"

                                Default is None
        
        figsize -               [tuple] The figure size to use

                                Default is (15,15)
        
        Returns:
        ------------
        fig -                   [Figure] The figure object on which all the plotting takes
                                place
        
    '''
    
    # Count how many plots we will need
    plot_num = 4 # 2 rows for predictions, 1 for trend, and 1 for residuals
    if (chronos_object is None):
        print("Chronos object not provided, seasonalities will not be plotted")
    else:
        if (chronos_object._Chronos__weekly_seasonality_order > 0):
            plot_num += 1
        if (chronos_object._Chronos__month_seasonality_order > 0):
            plot_num += 1
        if (chronos_object._Chronos__year_seasonality_order > 0):
            plot_num += 1
    


    
    
    fig = plt.figure(tight_layout=True, figsize=figsize)
    gs = gridspec.GridSpec(plot_num, 1)

    current_axs = 2
    # Plot the predictions
    ax = fig.add_subplot(gs[:current_axs, :])
    plot_predictions(predictions, chronos_object, axs=ax)

    
    # Plot the trend
    ax = fig.add_subplot(gs[current_axs, :])
    plot_trend(predictions, chronos_object, changepoint_threshold = changepoint_threshold, axs=ax)
    
    current_axs += 1

    if (chronos_object is not None):
        # Plot all seasonalities. Each into an individual subplot
        if (chronos_object._Chronos__weekly_seasonality_order > 0):
            ax = fig.add_subplot(gs[current_axs, : ])
            plot_weekly_seasonality(chronos_object, axs=ax)
            current_axs += 1

        if (chronos_object._Chronos__month_seasonality_order > 0):
            ax = fig.add_subplot(gs[current_axs, : ])
            plot_monthly_seasonality(chronos_object, axs=ax)
            current_axs += 1

        if (chronos_object._Chronos__year_seasonality_order > 0):
            ax = fig.add_subplot(gs[current_axs, : ])
            plot_yearly_seasonality(chronos_object, axs=ax)
            current_axs += 1


    # Finally, plot the residuals      
    ax = fig.add_subplot(gs[current_axs, :])       
    plot_residuals(predictions, chronos_object, axs=ax)


    # Adjust the spacing between the subplots so differnet text components 
    # do not overwrite each other
    plt.subplots_adjust(hspace=1.0)

    # Optionally save the figure
    if (figure_name is not None):
        plt.savefig(figure_name, dpi=96*2)#'''
    else:
        plt.show()

    return fig
########################################################################################################################
def plot_predictions(predictions, chronos_object=None, axs=None):
    '''
        A function which produces a plot of the predictions as well
        as historical data, if included in the predictions dataframe.
        An optional axis can be passed in for the drawing to take place on in case a 
        subplot is used. If no matplotlib axis is passed in, the function draws the 
        resulting figure and returns it

        Parameters:
        ------------
        predictions -           [DataFrame] A pandas dataframe returned from the .predict 
                                method of  this Chronos object. The dataframe is expected
                                to have the same columns as the dataframe used for fitting
                                as well as the following columns: 
                                yhat, yhat_upper, yhat_lower, trend, trend_upper, 
                                trend_lower

        chronos_object -        [Chronos] A fitted chronos object that was used to 
                                generate the predictions dataframe. If this object 
                                is not provided assumptions are made about the 
                                naming in the predictions dataframe

                                Default is None

        axs -                   [Axis] The axis on which to perform the plotting of the
                                matplotlib plot. If no axis is provided, a new figure
                                is created and returned

                                Default is None
        
        Returns:
        ------------
        fig -                   [Figure] The figure object on which all the plotting takes
                                place. Only returned if no axis is provided to axs
    '''
    if (chronos_object is None):
        print("No Chronos object provided, assuming timestamp and target columns are named 'ds' and 'y'.")
        time_col = "ds"
        target_col = "y"
    else:
        time_col = chronos_object._Chronos__time_col
        target_col = chronos_object._Chronos__target_col

    single_figure = False
    if (axs is None):
        single_figure = True

        fig = plt.figure(figsize=(15,5))
        gs = gridspec.GridSpec(1,1, figure=fig)
        axs = fig.add_subplot(gs[0, 0])

    

    # plot credibility intervals
    axs.fill_between(predictions[time_col], 
                     predictions['yhat_upper'], 
                     predictions['yhat_lower'], 
                     color=uncertainty_color_, 
                     alpha=0.3)

    # plot true data points, but only if there is at least one non-nan value
    if (predictions[target_col].isna().sum() < predictions.shape[0]):
        axs.scatter(predictions[time_col], 
                    predictions[target_col], 
                    c=history_color_, 
                    label="observed")

    # plot predictions
    axs.plot(predictions[time_col], 
             predictions['yhat'], 
             c=prediction_color_, 
             label="predictions", 
             linewidth=3)
    
    
    # Set up plot
    axs.set_xlim(predictions[time_col].min(), predictions[time_col].max())
    axs.set_xlabel("Date", size=16)
    axs.set_ylabel("Values", size=16)
    axs.set_title('Predictions', size=18)

    if (single_figure):
        plt.show()
        return fig
########################################################################################################################
#def plot_trend(self, x, trend, changepoint_threshold=0.0, trend_upper=None, trend_lower=None, predictions_start_date = None, axs=None):
def plot_trend(predictions, chronos_object=None, changepoint_threshold=0.0, axs=None):
    '''
        A function which plots the trend components, along with historical changepoints. 
        An optional axis can be passed in for the drawing to take place on in case a 
        subplot is used. If no matplotlib axis is passed in, the function draws the 
        resulting figure and returns it


        Parameters:
        ------------
        predictions -           [DataFrame] A pandas dataframe returned from the .predict 
                                method of  this Chronos object. The dataframe is expected
                                to have the same columns as the dataframe used for fitting
                                as well as the following columns: 
                                yhat, yhat_upper, yhat_lower, trend, trend_upper, 
                                trend_lower

        chronos_object -        [Chronos] A fitted chronos object that was used to 
                                generate the predictions dataframe. If this object 
                                is not provided assumptions are made about the 
                                naming in the predictions dataframe, and changepoints
                                are not labeled on the plot.

                                Default is None

        changepoint_threshold - [float] The threshold for the changepoints to be marked on
                                the plot. Must be a non-negative

                                Default is 0.0

        axs -                   [Axis] The axis on which to perform the plotting of the
                                matplotlib plot. If no axis is provided, a new figure
                                is created and returned

                                Default is None
        
        Returns:
        ------------
        fig -                   [Figure] The figure object on which all the plotting takes
                                place. Only returned if no axis is provided to axs

    '''

    # Set everything up
    if (chronos_object is None):
        print("No Chronos object provided, assuming timestamp and target columns are named 'ds' and 'y'.")
        print("Additionally, changepoints will not be labeled on the plot")
        time_col = "ds"        
        changepoint_values = []
        changepoint_positions = []
    else:
        # If we have a chronos object, grab the relevant data
        time_col = chronos_object._Chronos__time_col
        changepoint_values = chronos_object.changepoints_values.detach().numpy()
        changepoint_positions = chronos_object.changepoints_positions.detach().numpy()

    # If no axis is passed in, make sure to create a figure
    single_figure = False
    if (axs is None):
        single_figure = True

        fig = plt.figure(figsize=(15,5))
        gs = gridspec.GridSpec(1,1, figure=fig)
        axs = fig.add_subplot(gs[0,0])



    # Plot the predictions
    axs.plot(predictions[time_col], 
             predictions['trend'], 
             linewidth=3, 
             c=prediction_color_)

    # Set up the limits for the x axis
    axs.set_xlim(predictions[time_col].min(), predictions[time_col].max())

    # Optionally draw uncertainty trend
    axs.fill_between(predictions[time_col], 
                    predictions['trend_upper'], 
                    predictions['trend_lower'], 
                    color=prediction_color_, 
                    alpha=0.3)

    
    # Draw changepoints if they are present
    for index, changepoint_value in enumerate(changepoint_values):
        
        # Need to inverse transform every changepoint to the date it represents
        changepoint_second = int(changepoint_positions[index])

        # Now we can make it into a date to use it in our X-axis, which is date based
        changepoint_date_value = datetime.datetime.fromtimestamp(changepoint_second)
        

        if (abs(changepoint_value) >= changepoint_threshold):
            axs.axvline(changepoint_date_value, c="black", linestyle="dotted")#'''
    
    # Optionally mark where history ends and predictions begin
    if (chronos_object is not None):
        predictions_start_date = datetime.datetime.fromtimestamp(chronos_object._Chronos__history_max_time_seconds)
    
        axs.axvline(datetime.date(predictions_start_date.year, 
                                  predictions_start_date.month, 
                                  predictions_start_date.day), c="black", linestyle="--")

    
    # Make sure to add proper labels
    axs.set_xlabel('Date', size=18)
    axs.set_ylabel('Growth', size=18)
    axs.set_title('Trend', size=18)

    if (single_figure):
        plt.show()
        return fig

########################################################################################################################
def plot_weekly_seasonality(chronos_object, axs=None):
    '''
        A function which plots the weekly seasonality. An optional axis can be passed in
        for the drawing to take place on in case a subplot is used. If no matplotlib axis
        is passed in, the function draws the resulting figure and returns it


        Parameters:
        ------------
        chronos_object -    [Chronos] A fitted chronos object that was used to 
                            generate the predictions dataframe. 

        axs -               [Axis] Optional argument, a matplotlib subplot axis for 
                            the  drawing to take place on. If no axis is passed in 
                            as an argument, a new figure is created and returned 
                            upon drawing.

                            Default is None

        
        Returns:
        ------------
        fig -               [Figure] An optional return value. If no axis is passed
                            in as an argument, a new figure is created and returned
                            upon drawing.
    '''

    # Grab the weekly seasonality
    weekly_seasonality = chronos_object.get_seasonality("weekly")

    single_figure = False
    # Make sure to build a new figure if we don't have one
    if (axs is None):
        single_figure = True
        fig, axs = plt.subplots(1, 1, figsize=(15, 5))
    
    axs.plot(weekly_seasonality['X'], weekly_seasonality['Y'], linewidth=3, c=prediction_color_)

    # If we have additive seasonality we will positive and negative values
    # but if it's multiplicative, we will only have positive values and
    # won't be a good idea to include the 0 line, otherwise it warps
    # the scale
    if (chronos_object._Chronos__seasonality_mode == "add"):
        axs.axhline(0.0, c="black", linestyle="--")
    else:
        axs.axhline(100.0, c="black", linestyle="--")
        axs.yaxis.set_major_formatter(mtick.PercentFormatter())
    axs.set_xticks(weekly_seasonality['X'].values)
    axs.set_xticklabels(weekly_seasonality['Label'].values)
    axs.set_xlim(-0.1, weekly_seasonality['X'].max()+0.1)
    axs.set_xlabel('Weekday', size=18)
    axs.set_ylabel('Seasonality', size=18)
    axs.set_title('Weekly Seasonality', size=18)

    if (single_figure):
        plt.show()
        return fig
########################################################################################################################
def plot_monthly_seasonality(chronos_object, axs=None):
    '''
        A function which plots the monthly seasonality. An optional axis can be passed in
        for the drawing to take place on in case a subplot is used. If no matplotlib axis
        is passed in, the function draws the resulting figure and returns it


        Parameters:
        ------------
        chronos_object -    [Chronos] A fitted chronos object that was used to 
                            generate the predictions dataframe. 

        axs -               [Axis] Optional argument, a matplotlib subplot axis for the 
                            drawing to take place on. If no axis is passed in as an 
                            argument, a new figure is created and returned upon drawing.

                            Default is None

        
        Returns:
        ------------
        fig -               [Figure] An optional return value. If no axis is passed in 
                            as an argument, a new figure is created and returned upon 
                            drawing.
    '''
    # Grab the monthly seasonality
    monthly_seasonality = chronos_object.get_seasonality("monthly")


    single_figure = False
    # Make sure to build a new figure if we don't have one
    if (axs is None):
        single_figure = True
        fig, axs = plt.subplots(1, 1, figsize=(15, 5))

    axs.plot(monthly_seasonality['X'], monthly_seasonality['Y'], linewidth=3, c=prediction_color_)
    
    # If we have additive seasonality we will positive and negative values
    # but if it's multiplicative, we will only have positive values and
    # won't be a good idea to include the 0 line, otherwise it warps
    # the scale
    if (chronos_object._Chronos__seasonality_mode == "add"):
        axs.axhline(0.0, c="black", linestyle="--")
    else:
        axs.axhline(100.0, c="black", linestyle="--")
        axs.yaxis.set_major_formatter(mtick.PercentFormatter())
    axs.set_xticks(monthly_seasonality['X'].values[::9])
    axs.set_xticklabels(monthly_seasonality['Label'].values[::9])
    axs.set_xlim(-0.2, 30.2)
    axs.set_xlabel('Day of Month', size=18)
    axs.set_ylabel('Seasonality', size=18)
    axs.set_title("Monthly Seasonality", size=18)

    if (single_figure):
        plt.show()
        return fig

########################################################################################################################
def plot_yearly_seasonality(chronos_object, axs=None):
    '''
        A function which plots the yearly seasonality. An optional axis can be passed in
        for the drawing to take place on in case a subplot is used. If no matplotlib axis
        is passed in, the function draws the resulting figure and returns it


        Parameters:
        ------------
        chronos_object -    [Chronos] A fitted chronos object that was used to 
                            generate the predictions dataframe. 

        axs -               [Axis] Optional argument, a matplotlib subplot axis for the 
                            drawing to take place on. If no axis is passed in as an 
                            argument, a new figure is created and returned upon drawing.

                            Default is None

        
        Returns:
        ------------
        fig -               [Figure] An optional return value. If no axis is passed in 
                            as an argument, a new figure is created and returned upon 
                            drawing.
    '''
    # Grab the yearly seasonality
    yearly_seasonality = chronos_object.get_seasonality("yearly")

    single_figure = False
    # Make sure to build a new figure if we don't have one
    if (axs is None):
        single_figure = True
        fig, axs = plt.subplots(1, 1, figsize=(15, 5))

    axs.plot(yearly_seasonality['X'], yearly_seasonality['Y'], linewidth=3, c=prediction_color_)

    # If we have additive seasonality we will positive and negative values
    # but if it's multiplicative, we will only have positive values and
    # won't be a good idea to include the 0 line, otherwise it warps
    # the scale
    if (chronos_object._Chronos__seasonality_mode == "add"):
        axs.axhline(0.0, c="black", linestyle="--")
    else:
        axs.axhline(100.0, c="black", linestyle="--")
        axs.yaxis.set_major_formatter(mtick.PercentFormatter())
    axs.set_xlim(datetime.date(2019, 12, 31), datetime.date(2021, 1, 2))
    axs.set_xlabel('Day of Year', size=18)
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    axs.set_ylabel('Seasonality', size=18)
    axs.set_title('Yearly Seasonality', size=18)

    if (single_figure):
        plt.show()
        return fig

########################################################################################################################
def plot_residuals(predictions, chronos_object=None, axs=None):
    '''
        A function which plots the residuals of the fit. An optional axis can be passed in
        for the drawing to take place on in case a subplot is used. If no matplotlib axis
        is passed in, the function draws the resulting figure and returns it


        Parameters:
        ------------
        predictions -       [DataFrame] A pandas dataframe returned from the .predict 
                            method of  this Chronos object. The dataframe is expected
                            to have the same columns as the dataframe used for fitting
                            as well as the following columns: 
                            yhat, yhat_upper, yhat_lower, trend, trend_upper, 
                            trend_lower

        chronos_object -    [Chronos] A fitted chronos object that was used to 
                            generate the predictions dataframe. If this object 
                            is not provided assumptions are made about the 
                            naming in the predictions dataframe.

                            Default is None

        axs -               [Axis] Optional argument, a matplotlib subplot axis for the 
                            drawing to take place on. If no axis is passed in as an 
                            argument, a new figure is created and returned upon drawing.

                            Default is None

        
        Returns:
        ------------
        fig -               [Figure] An optional return value. If no axis is passed in 
                            as an argument, a new figure is created and returned upon 
                            drawing.
    '''
    if (chronos_object is None):
        print("No Chronos object provided, assuming timestamp and target columns are named 'ds' and 'y'.")
        time_col = "ds"
        target_col = "y"
    else:
        time_col = chronos_object._Chronos__time_col
        target_col = chronos_object._Chronos__target_col


    x = predictions[time_col]
    y_true = predictions[target_col]
    y_pred = predictions['yhat']

    y_residuals = y_true - y_pred
    

    single_figure = False
    # Make sure to build a new figure if we don't have one
    if (axs is None):
        single_figure = True
        fig = plt.figure(figsize=(15,5))
        gs = gridspec.GridSpec(1, 1, figure=fig)
        gs_section = gs[0, 0]
    
        axs = fig.add_subplot(gs[0, 0])


    # Make a fake plot just so the Residuals title is centered 
    axs.xaxis.set_ticks_position('none') 
    axs.yaxis.set_ticks_position('none') 
    axs.set_xlabel(' \n\n ', size=18)
    axs.set_title('Residuals', size=18)
    axs.tick_params(labelbottom=False, labelleft=False)

    
    
    # Make a sub-axis to include a scatter plot
    scatter_axis = inset_axes(axs, width="88%", height="100%", loc=6, borderpad=0.0)

    # Only plot the observations where there are residuals
    x_to_plot = x[~np.isnan(y_residuals)]
    y_residuals_to_plot = y_residuals[~np.isnan(y_residuals)]

    scatter_axis.scatter(x_to_plot, y_residuals_to_plot, c=prediction_color_, s=4)
    scatter_axis.set_xlim(x_to_plot.min(), x_to_plot.max())
    scatter_axis.set_xlabel('Date', size=18)
    scatter_axis.set_ylabel('Residuals', size=18)

    # Make another sub-axis to include a histogram
    hist_axis = inset_axes(axs, width="12%", height="100%", loc=7, borderpad=0.0)

    # Overprediction and underprediction will be in different colors so they are visually easy to tell apart
    hist_axis.hist(y_residuals[y_residuals <= 0.0], color=overprediction_color_, bins=50, orientation="horizontal")
    hist_axis.hist(y_residuals[y_residuals > 0.0], color=underperdiction_color_, bins=50, orientation="horizontal")

    # No need for tik marks on these axes
    hist_axis.xaxis.set_ticks_position('none') 
    hist_axis.yaxis.set_ticks_position('none') 
    hist_axis.set_yticklabels([])
    hist_axis.set_xticklabels([])
    hist_axis.tick_params(labelbottom=False, labelleft=False)

    
    if (single_figure):
        plt.show()
        return fig

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
# UNDER DEVELOPMENT
"""def get_seasonal_plotly_figure(self, seasonality_df):
    '''

        Parameters:
        ------------
        

        
        Returns:
        ------------
        
    '''

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

########################################################################################################################
def plot_weekly_seasonality_plotly(self):
    '''
        TODO: update
        Parameters:
        ------------
        

        
        Returns:
        ------------
        
    '''
    weekly_seasonality = self.get_weekly_seasonality()

    fig = self.get_seasonal_plotly_figure(weekly_seasonality)

    fig.update_layout(xaxis=dict(tickmode = 'array', 
                                    tickvals = weekly_seasonality['X'],
                                    ticktext = weekly_seasonality['Label']),
                        title=dict(text="Weekly Seasonality", x = 0.5))
    fig.show()




########################################################################################################################
def plot_monthly_seasonality_plotly(self):
    '''
        TODO: update

        Parameters:
        ------------
        

        
        Returns:
        ------------
        
    '''
    monthly_seasonality = self.get_monthly_seasonality()
    
    fig = self.get_seasonal_plotly_figure(monthly_seasonality)

    fig.update_layout(xaxis_range=[-0.2, 30.2])
    fig.update_layout(xaxis=dict(tickmode = 'array', 
                                    tickvals = monthly_seasonality['X'],
                                    ticktext = monthly_seasonality['Label']),
                        title=dict(text="Monthly Seasonality", x = 0.5))    
    fig.show()

    
########################################################################################################################
def plot_yearly_seasonality_plotly(self):
    '''
        TODO: update
        Parameters:
        ------------
        

        
        Returns:
        ------------
        
    '''
    yearly_seasonality = self.get_yearly_seasonality()

    
    fig = self.get_seasonal_plotly_figure(yearly_seasonality)
    
    fig.update_xaxes(dtick="M1", tickformat="%b", showgrid=False)
    # https://github.com/facebook/prophet/blob/20f590b7263b540eb5e7a116e03360066c58de4d/python/fbprophet/plot.py#L933        
    fig.update_layout(xaxis=go.layout.XAxis(tickformat = "%B %e", 
                                            type='date'), 
                        xaxis_range=["2019-12-30", "2021-01-02"],
                        title=dict(text="Yearly Seasonality", x = 0.5))
    
    fig.show()"""
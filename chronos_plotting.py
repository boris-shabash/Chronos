# -*- coding: utf-8 -*-
# Copyright (c) Boris Shabash

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import datetime



import plotly
import plotly.graph_objs as go


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
        The resulting figure will have the raw predictions, the trend, all different seasonalities,
        and the residuals (the error terms) from the prediction process.


        Parameters:
        ------------
        predictions -           A pandas dataframe returned from the .predict method of this Chronos object.
                                The dataframe is expected to have the same columns as the dataframe used for 
                                fitting as well as the following columns: 
                                yhat, yhat_upper, yhat_lower, trend, trend_upper, trend_lower

        changepoint_threshold - The threshold for the changepoints to be marked on the plot. Must be a non-negative

        figure_name -           An optional parameter for the figure name for it to be saved. E.g. "myfig.png"
        
        figsize -               The figure size to use
        
        Returns:
        ------------
        fig -                   The figure object on which all the plotting takes place
        
    '''
    
    # Count how many plots we will need
    plot_num = 4 # 2 rows for predictions, 1 for trend, and 1 for residuals
    if (chronos_object is None):
        print("Chronos object not provided, seasonalities will not be plotted")
    else:
        if (chronos_object.weekly_seasonality_order_ > 0):
            plot_num += 1
        if (chronos_object.month_seasonality_order_ > 0):
            plot_num += 1
        if (chronos_object.year_seasonality_order_ > 0):
            plot_num += 1
    


    
    
    fig = plt.figure(tight_layout=True, figsize=figsize)
    gs = gridspec.GridSpec(plot_num, 1)

    current_axs = 2
    # Plot the predictions
    plot_predictions(predictions, chronos_object, fig=fig, gs_section=gs[:current_axs, :])

    
    # Plot the trend
    plot_trend(predictions, chronos_object, changepoint_threshold = changepoint_threshold, fig=fig, gs_section=gs[current_axs, :])

    '''ax = fig.add_subplot(gs[current_axs, : ])
    plot_trend(x = predictions[self.time_col_], 
               trend = predictions['trend'], 
                    changepoint_threshold = changepoint_threshold, 
                    trend_upper = predictions['trend_upper'], 
                    trend_lower = predictions['trend_lower'], 
                    axs = ax)'''
    
    current_axs += 1

    if (chronos_object is not None):
        # Plot all seasonalities. Each into an individual subplot
        if (chronos_object.weekly_seasonality_order_ > 0):
            ax = fig.add_subplot(gs[current_axs, : ])
            plot_weekly_seasonality(chronos_object, axs=ax)
            current_axs += 1

        if (chronos_object.month_seasonality_order_ > 0):
            ax = fig.add_subplot(gs[current_axs, : ])
            plot_monthly_seasonality(chronos_object, axs=ax)
            current_axs += 1

        if (chronos_object.year_seasonality_order_ > 0):
            ax = fig.add_subplot(gs[current_axs, : ])
            plot_yearly_seasonality(chronos_object, axs=ax)
            current_axs += 1


    # Finally, plot the residuals             
    plot_residuals(predictions, chronos_object, fig, gs[current_axs])


    # Adjust the spacing between the subplots so differnet text components 
    # do not overwrite each other
    plt.subplots_adjust(hspace=1.0)

    # Optionally save the figure
    if (figure_name is not None):
        plt.savefig(figure_name, dpi=96*4)#'''
    else:
        plt.show()

    return fig
########################################################################################################################
def plot_predictions(predictions, chronos_object=None, fig=None, gs_section=None):
    '''
        TODO: update
    '''
    if (chronos_object is None):
        print("No Chronos object provided, assuming timestamp and target columns are named 'ds' and 'y'.")
        time_col = "ds"
        target_col = "y"
    else:
        time_col = chronos_object.time_col_
        target_col = chronos_object.target_col_

    single_figure = False
    if (fig is None):
        single_figure = True

        fig = plt.figure(figsize=(15,5))
        gs = gridspec.GridSpec(1,1, figure=fig)
        gs_section = gs[0,0]

    ax = fig.add_subplot(gs_section)

    # plot credibility intervals
    ax.fill_between(predictions[time_col], 
                    predictions['yhat_upper'], 
                    predictions['yhat_lower'], 
                    color=uncertainty_color_, 
                    alpha=0.3)

    # plot true data points, but only if there is at least one non-nan value
    if (predictions[target_col].isna().sum() < predictions.shape[0]):
        ax.scatter(predictions[time_col], 
                   predictions[target_col], 
                   c=history_color_, 
                   label="observed")

    # plot predictions
    ax.plot(predictions[time_col], 
            predictions['yhat'], 
            c=prediction_color_, 
            label="predictions", 
            linewidth=3)
    
    
    # Set up plot
    ax.set_xlim(predictions[time_col].min(), predictions[time_col].max())
    ax.set_xlabel("Date", size=16)
    ax.set_ylabel("Values", size=16)
    ax.set_title('Predictions', size=18)

    if (single_figure):
        plt.show()
        return fig
########################################################################################################################
#def plot_trend(self, x, trend, changepoint_threshold=0.0, trend_upper=None, trend_lower=None, predictions_start_date = None, axs=None):
def plot_trend(predictions, chronos_object=None, changepoint_threshold=0.0, fig=None, gs_section=None):
    '''
        TODO: update
        A function which plots the trend components, along with historical changepoints. 
        An optional axis can be passed in for the drawing to take place on in case a subplot is used.
        If no matplotlib axis is passed in, the function draws the resulting figure and returns it


        Parameters:
        ------------
        x -                     A tensor containing the timestamps, in days, for the period for which
                                plotting is desired

        trend -                 A tensor containing the trend values, excluding all seasonalities

        changepoint_threshold - The threshold for the value of a changepoint for it to be marked
                                on the plot

        trend_upper -           A tensor containing the upper uncertainty value for the trend. If None
                                no uncertainty is plotted. Both trend_upper and trend_lower must be 
                                specified if uncertainty is desired

        trend_lower -           A tensor containing the lower uncertainty value for the trend

        prediction_start_time - The start time to mark for the prediction. If None, the highest
                                value seen in the training set is used

        axs -                   Optional argument, a matplotlib subplot axis for the drawing to take place
                                on.

        
        Returns:
        ------------
        fig -                   An optional return value. If no axis is passed in as an argument, a new
                                figure is created and returned upon drawing.
    '''

    '''
        trend = predictions['trend'], 
                    changepoint_threshold = changepoint_threshold, 
                    trend_upper = predictions['trend_upper'], 
                    trend_lower = predictions['trend_lower']
    '''

    if (chronos_object is None):
        print("No Chronos object provided, assuming timestamp and target columns are named 'ds' and 'y'.")
        print("Additionally, changepoints will not be labeled on the plot")
        time_col = "ds"
        changepoint_values = []
        changepoint_positions = []
    else:
        time_col = chronos_object.time_col_
        changepoint_values = chronos_object.changepoints_values.detach().numpy()
        changepoint_positions = chronos_object.changepoints_positions.detach().numpy()

    single_figure = False
    if (fig is None):
        single_figure = True

        fig = plt.figure(figsize=(15,5))
        gs = gridspec.GridSpec(1,1, figure=fig)
        gs_section = gs[0,0]

    ax = fig.add_subplot(gs_section)




    ax.plot(predictions[time_col], 
            predictions['trend'], 
            linewidth=3, 
            c=prediction_color_)

    ax.set_xlim(predictions[time_col].min(), predictions[time_col].max())

    # Optionally draw uncertainty trend
    ax.fill_between(predictions[time_col], 
                    predictions['trend_upper'], 
                    predictions['trend_lower'], 
                    color=prediction_color_, 
                    alpha=0.3)

    # Optionally draw changepoint
    #changepoint_values = pyro.param(f'{self.param_prefix_}delta').detach().numpy()
    

    for index, changepoint_value in enumerate(changepoint_values):
        
        # Need to inverse transform every changepoint to the date it represents
        #changepoint_raw_value = self.changepoints[index].item()
        changepoint_second = int(changepoint_positions[index])

        # Now we can make it into a date to use it in our X-axis, which is date based
        changepoint_date_value = datetime.datetime.fromtimestamp(changepoint_second)
        

        if (abs(changepoint_value) >= changepoint_threshold):
            ax.axvline(changepoint_date_value, c="black", linestyle="dotted")#'''
    
    # Optionally mark where history ends and predictions begin
    if (chronos_object is not None):
        predictions_start_date = datetime.datetime.fromtimestamp(chronos_object.history_max_time_seconds)
    
        ax.axvline(datetime.date(predictions_start_date.year, 
                                predictions_start_date.month, 
                                predictions_start_date.day), c="black", linestyle="--")

    

    ax.set_xlabel('Date', size=18)
    ax.set_ylabel('Growth', size=18)

    ax.set_title('Trend', size=18)

    if (single_figure):
        plt.show()
        return fig

########################################################################################################################
def plot_weekly_seasonality(chronos_object, axs=None):
    '''
        TODO: update
        A function which plots the weekly seasonality. An optional axis can be passed in
        for the drawing to take place on in case a subplot is used. If no matplotlib axis
        is passed in, the function draws the resulting figure and returns it


        Parameters:
        ------------
        axs -   Optional argument, a matplotlib subplot axis for the drawing to take place
                on. If no axis is passed in as an argument, a new
                figure is created and returned upon drawing.

        
        Returns:
        ------------
        fig -   An optional return value. If no axis is passed in as an argument, a new
                figure is created and returned upon drawing.
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
    if (chronos_object.seasonality_mode_ == "add"):
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
        TODO: update
        A function which plots the monthly seasonality. An optional axis can be passed in
        for the drawing to take place on in case a subplot is used. If no matplotlib axis
        is passed in, the function draws the resulting figure and returns it


        Parameters:
        ------------
        axs -   Optional argument, a matplotlib subplot axis for the drawing to take place
                on. If no axis is passed in as an argument, a new
                figure is created and returned upon drawing.

        
        Returns:
        ------------
        fig -   An optional return value. If no axis is passed in as an argument, a new
                figure is created and returned upon drawing.
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
    if (chronos_object.seasonality_mode_ == "add"):
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
        TODO: update
        A function which plots the yearly seasonality. An optional axis can be passed in
        for the drawing to take place on in case a subplot is used. If no matplotlib axis
        is passed in, the function draws the resulting figure and returns it


        Parameters:
        ------------
        axs -   Optional argument, a matplotlib subplot axis for the drawing to take place
                on. If no axis is passed in as an argument, a new
                figure is created and returned upon drawing.

        
        Returns:
        ------------
        fig -   An optional return value. If no axis is passed in as an argument, a new
                figure is created and returned upon drawing.
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
    if (chronos_object.seasonality_mode_ == "add"):
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
def plot_residuals(predictions, chronos_object=None, fig=None, gs_section=None):
    '''
        TODO: update description
        A function which plots the residuals of the fit. An optional axis can be passed in
        for the drawing to take place on in case a subplot is used. If no matplotlib axis
        is passed in, the function draws the resulting figure and returns it


        Parameters:
        ------------
        x -         A 1D tensor or array containg the timestamp measurements for the time
                    series

        y_true -    A 1D tensor or array containing the true observations for the time series

        y_pred -    A 1D tensor or array containing the values predicted by this Chronos object


        axs -       Optional argument, a matplotlib subplot axis for the drawing to take place
                    on.

        
        Returns:
        ------------
        fig -       An optional return value. If no axis is passed in as an argument, a new
                    figure is created and returned upon drawing.
    '''
    if (chronos_object is None):
        print("No Chronos object provided, assuming timestamp and target columns are named 'ds' and 'y'.")
        time_col = "ds"
        target_col = "y"
    else:
        time_col = chronos_object.time_col_
        target_col = chronos_object.target_col_


    x = predictions[time_col]
    y_true = predictions[target_col]
    y_pred = predictions['yhat']

    y_residuals = y_true - y_pred

    single_figure = False
    # Make sure to build a new figure if we don't have one
    if (fig is None):
        single_figure = True
        fig = plt.figure(figsize=(15,5))
        gs = gridspec.GridSpec(1, 1, figure=fig)
        gs_section = gs[0]
    
    ax = fig.add_subplot(gs_section)
    ax.set_title('Residuals', size=18)
    #ax.set_xticks([], minor=True)
    #ax.set_yticks([], minor=True)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    ax.tick_params(labelbottom=False, labelleft=False)

    internal_gs_1 = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=gs_section, wspace=0.0)
    axs = fig.add_subplot(internal_gs_1[0, :-1])

    axs.scatter(x, y_residuals, c=prediction_color_, s=4)
    axs.set_xlim(x.min(), x.max())
    axs.set_xlabel('Date', size=18)
    axs.set_ylabel('Residuals', size=18)
    

    axs = fig.add_subplot(internal_gs_1[0, -1])
    axs.hist(y_residuals[y_residuals <= 0.0], color=overprediction_color_, bins=50, orientation="horizontal")#, ec="black")
    axs.hist(y_residuals[y_residuals > 0.0], color=underperdiction_color_, bins=50, orientation="horizontal")#, ec="black")
    axs.xaxis.set_ticks_position('none') 
    axs.yaxis.set_ticks_position('none') 
    axs.tick_params(labelbottom=False, labelleft=False)

    if (single_figure):
        plt.show()
        return fig

########################################################################################################################
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
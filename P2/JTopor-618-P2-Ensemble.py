# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:34:33 2017

@author: James Topor

***************************************************************************
NOTE: This python code is specific to the Quantopian investment algorithm
development platform. As such, this code CANNOT be executed within a basic
python shell or IDE. 
***************************************************************************

This code makes use of three separate machine learning algorithms for purposes
of attempting to predict the upward or downward movement of an individual financial
market security. The three algorithms used are as follows:
    
    1. Random Forest Classification
    2. Support Vector Classification
    3. Gaussian Naive Bayes Classification

These three classification algorithms are used as part of an "ensemble" stock trading
methodology, wherein the predictive output of each algorithm has a direct impact
on the structure of the next stock trade to be executed. The algorithms themselves
are fitted through the use of recent "minutely" stock/security data metrics, such
as price, volume, high, and low. 

The methodology used within the code proceeds as follows:
    
    build_models() module:
        1. To fit the classification models, the most recent 300 minutes of price, 
        volume, high, and low data are retrieved.
        
        2. Binary 0/1 vectors indicating the directional changes in prices, volumes, 
        highs and lows are derived from the 300-minute data set.
        
        3. A 15-item (items = minutes) length sliding "window" is applied to 
        the binary 0/1 vectors calculated in 
        step 2 above to extract small chronological subsets to be used for model fitting. 
        This process yields two data structures: one containing data relative to the
        independent variables and one containing only the dependent variable to be 
        predicted. In this code, we use the changes in price, volume, high, and low
        as the independent variables while the change in price is the dependent variable.
        
        4. Each of the three classification models is fitted using the binary 0/1 
        subsets derived in step 3. Once fitted, the rounded output of each classifier
        will be indicative of whether or not the respective classification algorithm
        sees the price of the security either rising (rounded output == 1) or falling
        (rounded output == 0).
        
    trade() module:
        1. Retrieve the most recent 15 minutes of price, volume, high, and low data
        
        2. Create binary 0/1 vectors indicating the directional changes of the
        15-minute snippet of data retrieved in step 1 above for the price, volume, 
        high, and low variables.
        
        3. Create a single feature vector comprised of the binary 0/1 vectors created
        in step 2 above.
        
        4. Submit the feature vector to each of the three classification algorithms.
        
        5. Sum the rounded outputs of the three classification algorithms. This sum
        is referred to as "votes" in the code.
        
        6. The number of "votes" is then used to assign a value to a 'weight'
        variable. The 'weight' variable is used to adjust the amount of the security
        to be bought or sold relative to the predictions of the three classification
        algorithms. These weights are user-configurable, but are set as follows 
        herein:
            if votes == 3 (all 3 classifiers predict a price increase)
                weight = 1.00
            if votes == 2 (2 of the 3 classifiers predict a price increase)
                weight = 0.75
            if votes == 1 (1 of the 3 classifiers predicts a price increase)
                weight = 0
            if votes == 0 (None of the 3 classifiers predicts a price increase)
                weight == 0
        
        7. The total amount of cash held in the portfolio is then determined
        
        8. The algorithm then allows a maximum of 80% of that cash to be used 
        for the purchase of shares of the security. The 'weight' variable scales
        the available cash amount before an order is placed. So, for example, if
        votes == 2, a total of (cash * 0.80 * 0.75) can be purchased. The exact
        number of shares to be purchased is determined by dividing the scaled available
        cash amount by the current price of the security.
        
        9. Place an order for the calculated amount of shares. Please note that for the
        code as currently configured:
            - if votes == 1, no stock will be bought or sold
            - if votes == 0, all currently held stock will be sold
                   
The build_models() and trade() modules are invoked according to a pre-defined schedule 
as shown within the initialize() and before_trading_start() modules.  build_models() is 
invoked prior to the start of trading each day, and trade() is first invoked during
the first minute of trading. Subsequently, both modules are invoked hourly up until 
30 minutes before the close of trading.

Additional comments:
            
    1. The trade() module has built-in flags that support the shorting of a security if
    so desired by the user. For example, shorting might be appropriate if the price
    of a security is predicted to decline. To enable shorting, set 
    context.shorting_enabled = True and set the weight variable to a negative value
    between (-1, 0) within the trade() module where the "elif votes == 0:" clause
    is evaluated.
    
    2. The user is free to change the values assigned to the weight variable if
    so desired. For example, the code as implemented here ensures that no trade
    is executed if only one of the three classifiers predicts a price increase.
    To change that behavior, simply increase the weight value within the trade()
    module where the "elif votes == 1:" clause is evaluated. 
    
Backtesting this code as-is with $100,000 in initial capital for the period 1/4/2010
- 4/7/2017 on Apple's stock (AAPL) produces a total return = 347.9%, alpha = 0.09,
sharpe = 1.29. The algorithm actually outperformed Apple's stock for long streches
of time as can be seen when executed within Quantopian.
    
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm  
from sklearn.naive_bayes import GaussianNB

import math

import numpy as np
import pandas as pd

###################################################

def initialize(context):

    context.window_length = 15 # Number of prior bars to study
    context.ts_length = 300 # length of time series to derive diff sequences from
    context.long = False
    context.short = False
    context.shorting_enabled = False

    # initialize the 3 machine learning / classification algorithms
    context.RFC = RandomForestClassifier(n_estimators=20, random_state = 1)  
    context.SVC = svm.SVC(random_state = 1)
    context.GNB = GaussianNB()
    
    context.RFC_pred = 0  
    context.SVC_pred = 0
    context.GNB_pred = 0
    
    # SPY SP500
    # context.s1 = sid(8554)
    # set_benchmark(sid(8554))
    
    # XLU Utility sector ETF
    # context.s1 = sid(19660)
    
    # DUKE energy
    # context.s1 = sid(2351)
    # set_benchmark(sid(2351))
    
    # apple
    set_benchmark(sid(24))
    context.s1 = sid(24)
        
    # Run every day, at market open.
    schedule_function(trade, date_rules.every_day(), 
                      time_rules.market_open(minutes=1))
    
    # Run every day, 1 hr after market open.
    schedule_function(model_trade, date_rules.every_day(), 
                      time_rules.market_open(minutes=60))
    
    # Run every day, 2 hrs after market open.
    schedule_function(model_trade, date_rules.every_day(), 
                      time_rules.market_open(minutes=120))
    
    # Run every day, 3 hrs after market open.
    schedule_function(model_trade, date_rules.every_day(), 
                      time_rules.market_open(minutes=180))
    
    # Run every day, 4 hrs after market open.
    schedule_function(model_trade, date_rules.every_day(), 
                      time_rules.market_open(minutes=240))
    
    # Run every day, 5 hrs after market open.
    schedule_function(model_trade, date_rules.every_day(), 
                      time_rules.market_open(minutes=300))
        
    # Run every day, 6 hrs after market open.
    schedule_function(model_trade, date_rules.every_day(), 
                      time_rules.market_open(minutes=360))
 
###################################################
# build a new set of classifier models every day before market open
def before_trading_start(context,data):
     build_models(context, data)
    
###################################################

def handle_data(context, data):
    # not used since we have scheduled functions
    pass

###################################################
# scheduled function
def model_trade(context, data):
    build_models(context, data)
    trade(context, data)

####################################################

def build_models(context, data):
        
    # Get block of minutely price, volume, high, low data
    recent_prices = data.history(context.s1, 'price', context.ts_length, '1m').values
    recent_volumes = data.history(context.s1, 'volume', context.ts_length, '1m').values
    recent_highs = data.history(context.s1, 'high', context.ts_length, '1m').values
    recent_lows = data.history(context.s1, 'low', context.ts_length, '1m').values
    
    # Make a list of 1's and 0's, 1 when the price increased from the prior bar  
    price_changes = np.diff(recent_prices) > 0  
    volume_changes = np.diff(recent_volumes) > 0
    high_changes = np.diff(recent_highs) > 0
    low_changes = np.diff(recent_lows) > 0
    
    X = [] # Independent, or input variables
    
    Y = [] # Dependent, or output variable
    
    # Create feature vectors for each 'window_length' subset
    for i in range(0, context.ts_length - context.window_length-1):
        
        # add independent variables calculated from past data
        feature = np.concatenate( (price_changes[i:i+context.window_length-1], 
                            volume_changes[i:i+context.window_length-1],
                            high_changes[i:i+context.window_length-1],
                            low_changes[i:i+context.window_length-1]) )
        
        # add features to array of features
        X.append(feature.flatten()) 
        # append price change vector to those already accummulated
        Y.append(price_changes[i+context.window_length]) 

    # fit all three models
    context.RFC.fit(X, Y) # Generate the random forest model
    context.SVC.fit(X, Y) # Generate the SVC model
    context.GNB.fit(X, Y) # Generate Gaussian Naive Bayes model

################################################################################
    
def trade(context, data): 
    
     # if there are open orders awaiting executlon, exit function
    if len(get_open_orders()) > 0:
        return

    if context.RFC : # Check to ensure a model has already been created
    
        # Get recent data for each predictive variable
        recent_prices = data.history(context.s1, 'price', context.window_length, '1m').values
                                          
        recent_volumes = data.history(context.s1, 'volume', context.window_length, '1m').values
        
        recent_highs = data.history(context.s1, 'high', context.window_length , '1m').values
        
        recent_lows = data.history(context.s1, 'low', context.window_length , '1m').values
        
        
        # Make a list of 1's and 0's, 1 when the price increased from the prior bar  
        price_changes = np.diff(recent_prices) > 0  
        volume_changes = np.diff(recent_volumes) > 0
        high_changes = np.diff(recent_highs) > 0
        low_changes = np.diff(recent_lows) > 0

        # create a single feature comprised of each variable's recent values
        target_feature = np.concatenate((price_changes, 
                                         volume_changes, 
                                         high_changes,
                                         low_changes)).flatten()
        
        # get predictions from each model
        context.RFC_pred = context.RFC.predict(target_feature)  
        context.SVC_pred = context.SVC.predict(target_feature)
        context.GNB_pred = context.GNB.predict(target_feature)
 
        # now tally "votes": sum predicted 0/1 values from the 3 models
        votes = int(context.RFC_pred) + int(context.SVC_pred) + int(context.GNB_pred)
        log.info(votes)
        
        # set the weight percentage based on the number of votes
        if votes == 3:
            weight = 1 # maximize stock purchase amount
        elif votes == 2:
            weight = 0.75 # buy some shares, but not maximal amount
        elif votes == 1: # if only 1 positive prediction, no trade is executed
            weight = 0 # votes ==1, weight == 0 results in no trade
        elif votes == 0:
            # if price decline predicted, set weight = 0 to sell everything
            # if shorting desired, set -1 <= weight <= 0, e.g., -0.5
            weight = 0
        
        # make sure stock is currently tradeable
        if data.can_trade(context.s1):
            # get total cash available for trading
            cash = context.portfolio.cash

            # allow 80% of current cash to be traded + calc number of shares
            # that amount of cash will purchase
            s1_shares = (cash * 0.80) / data.current(context.s1, 'price')
            
            # determine appropriate trade relative to votes + weight variables
            if votes >= 1:
                # if currently short, close it out
                if context.short == True:
                    order_target(context.s1, 0)
                # open or add to long trade
                if weight > 0: # if weight > 0, execute trade; else do nothing
                    # adjust the total amt of shares to purchase by weight value
                    order(context.s1, weight * s1_shares)
                    context.long = True
                    context.short = False
            else: # else all 3 classifiers have predicted price decline
                # sell everything
                order_target(context.s1, 0)
                
                # if shorting enabled and weight is negative, execute a short
                if context.shorting_enabled == True:
                    if -1 <= weight < 0:
                        order(context.s1, weight * s1_shares)
                        context.long = False
                        context.short = True
    
        record(RFC_pred = int(context.RFC_pred), SVC_pred = int(context.SVC_pred), 
               GNB_pred = int(context.GNB_pred))

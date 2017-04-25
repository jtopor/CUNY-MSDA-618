# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:16:28 2017

@author: Hammer
"""


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier  
from sklearn import svm  
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.naive_bayes import GaussianNB

from collections import deque
import math

import numpy as np
import pandas as pd
import random

###################################################

def initialize(context):
    
# Different pairs of securities were tested. Comment out / uncomment as desired.
# The algorithm does not perform equally for each pair - for example, 
# the algorithm performed rather poorly when using Wells Fargo and BofA as a pair

    # random.seed(42)
    context.window_length = 15 # Amount of prior bars to study
    context.ts_length = 300 # length of time series to derive diff sequences from
    context.long = False
    context.short = False

    context.RFC = RandomForestClassifier(n_estimators=20, random_state = 1)  
    context.SVC = svm.SVC(random_state = 1)
    context.GNB = GaussianNB()
    
    # context.KNC = KNC()
    
    context.RFC_pred = 0  
    context.SVC_pred = 0
    context.GNB_pred = 0
    # context.KNC_pred = 0
    
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
    
    # Build a new set of classifier models at end of each day
   # schedule_function(build_models, date_rules.every_day(), 
   #                   time_rules.market_close(minutes=10))
    
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
    
    # Run every day, 1 min before market close.
    #schedule_function(model_trade, date_rules.every_day(), 
    #                  time_rules.market_close(minutes=1))
 
###################################################
# build a new set of classifier models every day before market open
def before_trading_start(context,data):
     build_models(context, data)
    
###################################################

def handle_data(context, data):
    # not used since we have a scheduled function
    pass

###################################################

def model_trade(context, data):
    build_models(context, data)
    trade(context, data)

####################################################

def build_models(context, data):
        
    # Get the relevant daily prices, volumes
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
    
    # Now create feature vectors for each 'window_length' subset of 
    # t # create a vector containing the 10-day moving average price
    for i in range(0, context.ts_length - context.window_length-1):
        
        # add independent variables calculated from past data
        feature = np.concatenate( (price_changes[i:i+context.window_length-1], 
                            volume_changes[i:i+context.window_length-1],
                            high_changes[i:i+context.window_length-1],
                            low_changes[i:i+context.window_length-1]) )
        
        # X.append(feature) # Store prior price changes
        X.append(feature.flatten()) # add features to array of features
        Y.append(price_changes[i+context.window_length]) # Store the last price change

    context.RFC.fit(X, Y) # Generate the random forest model
    # context.AdaBC.fit(X, Y) # generate adaboost model
    context.SVC.fit(X, Y) # Generate the SVC model
    context.GNB.fit(X, Y) # Generate Gaussian Naive Bayes model

################################################################################
    
def trade(context, data): 
    
     # if there are open orders awaiting executlon, exit function
    if len(get_open_orders()) > 0:
        return

    if context.RFC : # Check to ensure a model has already been created
    
        # Get recent prices for most recent n-1 days
        recent_prices = data.history(context.s1, 'price', context.window_length, '1m').values
                                          
        recent_volumes = data.history(context.s1, 'volume', context.window_length, '1m').values
        
        recent_highs = data.history(context.s1, 'high', context.window_length , '1m').values
        
        recent_lows = data.history(context.s1, 'low', context.window_length , '1m').values
        
        
        # Make a list of 1's and 0's, 1 when the price increased from the prior bar  
        price_changes = np.diff(recent_prices) > 0  
        volume_changes = np.diff(recent_volumes) > 0
        high_changes = np.diff(recent_highs) > 0
        low_changes = np.diff(recent_lows) > 0

        target_feature = np.concatenate((price_changes, 
                                         volume_changes, 
                                         high_changes,
                                         low_changes)).flatten()
        
        context.RFC_pred = context.RFC.predict(target_feature)  
        context.SVC_pred = context.SVC.predict(target_feature)
        context.GNB_pred = context.GNB.predict(target_feature)
 

        votes = int(context.RFC_pred) + int(context.SVC_pred) + int(context.GNB_pred)
        log.info(votes)
        
        if votes == 3:
            position = 1 # max out the portofolio
        elif votes == 2:
            position = 0.75 # adjust portfolio
        elif votes == 1: # if only 1 positive prediction do nothing
            position = 0
        elif votes == 0:
            position = 0 # sell everything + short

               # make sure stock is currently tradeable
        if data.can_trade(context.s1):
            # get total cash available for trading
            cash = context.portfolio.cash

            # allow 40% of cash to be traded for long            
            s1_shares = (cash * 0.80) / data.current(context.s1, 'price')
            
            # adjust amt of shares to purchase by position value
            if votes >= 1:
                # if short position open, close it out
                if context.short == True:
                    order_target(context.s1, 0)
                # open or add to long position
                if position > 0:
                    order(context.s1, position * s1_shares)
                    context.long = True
                    context.short = False
            else:
                # sell everything + go short
                order_target(context.s1, 0)
                # order(context.s1, position * s1_shares)
                context.long = False
                # context.short = True
    
        record(RFC_pred = int(context.RFC_pred), SVC_pred = int(context.SVC_pred), 
               GNB_pred = int(context.GNB_pred))

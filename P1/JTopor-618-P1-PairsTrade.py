"""
Created on Wed Mar 15 13:37:01 2017
@author: JTopor

***************************************************************************
NOTE: This python code is specific to the Quantopian investment algorithm
development platform. As such, this code CANNOT be executed within a basic
python shell or IDE. 
***************************************************************************

This code implements a pairs trading algorithm that checks for the 
non-stationarity and cointegration of the times series of 2 stock prices before
executing an appropriate long/short trade. The algorithm is run twice each day:
60 minutes after market open and 30 minutes before market close. Running the
algorithm twice per day allows it to account for significant intra-day 
price swings.

The algorithm proceeds as follows:
    
    1. A 20-day price history / time series for each stock is compiled via 
    Quantopian's data.history() function.
    
    2. Both time series are normalized via application of a base-10 log function
    
    3. A daily "spread" is calculated based on the normalized time series.
    
    4. The mean and standard deviation of the daily spread is calculated
    
    5. The spot price of each stock is fetched and normalized via application
    of a base-10 log function.
    
    6. The spread between the normalized spot prices is calculated
    
    7. A Z-score is computed using the results of steps 4 and 6:
        zscore = (spot_spread - sprd_mean)/sprd_sdev
        
    8. The Z-score is then used as the basis for determining whether or not
    a new trade is needed.  The direction of the trade depends on the magnitude
    of the Z-score as well as the existance or non-existance of any previous 
    trades, as indicated by the global flags "context.in_high" and 
    "context.in_low".
    
    9. If the magnitude of the Z-score indicates that a new trade is called for,
    both of the normalized time series are evaluated 
    to determine whether or not they are stationary. If either series is 
    stationary, a trade is not called for. If both series are non-stationary,
    a co-integration test is applied.
    
    10. If cointegration exists, the appropriate long/short direction for a new
    trade is determined via the magnitude of the previously calculated Z-score
    
    11. Equal dollar amounts of both stocks are thenlonged/shorted as 
    determined in step 10.
    
Additional checks are done throughout the algorithm to ensure that open long/short
positions are closed out whenever mean reversion has occurred and also to 
ensure that new long/short positions are not opened if directionally identical
long/short is already in place.

NOTE: Backtesting this as-is with $50,000 in initial capital for the period
1/1/2010 - 3/20/2017 produces total return = 1.8%, alpha = 0.00, sharpe = 0.24,
which would appear to be in line with the market-neutral nature of pairs trading.

So in this instance if we own stock in Duke Energy, it appears we can hedge 
our position in that stock by pairs trading against the XLU energy sector ETF
using this algorithm.

"""
import math
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd

###################################################

def initialize(context):
    
# Different pairs of securities were tested. Comment out / uncomment as desired.
# The algorithm does not perform equally for each pair - for example, 
# the algorithm performed rather poorly when using Wells Fargo and BofA as a pair

    # XLU Utility sector ETF
    context.s1 = sid(19660)
    # DUKE energy
    context.s2 = sid(2351)
    
    # Reference to wells fargo
    # context.s1 = sid(8151)
    # bank of america
    # context.s2 = sid(700)
        
    # Reference to SPDR S+P 500
    # context.s1 = sid(8554)
    # SPDR DJIA
    # context.s2 = sid(2174)
    
    # Reference to ExxonMobil
    # context.s1 = sid(8347)
    # Chevron
    # context.s2 = sid(23112)
    
    # set flags indicating whether or not the algorithm has already engaged in
    # a particular trade
    context.in_high = False
    context.in_low = False
    
    context.security_list = [context.s1, context.s2]
    
    # Run every day, 1 hour after market open.
    schedule_function(pairs_trade, date_rules.every_day(), 
                      time_rules.market_open(minutes=60))
    
    # Run every day, .5 hour before market close.
    schedule_function(pairs_trade, date_rules.every_day(), 
                      time_rules.market_close(minutes=30))
    
###################################################

def handle_data(context, data):
    # not used since we have a scheduled function
    pass
    
###################################################  
# use augmented Dickey-Fuller to check for stationarity of a time series
def check_for_stationarity(X, cutoff=0.05):
    # H_0 in adfuller is unit root exists (non-stationary)
    # Need significant p-value for series to be stationary
    pvalue = adfuller(X)[1]
    if pvalue < cutoff:
        return True
    else:
        return False

####################################################

def pairs_trade(context, data):
    
    # if there are open orders awaiting executlon, exit function
    if len(get_open_orders()) > 0:
        return
    
    # init cointegration flag and stock ID's
    s_coint = False
    s1 = context.s1
    s2 = context.s2
    
    # get 20 days of pricing data for both stocks
    s1p20 = data.history(context.s1, 'price', 20, '1d')
    s2p20 = data.history(context.s2, 'price', 20, '1d')

    # extract pricing info from Quantopian series object
    s1_raw = pd.DataFrame(data = s1p20.values, columns = ["values"])
    s1_raw = s1_raw.astype(float)
    
    # extract pricing info from Quantopian series object
    s2_raw = pd.DataFrame(data = s2p20.values, columns = ["values"])
    s2_raw = s2_raw.astype(float)
    
    # calculate log10 normalization of each series + calculate spread
    s1_series = np.log10(s1_raw["values"])
    s2_series = np.log10(s2_raw["values"])
    spread = s1_series - s2_series 
    
    # calc mean + std dev of spread for time series
    sprd_mean = np.mean(spread)
    sprd_sdev = np.std(spread)
        
    # get spot prices, normalize them, and calculate spot spread
    s1_spot = np.log10(data.current(s1, 'price'))
    s2_spot = np.log10(data.current(s2, 'price'))       
    spot_spread = s1_spot - s2_spot
        
    # Compute z-score - check if sd = 0 to avoid div by zero error
    if sprd_sdev > 0:
        zscore = (spot_spread - sprd_mean)/sprd_sdev
    else:
        zscore = 0
              
    # now check if mean reversion has happened + whether any long/short position is open
    # if abs(Zscore) < 1, then any open long/shorts should be closed out since 
    # spread has reverted to mean.
    if abs(zscore) < 1 and (context.in_high or context.in_low) :
        if all(data.can_trade(context.security_list)):
            log.info("Mean reversion => close any outstanding positions")
            log.info("Z score = ")
            log.info(zscore)
            order_target(s1, 0)
            order_target(s2, 0)
            context.in_high = False
            context.in_low = False
                
            # exit function since we know there's nothing else to do during this iteration
            return
        
    # else if abs(zscore) > 1 then check whether a new long/short is required
    elif abs(zscore) > 1:
        if zscore > 1 and context.in_high:
            # if spread indicates short x / long y but we already have 
            # short x / long y outstanding, exit function
            return
        elif (zscore > 1 and context.in_low) or (zscore < -1 and context.in_high):
            # if zscore suddently swings from <1 to >1 or >1 to <1
            # a WHIPSAW has occurred so close out any open position
            order_target(s1, 0)
            order_target(s2, 0)
            context.in_high = False
            context.in_low = False
            log.info("##### Whipsaw Forces Closeout of Positions ####")
            # return since trades need to be processed before 
            # any new position can be opened
            return
            
        elif zscore < -1 and context.in_low:
            # if spread indicates long x / short y but we already have 
            # long x / short y outstanding, exit function
            return
            
        # Otherwise, check for non-stationarity + cointegration of both series
        # to determine whether a new long/short can be implemented
        
        # check both series for non-stationarity
        s1_stat = check_for_stationarity( s1_series, .10 )
        s2_stat = check_for_stationarity( s2_series, .10 )
    
        if not s1_stat and not s2_stat:
            log.info("Both series are non-stationary")
            # check for cointegration
            score, pvalue, _ = coint(s1_series, s2_series)
            if pvalue <= 0.05:
                s_coint = True
            else: # else exit since the non-stationary series are not cointegrated
                log.info("Series are not cointegerated")
                s_coint = False
                return
        else: # else exit since we don't have 2 non-stationary time series
                return

        # if cointegrated, execute the apporpriate long/short combo
        if s_coint == True:
            log.info("Cointegration => Trade Required")
            log.info("Zscore = ")
            log.info(zscore)
            
            # get total cash available for trading
            cash = context.portfolio.cash
            
            # how many shares of each can be bought or sold?
            # allow 40% of cash to be traded for each stock => 80% of all cash available
            
            s1_shares = (cash * 0.40) / data.current(s1, 'price')
            s2_shares = (cash * 0.40) / data.current(s2, 'price')

            if zscore > 1 and not context.in_high and all(data.can_trade(context.security_list)):
                log.info("##### Selling x and Buying y #####")
                log.info("x shares sold")
                log.info(-s1_shares)
                log.info("y shares bought")
                log.info(s2_shares)
                log.info("###################################")
                order(s1, -s1_shares)
                order(s2, s2_shares) 
                context.in_high = True
                context.in_low = False
        
            elif zscore < -1 and not context.in_low and all(data.can_trade(context.security_list)):
                log.info("##### Selling y and Buying x #####")
                log.info("x shares bought")
                log.info(s1_shares)
                log.info("y shares sold")
                log.info(-s2_shares)
                log.info("##################################")
                order(s1, s1_shares) 
                order(s2, -s2_shares) 
                context.in_high = False
                context.in_low = True
        
    record('zscore', zscore, lev=context.account.leverage)
   
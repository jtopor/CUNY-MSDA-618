# -*- coding: utf-8 -*-
"""
Created on Sat May 06 13:58:32 2017

@author: James Topor

***************************************************************************
NOTE: This python code is specific to the Quantopian investment algorithm
development platform. As such, this code CANNOT be executed within a basic
python shell or IDE. 
***************************************************************************

This algorithm makes use of a Kalman filter in an attempt to predict the movement of 
a security based on the value of a second independent security that is similar in 
nature to that of the first (i.e., from the same industry or sector and having a 
reasonably similar business model or purpose). 

The Kalman filter calculations are based upon the sample Kalman filter code provided to 
us on Blackboard. However, the specifics of both the implementation of the filter and the 
associated trading algorithm are distinct in several aspects:

1. Contrary to the sample Blackboard code which attempted to implement a pairs trading 
algorithm, the algorithm used here makes use of a more aggressive "one way" trading 
approach, wherein the "dependent" security whose price is being estimated is traded 
either "long" or "short" with no offsetting "short/long" executed for "independent" 
security whose price is being used to estimate that of the "dependent" security.

2. The algorithm used here does not permit a trade to occur if the filter has not 
yielded a non-zero estimate of the price of the security to be traded. By contrast, 
the code provided on Blackboard allowed a trade to occur even if the estimate of the price 
of the dependent variable (__yhat__) was zero. As such, the sample blackboard code allowed 
at least one unbalanced pairs trade to occur at the start of its execution. Checking the 
value of the __yhat__ variable to ensure that it is non-zero eliminates such behavior and 
ensures that the variables within the Kalman filter have attained at least semi-reasonable 
values prior to their being used as the basis of a trade.

3. Unlike the sample Blackboard code, the Kalman filter used here is periodically "reset" 
(or "re-initialized") to remove the influence of relatively outdated time series data from 
estimates of a security's value. The reset is controlled via a pair of user-settable global 
variables (__context.max_filter_iter__ and __context.filter_iter__). For the results discussed 
below, the maximum number of filter iterations allowed prior to a filter reset was 120, which 
within the algorithm equates to 120 trading days due to the once daily execution of the algorithm. 
Once 120 filter iterations have been executed, the __filter_reset()__ function is invoked to 
reset the values of each of the component filter variables to their original values, thereby 
enabling the filter to be free of the influence of relatively outdated time series data. 

NOTE: Repeated testing of the Kalman filter as implemented herein indicated that periodic resetting 
of the filter yielded signficant improvements in the algorithm's investment returns. 
More research would be needed to determine whether periodic resetting should be required 
within other contexts as well.

4. The algorithm is invoked once daily via a scheduled function as defined within 
the __initialize()__ function.

5. The amount of a security to be longed/shorted is determined by calculating the magnitude of 
the amount the estimate of the security's price exceeds its actual value: The larger the 
deviation, the more capital is allowed to be used for a trade. As with the sample Blackboard 
code, no trade is executed if that difference is within 1 standard deviation of the actual 
value of the security. However, if that difference exceeds 1 standard deviation, capital is 
allocated to a trade differently. That rule is quantified within Python as shown in the code 
snippet provided below. 

# ------------------------------------------------------------------------

# calculate how much the price estimate exceeds 1 standard deviation from actual price
trade_mag = (abs(e)/sqrt_Q) - 1
    
# allocate more cash to a trade based on amount estimate exceeds 1 std dev of actual price
if trade_mag <= 0.5:
    weight = .3
elif trade_mag > 0.5 and trade_mag <= 1:
    weight = 0.5
elif trade_mag > 1 and trade_mag <= 1.5:
    weight = 0.7
else: # else if difference > 2.5 standard deviations, trade 90% of cash
    weight = 0.9     

# calculate the total number of shares to long or short based on weight + available cash
s2_shares = (cash * weight) / data.current(context.s2, 'price')

# ------------------------------------------------------------------------

A long or short trade is then executed as appropriate.  By contrast, the 
sample Blackboard code executed the purchase or sale of 1000 shares 
whenever the estimate of the price of the security was more than 1 standard deviation 
away from the actual price of the security: As such, larger deviations resulted in no 
additional capital being applied to a trade.

"""

import  numpy   as  np
 
####################################################################################

def  initialize (context ):
    
### various pairings used during testing of algorithm

#    context.s2  = sid(7883) # United Techn. 15% w filter reset, 7.3% without
#    set_benchmark(sid(7883))
#    context.s1  = sid(698) #boeing
   
#    context.s1  = sid(33370) # USD ETF
#    set_benchmark(sid(27894))
#    context.s2  = sid(27894) # Euro ETF
    
#    context.s1  = sid(42950) # FaceBook - Works better w/ filter reset
#    set_benchmark(sid(8554)) # limited to 5.19.12 to present
#    context.s2  = sid(26578) # google as s2 = 43.3% w filter reset
    
    context.s1 = sid(2673) # ford 133.8% w filter reset; 44.6% without
    set_benchmark(sid(40430))
    context.s2 = sid(40430) # GM
    
    
    # delta = a small value used for initializing the transition covaraince
    context. delta  =   0.0001
        
    # np.eye produces an identity matrix of the size of 2 in this case
    # this variable is actually the transition covariance, which is an error term
    # why Vw is used as variable name???
    context. Vw  = context. delta  /   ( 1  - context. delta )   *  np. eye ( 2 )
    
    # init the observation covariance
    context. Ve  =   0.001
    context. beta  = np. zeros ( 2 )

    context. P  = np. zeros ( ( 2,   2 ) )   # Posterior error estimate
    context. R  =  None  # estimate of the measurement error aka noise covariance - set to None initially

    # set max number of times to update filter before re-initialization required
    context.max_filter_iter = 120
    # set counter for number of times filter has been used
    context.filter_iter = 0
    
    context. pos  =   None   # position: long or short
    
    # Run every day, 30 minutes before market close.
    schedule_function(use_kalman, date_rules.every_day(), 
                      time_rules.market_close(minutes=30))
    
####################################################################################
# The Kalman filter needs to be re-initialized periodically to remove the influence of 
# outdate pricing data accummulated within its variables
# -------------

def filter_reset(context, data):
    
    context. Vw  = context. delta  /   ( 1  - context. delta )   *  np. eye ( 2 )
    
    # init the observation covariance
    context. Ve  =   0.001
    context. beta  = np. zeros ( 2 )

    context. P  = np. zeros ( ( 2,   2 ) )   # Posterior error estimate
    context. R  =  None  # estimate of the measurement error aka noise covariance - set to None initially
    context.filter_iter = 0
    
####################################################################################

def handle_data(context, data):
    # not used since we have scheduled functions
    pass

####################################################################################
    
def use_kalman (context, data) :
    
    # check whether filter needs to be re-initialized
    if context.filter_iter == context.max_filter_iter:
        filter_reset(context, data)
    
    # increment filter usage counter
    context.filter_iter += 1
    
    # get current price of each asset
    x = np. asarray ( [data.current(context.s1, 'price'),   1.0 ] ). reshape ( ( 1,   2 ) )
    log.info("x")
    log.info(x)
    
    y = data.current(context.s2, 'price')
    log.info("y")
    log.info(y)
        
    # update covariance prediction: if first time through, set R to all zeroes
    if  context. R   is   not   None:
        context. R  = context. P  + context. Vw # this is the covariance prediction ??
    else:
        context. R  = np. zeros ( ( 2,   2 ) )
     
    # ---------------------------------------
    # update Kalman filter with latest price info
    
    # calculate an estimate of price of  context.s2 stock
    yhat = x. dot (context. beta ) 
    log.info("yhat")
    log.info(yhat)
   
    # calc estimate of the process error
    Q = x. dot (context. R ). dot (x. T )  + context. Ve
    log.info("Q")
    log.info(Q)
    
    # calc standard deviation of signal
    sqrt_Q = np. sqrt (Q )
    log.info("SQRT(Q)")
    log.info(sqrt_Q)
    
    # calc diff betw actual price and estimated price
    e = y - yhat   
    log.info("e")
    log.info(e)
    
    # calculate the magnitude of the deviation between the estimated price and the updated standard deviation
    trade_mag = (abs(e)/sqrt_Q) - 1
    log.info("Trade Magnitude")
    log.info(trade_mag)
    
    K = context. R. dot (x. T )  / Q
    log.info("K")
    log.info(K)

    context. beta  = context. beta  + K. flatten ( )   *  e   # calculate beta
    log.info("beta")
    log.info(context.beta)
 
    context. P  = context. R  - K   *  x. dot (context. R )    # estimate error
    log.info("P")
    log.info(context.P)
    
    # end update of Kalman filter
    # ---------------------------------------
       
    #record relevant data values
    #beta and alpha (difference betweens actual and expected)
    record (beta=context. beta [ 0 ], alpha=context. beta [ 1 ] )
    # e < 5 only used to filter out extreme values from backest plot; no other reason for it
    if  e   <   5: 
        record (spread= float (e ), Q_upper= float (sqrt_Q ), Q_lower= float (-sqrt_Q ) )

    # if estimate of price of stock y is 0, exit since no trade should be executed
    # this can happen during first few iterations after start or after filter reset
    if yhat == [ 0.]:
        log.info("yhat estimate == 0: Exiting use_kalman()")
        return
   
    # if any outstanding long or short, close position
    if  context. pos   is   not   None:
        if  context. pos  ==   'long'   and  e   >  -sqrt_Q:
            log.info('closing long')
            order_target (context.s2,   0 )
            context. pos  =   None
        elif  context. pos  ==   'short'   and  e   <  sqrt_Q:
            log.info('closing short')
            order_target (context.s2,   0 )
            context. pos  =   None

    # if there is no outstanding long or short, open a new one
    if  context. pos   is   None:
        # get total cash available for trading
        cash = context.portfolio.cash

        # calculate percentage of cash to be used for trade
        # set the weight percentage based on the trade_nag value
        if trade_mag <= 0.5:
            weight = .3
        elif trade_mag > 0.5 and trade_mag <= 1:
            weight = 0.5
        elif trade_mag > 1 and trade_mag <= 1.5:
            weight = 0.7
        else: # else if difference > 2.5 standard deviations, trade 90% of cash
            weight = 0.9        
        
        if  e   <  -sqrt_Q:
            # go long on context.s2
            s2_shares = (cash * weight) / data.current(context.s2, 'price')
            order (context.s2,   s2_shares )
            context. pos  =   'long'
            
        elif  e   >  sqrt_Q:
            # go short on context.s2
            s2_shares = (cash * weight) / data.current(context.s2, 'price')
            order (context.s2, - s2_shares )
            context. pos  =   'short'
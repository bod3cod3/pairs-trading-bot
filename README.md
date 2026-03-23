# Overview
This bot implements a pairs trading strategy based on the idea that certain pairs of assets, 
such as ETFs, exhibit long-term relationships. When the spread of their prices deviates from
equilibrium, the strategy executes trades that take positions expecting reversion. 

# Logic
1. Use the Engle-Granger test to screen for cointegrated asset pairs
2. Compute the spread
    * Linear regression to estimate the hedge ratio (beta)
    * spread = asset_A_price - beta * asset_B_price
3. Calculate the z-score of the spread
    * compute mean and standard deviation
4. Generate trading signals
    * if z is below -(threshold), long the spread
    * if z is above (threshold), short the spread
    * when z returns to a certain equilibrium, exit the position
5. Backtesting

# Structure

data.py -> load price data

hedge.py -> compute the hedge ration and spread

test_data.py -> backtesting and asset pair screening

# Disclaimer

This is a personal project for educational purposes and does not constitute financial advice

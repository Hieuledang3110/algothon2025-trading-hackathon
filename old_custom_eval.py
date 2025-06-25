#!/usr/bin/env python
#This is Eric's branch, the main algorithm is located inside test.py

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from test import trendFollow as getPosition
from test import SMA
from test import WMA
from sklearn.linear_model import LinearRegression

##FEEDBACK VARIABLES

#Set if you want to look at the chart
openChart = True
#Opens up the graphs related to this stock
stockIndex = 6
#Change this to see a stock's price before the trading period begins
headStart = 500
#The amount of biggest losers/winners you want to see
nStocksFeedback = 22



nInst = 0
nt = 0
commRate = 0.0005
dlrPosLimit = 10000


def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T

pricesFile= "./prices.txt"   # "./priceSlice_test.txt"
prcAll = loadPrices(pricesFile)
print ("Loaded %d instruments for %d days" % (nInst, nt))

def smooth_trend_regression(values, long_window=200, smooth_window=20):
    # Step 1: Get long-term trend using all data
    X_all = np.arange(len(values)).reshape(-1, 1)
    long_trend = LinearRegression().fit(X_all, values)
    
    # Step 2: Smooth recent residuals
    trend_line = long_trend.predict(X_all)
    residuals = values - trend_line
    recent_residuals = residuals[-smooth_window:]
    smoothed_residual = np.mean(recent_residuals)
    
    # Combine: long trend + smoothed recent deviation
    next_trend = long_trend.predict([[len(values)]])[0]
    return next_trend + smoothed_residual

def calcPL(prcHist, numTestDays,stockIndex,headStart,nStocksFeedback):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0    
    value = 0
    todayPLL = []
    (_,nt) = prcHist.shape
    startDay = nt + 1 - numTestDays

    #Temp addition to track position history
    positionHistory = [[] for _ in range(len(prcHist))]
    DailyPLHistory = [[] for _ in range(len(prcHist))]
    TotalPLHistory = [[0] for _ in range(len(prcHist))]
    DailyValueHistory = [[10000] for _ in range(len(prcHist))]
    DailyCashHistory = [[10000] for _ in range(len(prcHist))]

    for t in range(startDay, nt+1):
        prcHistSoFar = prcHist[:,:t]
        curPrices = prcHistSoFar[:,-1]

        if (t < nt):
            # Trading, do not do it on the very last day of the test
            newPosOrig = getPosition(prcHistSoFar)
            print(newPosOrig)
            posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
            newPos = np.clip(newPosOrig, -posLimits, posLimits)
            deltaPos = newPos - curPos
            dvolumes = curPrices * np.abs(deltaPos)
            dvolume = np.sum(dvolumes)
            totDVolume += dvolume
            comm = dvolume * commRate
            cash -= curPrices.dot(deltaPos) + comm

            #Temp addition to track position history
            for i in range(len(prcHist)):
                positionHistory[i].append(newPosOrig[i])
                DailyCashHistory[i].append(float(DailyCashHistory[i][-1] - curPrices[i]*deltaPos[i] + dvolumes[i]*commRate))

        else:
            newPos = np.array(curPos)
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue
        ret = 0.0

        #Temp addition to track position history
        for i in range(len(DailyPLHistory)):
            DailyValueHistory[i].append(curPos[i]*curPrices[i]+DailyCashHistory[i][-1])
            DailyPLHistory[i].append(DailyValueHistory[i][-1]-DailyValueHistory[i][-2])
            TotalPLHistory[i].append(sum(DailyPLHistory[i]))

        if (totDVolume > 0):
            ret = value / totDVolume
        if (t > startDay):
            print ("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" % (t,value, todayPL, totDVolume, ret))
            todayPLL.append(todayPL)
    pll = np.array(todayPLL)
    (plmu,plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0

    for i in range(len(positionHistory[stockIndex])):
        positionHistory[stockIndex][i] *= (prcHist[stockIndex][startDay]/10000)
        positionHistory[stockIndex][i] *= float(prcHist[stockIndex][startDay])

    #DONT FORGET TO CHANGE THIS
    movingAverageLength1 = 30
    SMAs = [[] for _ in range(len(prcHist))]
    EMAs = [[] for _ in range(len(prcHist))]
    for i in range(len(prcHist)):
        stockPrices = prcHist[i]
        for j in range(movingAverageLength1,len(stockPrices)):
            SMAs[i].append(SMA(movingAverageLength1,stockPrices[0:j+1]))
            EMAs[i].append(WMA(movingAverageLength1,stockPrices[0:j+1]))


    data = np.loadtxt("prices.txt")
    data = np.rot90(data)
    print(data)
    stock_index = 12
    prices = data[stockIndex]

    # Calculate predictions from day 150 onwards
    predictions = []
    upper_bounds = []
    lower_bounds = []
    actual_days = []

    for day in range(150, len(prices)):
        current_data = prices[:day]
        prediction = smooth_trend_regression(current_data, long_window=200, smooth_window=20)
        
        # Calculate bounds
        percentage = 0.06
        upper_bound = prediction * (1 + percentage)
        lower_bound = prediction * (1 - percentage)
        
        predictions.append(prediction)
        upper_bounds.append(upper_bound)
        lower_bounds.append(lower_bound)
        actual_days.append(day)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(prices)), prices, 'b-', label='Actual Prices')
    plt.plot(actual_days, predictions, 'ro-', label='Predictions', markersize=3)
    plt.plot(actual_days, upper_bounds, 'g--', label='Upper Bound')
    plt.plot(actual_days, lower_bounds, 'g--', label='Lower Bound')
    plt.fill_between(actual_days, lower_bounds, upper_bounds, alpha=0.2, color='green')

    plt.plot([i for i in range(startDay+1, nt+1)],positionHistory[stockIndex],color="orange", linewidth=1)

    plt.title(f'Stock {stockIndex} - Regression Predictions')
    plt.legend()
    plt.grid(True)

    '''
    plt.figure(1)
    plt.title('Black = Stock Price, Orange = Position (scaled so that the max position is roughly equal to the average price of the stock)')
    plt.plot([i for i in range(startDay+1, nt+1)],positionHistory[stockIndex],color="orange", linewidth=1)
    plt.plot([i for i in range(max(startDay+1-headStart,0), nt)],prcHist[stockIndex,range(max(startDay+1-headStart,0),nt)],color="black", linewidth=1)
    '''

    plt.figure(2)
    plt.title('Green = Total Earnings, Red = Daily Earnings)')
    plt.plot([i for i in range(startDay+1, nt+1)],DailyPLHistory[stockIndex][:-1],color="red", linewidth=1)
    plt.plot([i for i in range(startDay+1, nt+1)],TotalPLHistory[stockIndex][:-2],color="green", linewidth=2)
    plt.plot([i for i in range(startDay+1, nt+1)],[0 for _ in range(startDay+1, nt+1)],color="grey", linewidth=1)
    

    rep
    

    if (plstd > 0):
        annSharpe = np.sqrt(249) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume)


# Calculate test period (80% to 100% of data)
total_days = nt
test_start = int(total_days * 0.2)
test_days = total_days - test_start
 
print(f"Total days in dataset: {total_days}")
print(f"Test period: days {test_start} to {total_days} ({test_days} days)")
print(f"Using final 20% of data for evaluation")

# Use test data for evaluation
# calcPL will evaluate on the LAST test_days of the provided price data (final 20%)
(meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll, test_days,stockIndex,headStart,nStocksFeedback)

score = meanpl - 0.1*plstd
print ("=====")
print ("mean(PL): %.1lf" % meanpl)
print ("return: %.5lf" % ret)
print ("StdDev(PL): %.2lf" % plstd)
print ("annSharpe(PL): %.2lf " % sharpe)
print ("totDvolume: %.0lf " % dvol)
print ("Score: %.2lf" % score)

if(openChart):
    plt.show()
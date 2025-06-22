#!/usr/bin/env python
#This is Eric's branch, the main algorithm is located inside test.py

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from test import getMyPosition as getPosition

##FEEDBACK VARIABLES

#Set if you want to look at the chart
openChart = True
#Opens up the graphs related to this stock
stockIndex = 46
#Change this to see a stock's price before the trading period begins
headStart = 500
#The amount of biggest losers/winners you want to see
nStocksFeedback = 5



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

    plt.figure(1)
    plt.title('Black = Stock Price, Orange = Position (scaled so that the max position is roughly equal to the average price of the stock)')
    plt.plot([i for i in range(startDay+1, nt+1)],positionHistory[stockIndex],color="orange", linewidth=1)
    plt.plot([i for i in range(startDay+1-headStart, nt+1)],prcHist[stockIndex,range(startDay-headStart, nt)],color="black", linewidth=1)
    plt.figure(2)
    plt.title('Green = Total Earnings, Red = Daily Earnings)')
    plt.plot([i for i in range(startDay+1, nt+1)],DailyPLHistory[stockIndex][:-1],color="red", linewidth=1)
    plt.plot([i for i in range(startDay+1, nt+1)],TotalPLHistory[stockIndex][:-2],color="green", linewidth=2)
    plt.plot([i for i in range(startDay+1, nt+1)],[0 for _ in range(startDay+1, nt+1)],color="grey", linewidth=1)

    biggestWinnersIndices = []
    biggestLosersIndices = []
    biggestWinnersPL = []
    biggestLosersPL = []
    FinalTotalPL = []
    for arr in TotalPLHistory:
        FinalTotalPL.append(arr[-1])
    

    for i in range(nStocksFeedback):
        temp = FinalTotalPL.index(max(FinalTotalPL))
        biggestWinnersIndices.append(temp)   
        biggestWinnersPL.append(int(FinalTotalPL[temp]))
        FinalTotalPL[temp] = 0
        temp = FinalTotalPL.index(min(FinalTotalPL))
        biggestLosersIndices.append(temp)
        biggestLosersPL.append(int(FinalTotalPL[temp]))
        FinalTotalPL[temp] = 0
    
    print(f"=====")
    print(f"Biggest Loser's indices: {biggestLosersIndices}")
    print(f"Biggest Loser's P/L: {biggestLosersPL}")
    print(f"Biggest Winners's indices: {biggestWinnersIndices}")
    print(f"Biggest Winners's P/L: {biggestWinnersPL}")
    

    if (plstd > 0):
        annSharpe = np.sqrt(249) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume)

# Calculate test period (80% to 100% of data)
total_days = nt
test_start = int(total_days * 0.8)
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
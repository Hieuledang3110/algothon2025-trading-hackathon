#!/usr/bin/env python

import numpy as np
import pandas as pd
from testing import getMyPosition as getPosition
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

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


def calcPL(prcHist, numTestDays):
    instrument_position_history = {i: [0] for i in range(50)}
    dvolume_history = {i: [0] for i in range(50)}
    instrument_PL_history = [[0] for _ in range(50)]
    instrument_PL_history_no_comm = [[0] for _ in range(50)]
    instrument_value_history = [[10000] for _ in range(50)]
    instrument_value_history_no_comm = [[10000] for _ in range(50)]
    instrument_cash_history = [[10000] for _ in range(50)]
    instrument_cash_history_no_comm = [[10000] for _ in range(50)]
    instrument_comm_history = [[0] for i in range(50)]
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0
    todayPLL = []
    (_,nt) = prcHist.shape
    startDay = nt + 1 - numTestDays
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
        else:
            newPos = np.array(curPos)
        for i in range(50):
            instrument_position_history[i].append(newPos[i])
            deltaPosition = instrument_position_history[i][-1]-instrument_position_history[i][-2]
            DollarVolume = deltaPosition*curPrices[i]
            dvolume_history[i].append(DollarVolume)
            commission = abs(DollarVolume*commRate)
            instrument_comm_history[i].append(commission)
            instrument_cash_history[i].append(instrument_cash_history[i][-1] - DollarVolume - commission)
            instrument_cash_history_no_comm[i].append(instrument_cash_history_no_comm[i][-1] - DollarVolume)
            instrument_value_history[i].append(instrument_cash_history[i][-1]+curPrices[i]*instrument_position_history[i][-1])
            instrument_value_history_no_comm[i].append(instrument_cash_history_no_comm[i][-1]+curPrices[i]*instrument_position_history[i][-1])
            instrument_PL_history[i].append(instrument_value_history[i][-1]-instrument_value_history[i][-2])
            instrument_PL_history_no_comm[i].append(instrument_value_history_no_comm[i][-1]-instrument_value_history_no_comm[i][-2])

        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue
        ret = 0.0
        if (totDVolume > 0):
            ret = value / totDVolume
        if (t > startDay):
            print ("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" % (t,value, todayPL, totDVolume, ret))
            todayPLL.append(todayPL)
    pll = np.array(todayPLL)
    (plmu,plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(249) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume,instrument_position_history,dvolume_history,instrument_PL_history,instrument_value_history,instrument_cash_history,instrument_comm_history,instrument_PL_history_no_comm)


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


def plot_single_instrument_analysis(instrument_id, 
                                  instrument_position_history,
                                  dvolume_history, 
                                  instrument_PL_history,
                                  instrument_value_history,
                                  instrument_cash_history,
                                  instrument_comm_history,
                                  price_history,
                                  start_day=0):
    
    # Extract data for the specific instrument
    positions = instrument_position_history[instrument_id]
    volumes = dvolume_history[instrument_id]
    pnl = instrument_PL_history[instrument_id]
    values = instrument_value_history[instrument_id]
    cash = instrument_cash_history[instrument_id]
    commissions = instrument_comm_history[instrument_id]
    prices = price_history[instrument_id]
    days = range(start_day, start_day + len(positions))
    fullDays = range(0,750)
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Instrument {instrument_id} - Complete Analysis', fontsize=16)
    
    # 1. Stock Price with P&L overlay (green/red background)
    ax1 = axes[0, 0]
    ax1.plot([i for i in range(start_day+1, start_day + len(positions))], prices[start_day:start_day + len(positions)], 'b-', linewidth=2, label='Price')
    
    # Color background based on P&L
    for i in range(len(days)-1):
        if i < len(pnl):
            if pnl[i] > 0.0001:
                color = 'green'
                ax1.axvspan(days[i-1], days[i], alpha=0.1, color=color)
            elif pnl[i] < -0.0001:
                color = 'red'
                ax1.axvspan(days[i-1], days[i], alpha=0.1, color=color)
            # No color when pnl[i] == 0 (skip the axvspan call entirely)
    
    ax1.set_title('Stock Price with P&L Overlay')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Individual P&L Chart
    ax2 = axes[0, 1]
    colors = ['green' if x >= 0 else 'red' for x in pnl]
    ax2.bar(days[:len(pnl)], pnl, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    ax2.set_title('Daily P&L')
    ax2.set_ylabel('P&L ($)')
    ax2.grid(True, alpha=0.3)
    
    
    # 3. Cumulative P&L
    ax3 = axes[1, 0]
    cumulative_pnl = np.cumsum(pnl)
    ax3.plot(days[:len(cumulative_pnl)], cumulative_pnl, 'g-', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Cumulative P&L')
    ax3.set_ylabel('Cumulative P&L ($)')
    ax3.grid(True, alpha=0.3)
    

    # 4. Position History
    ax4 = axes[1, 1]
    ax4.plot(days, positions, 'purple', linewidth=2, marker='o', markersize=2)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_title('Position History')
    ax4.set_ylabel('Position (shares)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Trading Volume
    ax5 = axes[2, 0]
    ax5.bar(days[:len(volumes)], volumes, alpha=0.7, color='orange')
    ax5.set_title('Daily Trading Volume')
    ax5.set_ylabel('Volume ($)')
    ax5.set_xlabel('Days')
    ax5.grid(True, alpha=0.3)
    
    # 6. Portfolio Value & Cash
    ax6 = axes[2, 1]
    ax6.plot(days[:len(values)], values, 'blue', label='Portfolio Value', linewidth=2)
    ax6.plot(days[:len(cash)], cash, 'green', label='Cash', linewidth=2)
    ax6.plot(days[:len(commissions)], np.cumsum(commissions), 'red', label='Cumulative Commissions', linewidth=1)
    ax6.set_title('Portfolio Metrics')
    ax6.set_ylabel('Value ($)')
    ax6.set_xlabel('Days')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Create second figure with technical analysis
    fig2, (ax_tech, ax_macd) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
    fig2.suptitle(f'Instrument {instrument_id} - Technical Analysis with PL', fontsize=16)

    # Plot stock price
    ax_tech.plot(fullDays, prices, 'b-', linewidth=2, label='Price')

    # Color background based on P&L excluding comissions
    for i in range(len(pnl)):
        if i < len(pnl):
            if pnl[i] > 0.0001:
                color = 'green'
                ax_tech.axvspan(start_day+i-2, start_day+i-1, alpha=0.1, color=color)
            elif pnl[i] < -0.0001:
                color = 'red'
                ax_tech.axvspan(start_day+i-2, start_day+i-1, alpha=0.1, color=color)

    # Add 30-day EMA
    if len(prices) >= 30:
        import pandas as pd
        ema_30 = pd.Series(prices).ewm(span=30).mean()
        sma_of_ema = pd.Series(ema_30).rolling(4).mean()
        sma_of_ema = sma_of_ema.values
        ax_tech.plot(range(len(sma_of_ema)), sma_of_ema, 'orange', label='4-day SMA of 30-day EMA', linewidth=2)

    # Add smooth predictions and bounds:
    smooth_predictions = []
    for i in range(0, start_day + len(positions)):
        smooth_predictions.append(smooth_trend_regression(prices[:i+1],long_window=200,smooth_window=5))

    if 'smooth_predictions' in locals() and len(smooth_predictions) > 0:
        pred_days = days[:len(smooth_predictions)]
        ax_tech.plot((([i for i in range(len(smooth_predictions))])), smooth_predictions, 'ro-', label='Smooth Predictions', markersize=2)
        
        
    # Calculate bounds
    percentage = 0.045  
    upper_bound = np.array(smooth_predictions) * (1 + percentage)
    lower_bound = np.array(smooth_predictions) * (1 - percentage)
    
    ax_tech.plot([i for i in range(len(smooth_predictions))], upper_bound, 'g--', label='Upper Bound', alpha=0.7)
    ax_tech.plot([i for i in range(len(smooth_predictions))], lower_bound, 'g--', label='Lower Bound', alpha=0.7)
    ax_tech.fill_between([i for i in range(len(smooth_predictions))], lower_bound, upper_bound, alpha=0.2, color='green')

    if len(prices) >= 30:
        sma_10 = pd.Series(prices).rolling(5).mean()
        sma_30 = pd.Series(prices).rolling(30).mean()
        ema_4_of_sma_10 = pd.Series(sma_10).ewm(span=4).mean()

        sma_10 = pd.Series(smooth_predictions).rolling(20).mean()
        ema_of_sma = pd.Series(sma_10).ewm(span=4).mean()
        ema_of_sma = ema_of_sma.values

        smoother_predictions = pd.Series(smooth_predictions).ewm(span=4).mean()
        smoother_predictions = smoother_predictions.values

        ax_tech.plot(range(len(ema_4_of_sma_10)), ema_4_of_sma_10, 'cyan', label='4-day EMA of 30-day SMA', linewidth=2)
        ax_tech.plot(range(len(sma_30)), sma_30, 'purple', label='30-day SMA', linewidth=2)

        # ax_tech.plot(range(len(ema_of_sma)), ema_of_sma, 'grey', label='ema_of_sma', linewidth=2)

        ax_tech.plot(range(len(smoother_predictions)), smoother_predictions, 'grey', label='4-day EMA of Smooth Predictions', linewidth=2)




    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD line, signal line, and histogram"""
        import pandas as pd
        
        prices_series = pd.Series(prices)
        
        # Calculate EMAs
        ema_fast = prices_series.ewm(span=fast).mean()
        ema_slow = prices_series.ewm(span=slow).mean()
        
        # MACD line = fast EMA - slow EMA
        macd_line = ema_fast - ema_slow
        
        # Signal line = EMA of MACD line
        signal_line = macd_line.ewm(span=signal).mean()
        
        # Histogram = MACD - Signal
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    # Calculate MACD
    macd_line, signal_line, histogram = calculate_macd(prices)

    # Plot MACD
    macd_days = range(len(macd_line))
    ax_macd.plot(macd_days, macd_line, 'blue', label='MACD Line', linewidth=1.5)
    ax_macd.plot(macd_days, signal_line, 'red', label='Signal Line', linewidth=1.5)

    # Plot histogram as bars
    colors = ['green' if h >= 0 else 'red' for h in histogram]
    ax_macd.bar(macd_days, histogram, color=colors, alpha=0.3, label='Histogram')

    # Add zero line
    ax_macd.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # MACD subplot formatting
    ax_macd.set_title('MACD (12, 26, 9)')
    ax_macd.set_xlabel('Days')
    ax_macd.set_ylabel('MACD')
    ax_macd.legend(loc='upper left')
    ax_macd.grid(True, alpha=0.3)


    ax_tech.set_title('Stock Price with Technical Indicators & P&L Overlay')
    ax_tech.set_xlabel('Days')
    ax_tech.set_ylabel('Price')
    ax_tech.legend()
    ax_tech.grid(True, alpha=0.3)
    
    # Print summary stats
    print(f"\n=== Instrument {instrument_id} Summary ===")
    print(f"Total P&L: ${sum(pnl):.2f}")
    print(f"Total Commissions: ${sum(commissions):.2f}")
    print(f"Total Volume Traded: ${sum(abs(v) for v in volumes):.2f}")
    print(f"Win Rate: {(np.array(pnl) > 0).sum() / len(pnl) * 100:.1f}%")
    print(f"Best Day: ${max(pnl):.2f}")
    print(f"Worst Day: ${min(pnl):.2f}")

    plt.tight_layout()
    


def analyze_winners_losers(instrument_PL_history, n=5):
    """
    Find the n biggest winners and losers by total P&L
    
    Args:
        instrument_PL_history: 2D array where [i][j] = instrument i, day j P&L
        n: number of top winners/losers to show
    """
    
    # Calculate total P&L for each instrument
    total_pnl = {}
    for instrument_id in range(len(instrument_PL_history)):
        pnl_list = instrument_PL_history[instrument_id]
        total_pnl[instrument_id] = sum(pnl_list)
    
    # Sort by total P&L
    sorted_instruments = sorted(total_pnl.items(), key=lambda x: x[1], reverse=True)
    
    # Get top winners and losers
    biggest_winners = sorted_instruments[:n]
    biggest_losers = sorted_instruments[-n:][::-1]  # Reverse to show worst first
    
    print(f"\n=== TOP {n} WINNERS ===")
    for i, (instrument_id, total_profit) in enumerate(biggest_winners, 1):
        print(f"{i}. Instrument {instrument_id}: ${total_profit:.2f}")
    
    print(f"\n=== TOP {n} LOSERS ===")
    for i, (instrument_id, total_loss) in enumerate(biggest_losers, 1):
        print(f"{i}. Instrument {instrument_id}: ${total_loss:.2f}")
    
    return biggest_winners, biggest_losers


def get_average_price(allPrices):
    averages = []
    for day in range(len(allPrices[0])):
        todaySum = 0
        for price in allPrices:
            todaySum += price[day]
        todayAverage = todaySum/len(allPrices)
        averages.append(todayAverage)
    return averages


# Calculate test period (80% to 100% of data)
total_days = nt
test_start = int(total_days * 0.8)  
test_days = total_days - test_start

print(f"Total days in dataset: {total_days}")
print(f"Test period: days {test_start} to {total_days} ({test_days} days)")
print(f"Using final 20% of data for evaluation")

# Use test data for evaluation
# calcPL will evaluate on the LAST test_days of the provided price data (final 20%)
print(prcAll)
(meanpl, ret, plstd, sharpe, dvol,instrument_position_history,dvolume_history,instrument_PL_history,instrument_value_history,instrument_cash_history,instrument_comm_history,instrument_PL_history_no_comm) = calcPL(prcAll, test_days)
    
winners, losers = analyze_winners_losers(instrument_PL_history, n=10)    

score = meanpl - 0.1*plstd
print ("=====")
print ("mean(PL): %.1lf" % meanpl)
print ("return: %.5lf" % ret)
print ("StdDev(PL): %.2lf" % plstd)
print ("annSharpe(PL): %.2lf " % sharpe)
print ("totDvolume: %.0lf " % dvol)
print ("Score: %.2lf" % score)

#Note that PL is shown without comission fees deducted                                                  vvvvvvv
stockID = 41
plot_single_instrument_analysis(stockID, instrument_position_history, dvolume_history, instrument_PL_history_no_comm, instrument_value_history,instrument_cash_history, instrument_comm_history,prcAll, start_day=test_start)



averages = get_average_price(prcAll)
plt.figure(100)
plt.plot(range(len(averages)),averages,label="Average") 
plt.title("Average") 

prcAllButOne = np.delete(prcAll, stockID, axis=0)
averagesAllButOne = get_average_price(prcAllButOne)
plt.plot(range(len(averagesAllButOne)),averagesAllButOne,label=f"Average without instrument{stockID}") 
plt.legend()

plt.show()
#!/usr/bin/env python

import numpy as np
import pandas as pd
from main import getMyPosition as getPosition
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
    """
    Enhanced plotting function with RSI MA and trend strength indicators
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Extract data for the specific instrument
    positions = instrument_position_history[instrument_id]
    volumes = dvolume_history[instrument_id]
    pnl = instrument_PL_history[instrument_id]
    values = instrument_value_history[instrument_id]
    cash = instrument_cash_history[instrument_id]
    commissions = instrument_comm_history[instrument_id]
    prices = price_history[instrument_id]
    days = range(start_day, start_day + len(positions))
    fullDays = range(0, len(prices))
    
    # Load trading algorithm parameters (same as in main.py)
    params = {
        "fast_ma": 10,
        "slow_ma": 30,
        "trend_strength": 20,
        "breakout_lookback": 20,
        "trend_threshold": 0.02,
        "stop_loss_pct": 0.10,
        "min_trend_duration": 5
    }
    
    # Calculate technical indicators used by the algorithm
    def calculate_moving_average_series(prices, period):
        """Calculate moving average for entire series"""
        ma_values = []
        for i in range(len(prices)):
            if i < period - 1:
                ma_values.append(np.nan)
            else:
                ma_values.append(np.mean(prices[max(0, i-period+1):i+1]))
        return np.array(ma_values)
    
    def calculate_trend_strength_series(prices, lookback_days):
        """Calculate trend strength for entire series"""
        trend_values = []
        for i in range(len(prices)):
            if i < lookback_days:
                trend_values.append(np.nan)
            else:
                start_price = prices[i - lookback_days]
                end_price = prices[i]
                if start_price == 0:
                    trend_values.append(0.0)
                else:
                    trend_strength = (end_price - start_price) / start_price
                    trend_values.append(trend_strength * 100)  # Convert to percentage
        return np.array(trend_values)
    
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        df = pd.DataFrame({'price': prices})
        rolling_mean = df['price'].rolling(window=period).mean()
        rolling_std = df['price'].rolling(window=period).std()
        
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        
        return rolling_mean.values, upper_band.values, lower_band.values
    
    def calculate_rsi_series(prices, period=14):
        """Calculate RSI for entire series"""
        df = pd.DataFrame({'price': prices})
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    def calculate_rsi_moving_average(rsi_values, ma_period=30):
        """Calculate moving average of RSI values"""
        df = pd.DataFrame({'rsi': rsi_values})
        rsi_ma = df['rsi'].rolling(window=ma_period).mean()
        return rsi_ma.values
    
    def calculate_trend_strength_moving_average(trend_values, ma_period=30):
        """Calculate moving average of trend strength values"""
        df = pd.DataFrame({'trend': trend_values})
        trend_ma = df['trend'].rolling(window=ma_period).mean()
        return trend_ma.values
    
    def calculate_support_resistance(prices, lookback=20):
        """Calculate support and resistance levels"""
        support_levels = []
        resistance_levels = []
        
        for i in range(len(prices)):
            if i < lookback:
                support_levels.append(np.nan)
                resistance_levels.append(np.nan)
            else:
                recent_prices = prices[max(0, i-lookback):i]
                support_levels.append(np.min(recent_prices))
                resistance_levels.append(np.max(recent_prices))
        
        return np.array(support_levels), np.array(resistance_levels)
    
    def detect_trading_signals(prices, params):
        """Calculate actual buy/sell signals using the EXACT algorithm logic from main.py"""
        
        # Recreate the functions from main.py
        def calculate_moving_average(prices, period):
            """Calculate simple moving average (from main.py)"""
            if len(prices) < period:
                period = len(prices)
            if period == 0:
                return prices[-1] if len(prices) > 0 else 0
            return np.mean(prices[-period:])

        def calculate_trend_strength(prices, lookback_days):
            """Calculate trend strength over lookback period (from main.py)"""
            if len(prices) < lookback_days + 1:
                lookback_days = len(prices) - 1
            if lookback_days < 2:
                return 0.0
            start_price = prices[-lookback_days-1]
            end_price = prices[-1]
            if start_price == 0:
                return 0.0
            trend_strength = (end_price - start_price) / start_price
            return trend_strength

        def detect_breakout(prices, lookback_days):
            """Detect if current price breaks out of recent range (from main.py)"""
            if len(prices) < lookback_days + 1:
                return 0
            recent_prices = prices[-lookback_days-1:-1]
            current_price = prices[-1]
            resistance = np.max(recent_prices)
            support = np.min(recent_prices)
            price_range = resistance - support
            if price_range == 0:
                return 0
            if current_price > resistance + 0.01 * price_range:
                return 1  # Bullish breakout
            elif current_price < support - 0.01 * price_range:
                return -1  # Bearish breakout
            return 0

        def check_trend_confirmation(fast_ma, slow_ma, trend_strength, breakout_signal, params):
            """Check if trend is confirmed and strong enough to trade (from main.py)"""
            # Moving average crossover signal
            ma_signal = 0
            if fast_ma > slow_ma:
                ma_signal = 1  # Uptrend
            elif fast_ma < slow_ma:
                ma_signal = -1  # Downtrend
            
            # Trend must be strong enough
            strong_trend = abs(trend_strength) > params["trend_threshold"]
            
            # Confirm trend direction matches moving averages
            trend_confirmed = False
            if ma_signal == 1 and trend_strength > 0:  # Both indicate uptrend
                trend_confirmed = True
            elif ma_signal == -1 and trend_strength < 0:  # Both indicate downtrend
                trend_confirmed = True
            
            # Add breakout confirmation if available
            if params.get("volume_confirmation", True) and breakout_signal != 0:
                if (ma_signal == 1 and breakout_signal == 1) or (ma_signal == -1 and breakout_signal == -1):
                    trend_confirmed = True
            
            return ma_signal if (trend_confirmed and strong_trend) else 0

        # Calculate signals for each day
        buy_signals = []
        sell_signals = []
        
        min_data_needed = max(params["fast_ma"], params["slow_ma"]) + 1
        
        for day in range(min_data_needed, len(prices)):
            # Get price history up to this day (simulating daily algorithm calls)
            day_prices = prices[:day+1]
            
            # Calculate indicators for this day
            fast_ma = calculate_moving_average(day_prices, params["fast_ma"])
            slow_ma = calculate_moving_average(day_prices, params["slow_ma"])
            trend_strength = calculate_trend_strength(day_prices, params["trend_strength"])
            breakout_signal = detect_breakout(day_prices, params["breakout_lookback"])
            
            # Get actual trading signal
            trend_signal = check_trend_confirmation(fast_ma, slow_ma, trend_strength, 
                                                  breakout_signal, params)
            
            # Record signal (this is when algorithm decides to buy/sell)
            if trend_signal == 1:  # Buy signal
                buy_signals.append(day)
            elif trend_signal == -1:  # Sell signal
                sell_signals.append(day)
        
        return buy_signals, sell_signals
    
    # Calculate ALL technical indicators
    fast_ma = calculate_moving_average_series(prices, params["fast_ma"])
    slow_ma = calculate_moving_average_series(prices, params["slow_ma"])
    trend_strength = calculate_trend_strength_series(prices, params["trend_strength"])
    trend_strength_ma = calculate_trend_strength_moving_average(trend_strength, 30)  # 30-day trend strength MA
    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(prices, 20, 2)
    rsi = calculate_rsi_series(prices, 14)
    rsi_ma = calculate_rsi_moving_average(rsi, 30)  # 30-day RSI moving average
    support, resistance = calculate_support_resistance(prices, params["breakout_lookback"])
    buy_signals, sell_signals = detect_trading_signals(prices, params)
    
    # Create subplots - KEEP ORIGINAL SUMMARY CHARTS THE SAME
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Instrument {instrument_id} - Complete Analysis', fontsize=16)
    
    # 1. Stock Price with P&L overlay (UNCHANGED)
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
    
    ax1.set_title('Stock Price with P&L Overlay')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Individual P&L Chart (UNCHANGED)
    ax2 = axes[0, 1]
    colors = ['green' if x >= 0 else 'red' for x in pnl]
    ax2.bar(days[:len(pnl)], pnl, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    ax2.set_title('Daily P&L')
    ax2.set_ylabel('P&L ($)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative P&L (UNCHANGED)
    ax3 = axes[1, 0]
    cumulative_pnl = np.cumsum(pnl)
    ax3.plot(days[:len(cumulative_pnl)], cumulative_pnl, 'g-', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Cumulative P&L')
    ax3.set_ylabel('Cumulative P&L ($)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Position History (UNCHANGED)
    ax4 = axes[1, 1]
    ax4.plot(days, positions, 'purple', linewidth=2, marker='o', markersize=2)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_title('Position History')
    ax4.set_ylabel('Position (shares)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Trading Volume (UNCHANGED)
    ax5 = axes[2, 0]
    ax5.bar(days[:len(volumes)], volumes, alpha=0.7, color='orange')
    ax5.set_title('Daily Trading Volume')
    ax5.set_ylabel('Volume ($)')
    ax5.set_xlabel('Days')
    ax5.grid(True, alpha=0.3)
    
    # 6. Portfolio Value & Cash (UNCHANGED)
    ax6 = axes[2, 1]
    ax6.plot(days[:len(values)], values, 'blue', label='Portfolio Value', linewidth=2)
    ax6.plot(days[:len(cash)], cash, 'green', label='Cash', linewidth=2)
    ax6.plot(days[:len(commissions)], np.cumsum(commissions), 'red', label='Cumulative Commissions', linewidth=1)
    ax6.set_title('Portfolio Metrics')
    ax6.set_ylabel('Value ($)')
    ax6.set_xlabel('Days')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # ============================================================================
    # ENHANCED TECHNICAL ANALYSIS CHART - WITH RSI MA AND TREND STRENGTH
    # ============================================================================
    fig2, (ax_main, ax_trend, ax_rsi) = plt.subplots(3, 1, figsize=(15, 16), 
                                                     gridspec_kw={'height_ratios': [3, 1, 1.2]}, 
                                                     sharex=True)
    fig2.suptitle(f'Instrument {instrument_id} - Enhanced Technical Analysis', fontsize=16)

    # Main price chart with algorithm indicators
    ax_main.plot(fullDays, prices, 'black', linewidth=2, label='Price', alpha=0.8)

    # Moving averages used by algorithm
    valid_fast = ~np.isnan(fast_ma)
    valid_slow = ~np.isnan(slow_ma)
    ax_main.plot(np.array(fullDays)[valid_fast], fast_ma[valid_fast], 
                'blue', label=f'Fast MA ({params["fast_ma"]})', linewidth=2)
    ax_main.plot(np.array(fullDays)[valid_slow], slow_ma[valid_slow], 
                'red', label=f'Slow MA ({params["slow_ma"]})', linewidth=2)

    # Bollinger Bands - solid lines for better visibility
    valid_bb = ~np.isnan(bb_middle)
    ax_main.plot(np.array(fullDays)[valid_bb], bb_middle[valid_bb], 
                'gray', label='BB Middle (20)', linewidth=1.5, linestyle='-')
    ax_main.plot(np.array(fullDays)[valid_bb], bb_upper[valid_bb], 
                'orange', label='BB Upper (2σ)', linewidth=1.5, linestyle='-')
    ax_main.plot(np.array(fullDays)[valid_bb], bb_lower[valid_bb], 
                'orange', label='BB Lower (2σ)', linewidth=1.5, linestyle='-')

    # Support and Resistance levels used for breakout detection
    valid_support = ~np.isnan(support)
    valid_resistance = ~np.isnan(resistance)
    ax_main.plot(np.array(fullDays)[valid_support], support[valid_support], 
                'green', label=f'Support ({params["breakout_lookback"]}d)', 
                linewidth=1, linestyle=':', alpha=0.7)
    ax_main.plot(np.array(fullDays)[valid_resistance], resistance[valid_resistance], 
                'red', label=f'Resistance ({params["breakout_lookback"]}d)', 
                linewidth=1, linestyle=':', alpha=0.7)

    # Buy/Sell signal day shading (light shading covering entire days)
    for buy_day in buy_signals:
        ax_main.axvspan(buy_day-0.5, buy_day+0.5, color='darkgreen', alpha=0.2,
                       label='Buy Signal Day' if buy_day == buy_signals[0] else "")
    
    for sell_day in sell_signals:
        ax_main.axvspan(sell_day-0.5, sell_day+0.5, color='darkred', alpha=0.2,
                       label='Sell Signal Day' if sell_day == sell_signals[0] else "")

    ax_main.set_title('Price with Trading Algorithm Indicators')
    ax_main.set_ylabel('Price')
    ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_main.grid(True, alpha=0.3)

    # Trend Strength subplot with 30-day moving average
    valid_trend = ~np.isnan(trend_strength)
    valid_trend_ma = ~np.isnan(trend_strength_ma)
    
    ax_trend.plot(np.array(fullDays)[valid_trend], trend_strength[valid_trend], 
                 'purple', linewidth=2, label=f'Trend Strength ({params["trend_strength"]}d)')
    ax_trend.plot(np.array(fullDays)[valid_trend_ma], trend_strength_ma[valid_trend_ma], 
                 'darkblue', linewidth=2, label='Trend Strength MA (30)', linestyle='--')
    
    ax_trend.axhline(y=params["trend_threshold"]*100, color='green', linestyle='--', alpha=0.7, 
                    label=f'Threshold (+{params["trend_threshold"]*100:.1f}%)')
    ax_trend.axhline(y=-params["trend_threshold"]*100, color='red', linestyle='--', alpha=0.7,
                    label=f'Threshold (-{params["trend_threshold"]*100:.1f}%)')
    ax_trend.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # Add buy/sell signal day shading to trend strength chart
    for buy_day in buy_signals:
        ax_trend.axvspan(buy_day-0.5, buy_day+0.5, color='darkgreen', alpha=0.15)
    for sell_day in sell_signals:
        ax_trend.axvspan(sell_day-0.5, sell_day+0.5, color='darkred', alpha=0.15)
    
    ax_trend.set_title('Trend Strength with 30-day Moving Average')
    ax_trend.set_ylabel('Trend %')
    ax_trend.legend()
    ax_trend.grid(True, alpha=0.3)

    # Enhanced RSI subplot with 30-day moving average
    valid_rsi = ~np.isnan(rsi)
    valid_rsi_ma = ~np.isnan(rsi_ma)
    
    ax_rsi.plot(np.array(fullDays)[valid_rsi], rsi[valid_rsi], 'purple', linewidth=2, label='RSI (14)')
    ax_rsi.plot(np.array(fullDays)[valid_rsi_ma], rsi_ma[valid_rsi_ma], 'darkblue', linewidth=2, 
               label='RSI MA (30)', linestyle='--')
    
    ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax_rsi.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    ax_rsi.fill_between(fullDays, 30, 70, alpha=0.1, color='gray')
    
    # Add buy/sell signal day shading to RSI chart
    for buy_day in buy_signals:
        ax_rsi.axvspan(buy_day-0.5, buy_day+0.5, color='darkgreen', alpha=0.15)
    for sell_day in sell_signals:
        ax_rsi.axvspan(sell_day-0.5, sell_day+0.5, color='darkred', alpha=0.15)
    
    ax_rsi.set_title('RSI with 30-day Moving Average')
    ax_rsi.set_xlabel('Days')
    ax_rsi.set_ylabel('RSI')
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend()
    ax_rsi.grid(True, alpha=0.3)
    
    # Print algorithm-specific summary stats
    print(f"\n=== Instrument {instrument_id} Enhanced Algorithm Analysis ===")
    print(f"Total P&L: ${sum(pnl):.2f}")
    print(f"Algorithm Buy Signals: {len(buy_signals)} on days: {buy_signals}")
    print(f"Algorithm Sell Signals: {len(sell_signals)} on days: {sell_signals}")
    print(f"Fast MA Period: {params['fast_ma']} days")
    print(f"Slow MA Period: {params['slow_ma']} days")
    print(f"Trend Strength Period: {params['trend_strength']} days")
    print(f"Trend Threshold: ±{params['trend_threshold']*100:.1f}%")
    print(f"Stop Loss: {params['stop_loss_pct']*100:.1f}%")
    print(f"Min Hold Period: {params['min_trend_duration']} days")
    print(f"Breakout Lookback: {params['breakout_lookback']} days")
    print(f"RSI MA Period: 30 days")
    print(f"Trend Strength MA Period: 30 days")
    print(f"Total Commissions: ${sum(commissions):.2f}")
    print(f"Win Rate: {(np.array(pnl) > 0).sum() / len(pnl) * 100:.1f}%")
    print(f"Best Day: ${max(pnl):.2f}")
    print(f"Worst Day: ${min(pnl):.2f}")

    plt.tight_layout()
    return fig, fig2

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
test_start = int(total_days * 0.1)  
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
stockID = 3
plot_single_instrument_analysis(stockID, instrument_position_history, dvolume_history, instrument_PL_history_no_comm, instrument_value_history,instrument_cash_history, instrument_comm_history,prcAll, start_day=test_start)


plt.show()

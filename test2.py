#Eric is using this for testing, hoping to start afresh since the other test file has become a clusterfuck
 
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


#Test function to check if the P/L feedback is working
def alwaysBuy(data):
    positions = [100000 for _ in range(50)]
    return positions

#Test function to check if the P/L feedback is working
def alwaysSell(data):
    positions = [-100000 for _ in range(50)]
    return positions

def macd_ema_sma_slope_strategy(prcAll, max_position_value=8000, min_slope_threshold=0.0007):
    """
    MACD + EMA-SMA strategy: No trades when slope of 4-day SMA of 30-day EMA is insignificant
    
    Args:
        prcAll: 2D array of price histories [instrument][day]
        max_position_value: maximum dollar position per stock
        min_slope_threshold: minimum normalized slope to allow trading
    
    Returns:
        positions: array of positions for each instrument
    """
    
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD components"""
        if len(prices) < slow:
            return None, None, None
        
        import pandas as pd
        prices_series = pd.Series(prices)
        
        # Calculate EMAs
        ema_fast = prices_series.ewm(span=fast).mean()
        ema_slow = prices_series.ewm(span=slow).mean()
        
        # MACD line and signal line
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        
        return macd_line.values, signal_line.values, None
    
    def calculate_ema_sma_slope(prices, ema_period=30, sma_period=4, slope_window=5):
        """Calculate slope significance of 4-day SMA of 30-day EMA"""
        if len(prices) < ema_period + sma_period + slope_window:
            return 0, False  # direction, is_significant
        
        import pandas as pd
        
        # Step 1: Calculate 30-day EMA
        ema_30 = pd.Series(prices).ewm(span=ema_period).mean()
        
        # Step 2: Calculate 4-day SMA of the EMA
        sma_of_ema = ema_30.rolling(sma_period).mean().dropna()
        
        if len(sma_of_ema) < slope_window:
            return 0, False
        
        # Step 3: Calculate slope using linear regression over last 5 points
        recent_values = sma_of_ema.iloc[-slope_window:].values
        x_values = np.arange(len(recent_values))
        
        # Linear regression to get slope
        slope = np.polyfit(x_values, recent_values, 1)[0]
        
        # Normalize slope by average value to make it comparable across different price levels
        avg_value = np.mean(recent_values)
        normalized_slope = abs(slope) / avg_value if avg_value > 0 else 0
        
        # Determine if slope is significant
        is_significant = normalized_slope > min_slope_threshold
        
        # Return direction and significance
        direction = 1 if slope > 0 else (-1 if slope < 0 else 0)
        
        return direction, is_significant
    
    positions = []
    
    for i in range(len(prcAll)):
        prices = prcAll[i]
        current_price = prices[-1]
        
        if len(prices) < 40:  # Need 30 for EMA + 4 for SMA + 5 for slope + buffer
            positions.append(0)
            continue
        
        # Calculate MACD
        macd_line, signal_line, _ = calculate_macd(prices)
        
        if macd_line is None or len(macd_line) < 2:
            positions.append(0)
            continue
        
        # Calculate EMA-SMA slope and significance
        ema_sma_direction, slope_is_significant = calculate_ema_sma_slope(prices)
        
        # Skip trading if slope is not significant
        if not slope_is_significant:
            positions.append(0)
            continue
        
        # Current and previous MACD values
        macd_current = macd_line[-1]
        macd_prev = macd_line[-2]
        signal_current = signal_line[-1]
        signal_prev = signal_line[-2]
        
        # Detect crossovers
        macd_crossed_above = (macd_prev <= signal_prev) and (macd_current > signal_current)
        macd_crossed_below = (macd_prev >= signal_prev) and (macd_current < signal_current)
        
        # Position logic with significant slope filter
        position = 0
        
        if (macd_crossed_above and macd_current < 0 and signal_current < 0 
            and ema_sma_direction >= 0):  # Only buy if EMA-SMA trending up/flat with significant slope
            # Buy signal
            position_value = max_position_value
            position = int(position_value / current_price) if current_price > 0 else 0
            
        elif (macd_crossed_below and macd_current > 0 and signal_current > 0 
              and ema_sma_direction <= 0):  # Only sell if EMA-SMA trending down/flat with significant slope
            # Sell signal
            position_value = max_position_value
            position = -int(position_value / current_price) if current_price > 0 else 0
            
        # Hold position based on current state (with significant slope requirement)
        elif (macd_current > signal_current and macd_current < 0 
              and ema_sma_direction >= 0):
            # Continue holding long
            position_value = max_position_value * 0.8
            position = int(position_value / current_price) if current_price > 0 else 0
            
        elif (macd_current < signal_current and macd_current > 0 
              and ema_sma_direction <= 0):
            # Continue holding short
            position_value = max_position_value * 0.8
            position = -int(position_value / current_price) if current_price > 0 else 0
        
        positions.append(position)
    
    return np.array(positions)

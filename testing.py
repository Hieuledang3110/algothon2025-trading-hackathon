import numpy as np
import json

# Global variable to track positions and trends
position_tracker = {}

def load_parameters():
    """Load trading model hyperparameters from JSON file"""
    try:
        with open('model_parameters.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Default parameters for trend following strategy
        return {
            "fast_ma": 10,                # Fast moving average period
            "slow_ma": 30,                # Slow moving average period
            "trend_strength": 20,         # Period to confirm trend strength
            "position_limit": 100000,      # Maximum dollar position per instrument
            "min_trend_duration": 5,      # Minimum days to hold position
            "trend_threshold": 0.02,      # Minimum trend strength (2%)
            "stop_loss_pct": 0.10,        # Stop loss percentage (8%)
            "volume_confirmation": True,   # Use volume-like confirmation
            "breakout_lookback": 20       # Days to look back for breakout levels
        }

def calculate_moving_average(prices, period):
    """Calculate simple moving average"""
    if len(prices) < period:
        period = len(prices)  # Use all available data
    
    if period == 0:
        return prices[-1] if len(prices) > 0 else 0
    
    return np.mean(prices[-period:])  # Average of last N prices

def calculate_trend_strength(prices, lookback_days):
    """Calculate trend strength over lookback period"""
    if len(prices) < lookback_days + 1:
        lookback_days = len(prices) - 1
    
    if lookback_days < 2:
        return 0.0
    
    # Calculate price change over the period
    start_price = prices[-lookback_days-1]
    end_price = prices[-1]
    
    if start_price == 0:
        return 0.0
    
    # Trend strength as percentage change
    trend_strength = (end_price - start_price) / start_price
    return trend_strength

def detect_breakout(prices, lookback_days):
    """Detect if current price breaks out of recent range"""
    if len(prices) < lookback_days + 1:
        return 0  # No breakout signal
    
    # Get recent price range (excluding current price)
    recent_prices = prices[-lookback_days-1:-1]
    current_price = prices[-1]
    
    # Calculate support and resistance levels
    resistance = np.max(recent_prices)
    support = np.min(recent_prices)
    price_range = resistance - support
    
    if price_range == 0:
        return 0
    
    # Breakout signals
    if current_price > resistance + 0.01 * price_range:  # Break above resistance
        return 1  # Bullish breakout
    elif current_price < support - 0.01 * price_range:   # Break below support
        return -1  # Bearish breakout
    
    return 0  # No breakout

def check_trend_confirmation(fast_ma, slow_ma, trend_strength, breakout_signal, params):
    """Check if trend is confirmed and strong enough to trade"""
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

def check_position_duration(instrument_id, current_day, params):
    """Check if minimum position duration has been met"""
    global position_tracker
    
    if instrument_id not in position_tracker:
        return True  # No previous position, OK to trade
    
    last_trade_day = position_tracker[instrument_id].get('last_trade_day', 0)
    days_held = current_day - last_trade_day
    
    return days_held >= params["min_trend_duration"]

def check_stop_loss(instrument_id, current_price, current_position, params):
    """Check if stop loss should be triggered"""
    global position_tracker
    
    if instrument_id not in position_tracker:
        position_tracker[instrument_id] = {
            'position': 0, 
            'entry_price': 0, 
            'last_trade_day': 0
        }
    
    tracker = position_tracker[instrument_id]
    prev_position = tracker['position']
    entry_price = tracker['entry_price']
    
    # If position changed from 0, record new entry
    if prev_position == 0 and current_position != 0:
        tracker.update({
            'position': current_position,
            'entry_price': current_price,
            'last_trade_day': 0  # Will be updated in main function
        })
        return False
    
    # Check stop loss for existing positions
    if prev_position != 0 and entry_price > 0:
        price_change_pct = (current_price - entry_price) / entry_price
        
        # Stop loss conditions
        if ((prev_position > 0 and price_change_pct <= -params["stop_loss_pct"]) or
            (prev_position < 0 and price_change_pct >= params["stop_loss_pct"])):
            
            # Reset tracker
            tracker.update({'position': 0, 'entry_price': 0, 'last_trade_day': 0})
            return True
    
    # Update position
    tracker['position'] = current_position
    return False

def calculate_position_size(signal_strength, current_price, params):
    """Calculate position size based on signal strength"""
    if signal_strength == 0:
        return 0
    
    # Calculate maximum shares we can afford
    max_shares = int(params["position_limit"] / current_price)
    
    # Use a fixed percentage of max position for trend following
    # This reduces over-leveraging and provides more stable returns
    position_fraction = 0.6  # Use 60% of maximum position
    base_position = int(max_shares * position_fraction)
    
    # Scale by signal direction
    return base_position * signal_strength

def getMyPosition(price_data):
    """
    Main trading function implementing trend following strategy
    
    Args:
        price_data: numpy array of shape (n_instruments, n_days) containing price history
        
    Returns:
        numpy array of position sizes for each instrument
    """
    global position_tracker
    
    params = load_parameters()           # Load hyperparameters
    n_instruments = price_data.shape[0]  # Number of instruments
    n_days = price_data.shape[1]         # Number of days
    positions = np.zeros(n_instruments)  # Initialize positions
    
    # Current day for position duration tracking
    current_day = n_days - 1
    
    # Process each instrument
    for i in range(n_instruments):
        instrument_prices = price_data[i]  # Price history for this instrument
        current_price = instrument_prices[-1]  # Current price
        
        # Skip if insufficient data
        if len(instrument_prices) < max(params["fast_ma"], params["slow_ma"]) + 1:
            positions[i] = 0
            continue
        
        # Calculate technical indicators
        fast_ma = calculate_moving_average(instrument_prices, params["fast_ma"])
        slow_ma = calculate_moving_average(instrument_prices, params["slow_ma"])
        trend_strength = calculate_trend_strength(instrument_prices, params["trend_strength"])
        breakout_signal = detect_breakout(instrument_prices, params["breakout_lookback"])
        
        # Get trend confirmation signal
        trend_signal = check_trend_confirmation(fast_ma, slow_ma, trend_strength, 
                                              breakout_signal, params)
        
        # Check if we can make a new trade (respect minimum holding period)
        can_trade = check_position_duration(i, current_day, params)
        
        # Determine position
        if trend_signal != 0 and can_trade:
            # Calculate position size
            position_size = calculate_position_size(trend_signal, current_price, params)
            
            # Check stop loss
            if check_stop_loss(i, current_price, position_size, params):
                positions[i] = 0  # Exit due to stop loss
            else:
                positions[i] = position_size
                # Update last trade day
                if i in position_tracker:
                    position_tracker[i]['last_trade_day'] = current_day
        else:
            # Hold existing position or check stop loss
            current_pos = position_tracker.get(i, {}).get('position', 0)
            if check_stop_loss(i, current_price, current_pos, params):
                positions[i] = 0  # Exit due to stop loss
            else:
                positions[i] = current_pos  # Hold current position
    
    return positions.astype(int)  # Return integer positions
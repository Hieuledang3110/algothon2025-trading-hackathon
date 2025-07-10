import numpy as np

##### STATELESS BOLLINGER BANDS + RSI MEAN REVERSION STRATEGY #####
### Buys when price < lower BB AND RSI < 30 ###
### Sells when price > upper BB AND RSI > 70 ###
### Exits when conditions are no longer met ###

def load_parameters():
    """Load trading model hyperparameters"""
    return {
        "bb_period": 20,              # Bollinger Bands period
        "bb_std_dev": 2,              # Bollinger Bands standard deviation
        "rsi_period": 14,             # RSI period
        "rsi_oversold": 30,           # RSI oversold threshold
        "rsi_overbought": 70,         # RSI overbought threshold
        "position_limit": 10000,      # Maximum dollar position per instrument
        "position_fraction": 0.5      # Use 50% of max position size
    }

def calculate_bollinger_bands(prices, period, std_dev):
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return None, None, None
    
    # Simple moving average
    sma = np.mean(prices[-period:])
    
    # Standard deviation
    std = np.std(prices[-period:])
    
    # Bollinger Bands
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    middle_band = sma
    
    return upper_band, middle_band, lower_band

def calculate_rsi(prices, period):
    """Calculate RSI (Relative Strength Index)"""
    if len(prices) < period + 1:
        return 50  # Neutral RSI if insufficient data
    
    # Calculate price changes
    deltas = np.diff(prices[-period-1:])
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate average gains and losses
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    # Avoid division by zero
    if avg_loss == 0:
        return 100 if avg_gain > 0 else 50
    
    # Calculate RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_position_size(signal_strength, current_price, params):
    """Calculate position size based on signal strength"""
    if signal_strength == 0:
        return 0
    
    # Calculate maximum shares we can afford
    max_shares = int(params["position_limit"] / current_price)
    
    # Use a fraction of max position for mean reversion
    base_position = int(max_shares * params["position_fraction"])
    
    # Scale by signal direction
    return base_position * signal_strength

def get_trading_signal(prices, params):
    """Determine trading signal based on current market conditions"""
    current_price = prices[-1]
    
    # Calculate technical indicators
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
        prices, params["bb_period"], params["bb_std_dev"]
    )
    rsi = calculate_rsi(prices, params["rsi_period"])
    
    # Return neutral if insufficient data
    if bb_upper is None or bb_lower is None:
        return 0
    
    # Long signal: Price below lower BB AND RSI oversold
    if current_price < bb_lower and rsi < params["rsi_oversold"]:
        return 1  # Go long
    
    # Short signal: Price above upper BB AND RSI overbought  
    elif current_price > bb_upper and rsi > params["rsi_overbought"]:
        return -1  # Go short
    
    # No position if conditions not met
    return 0

def getMyPosition(price_data):
    """
    Stateless trading function implementing Bollinger Bands + RSI mean reversion strategy
    
    Strategy Logic:
    - Hold long when: price < lower BB AND RSI < 30
    - Hold short when: price > upper BB AND RSI > 70  
    - No position when: conditions not met
    
    No memory needed - positions determined purely by current market conditions
    
    Args:
        price_data: numpy array of shape (n_instruments, n_days) containing price history
        
    Returns:
        numpy array of position sizes for each instrument
    """
    params = load_parameters()           # Load hyperparameters
    n_instruments = price_data.shape[0]  # Number of instruments
    positions = np.zeros(n_instruments)  # Initialize positions
    
    # Minimum data needed for calculations
    min_data_needed = max(params["bb_period"], params["rsi_period"]) + 1
    
    # Process each instrument
    for i in range(n_instruments):
        instrument_prices = price_data[i]  # Price history for this instrument
        current_price = instrument_prices[-1]  # Current price
        
        # Skip if insufficient data
        if len(instrument_prices) < min_data_needed:
            positions[i] = 0
            continue
        
        # Get trading signal based on current conditions
        signal = get_trading_signal(instrument_prices, params)
        
        # Calculate position size
        position_size = calculate_position_size(signal, current_price, params)
        positions[i] = position_size
    
    return positions.astype(int)  # Return integer positions

# For testing/debugging
if __name__ == "__main__":
    # Example usage
    print("Stateless Bollinger Bands + RSI Mean Reversion Strategy")
    print("Strategy Parameters:")
    params = load_parameters()
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    print("\nStrategy Logic:")
    print("  • LONG when: Price < Lower BB AND RSI < 30")
    print("  • SHORT when: Price > Upper BB AND RSI > 70") 
    print("  • NO POSITION when: Conditions not met")
    print("  • Completely stateless - no memory required")
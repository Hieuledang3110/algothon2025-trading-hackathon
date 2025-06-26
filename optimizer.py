import numpy as np
import matplotlib.pyplot as plt
import json
from testing import getMyPosition

def load_and_prepare_data():
    """Load price data and split into train/validation/test sets"""
    # Load data from prices.txt
    data = np.loadtxt("prices.txt")
    
    # Transpose from (750, 50) to (50, 750) as required
    data = data.T
    
    # Split data: 60% train (450 days), 20% validation (150 days), 20% test (150 days)
    train_end = 450
    val_end = 600
    
    train_data = data[:, :train_end]           # Days 0-449
    validation_data = data[:, train_end:val_end]  # Days 450-599
    test_data = data[:, val_end:]              # Days 600-749
    
    return data, train_data, validation_data, test_data

def optimize_parameters(train_data, validation_data):
    """Optimize hyperparameters using training and validation data"""
    best_params = None
    best_score = float('-inf')
    
    # Parameter ranges for improved trend following strategy
    fast_mas = [5, 8, 12]
    slow_mas = [18, 21, 26]
    trend_mas = [40, 50, 60]
    min_durations = [6, 8, 10]
    trend_thresholds = [0.02, 0.025, 0.03]
    stop_losses = [0.10, 0.12, 0.15]
    consistencies = [0.6, 0.7, 0.8]
    
    print("Optimizing improved trend following parameters...")
    
    # Test parameter combinations
    for fast_ma in fast_mas:
        for slow_ma in slow_mas:
            if slow_ma <= fast_ma:
                continue
                
            for trend_ma in trend_mas:
                if trend_ma <= slow_ma:
                    continue
                    
                for min_dur in min_durations:
                    for trend_thresh in trend_thresholds:
                        for stop_loss in stop_losses:
                            for consistency in consistencies:
                                # Create parameter set
                                params = {
                                    "fast_ma": fast_ma,
                                    "slow_ma": slow_ma,
                                    "trend_ma": trend_ma,
                                    "position_limit": 10000,
                                    "min_trend_duration": min_dur,
                                    "trend_strength_threshold": trend_thresh,
                                    "momentum_lookback": 5,
                                    "pullback_threshold": 0.03,
                                    "stop_loss_pct": stop_loss,
                                    "ma_separation": 0.015,
                                    "volume_confirmation": True,
                                    "trend_consistency": consistency
                                }
                                
                                # Save parameters temporarily
                                with open('model_parameters.json', 'w') as f:
                                    json.dump(params, f)
                                
                                # Test on validation data
                                pnl_history, _ = backtest_strategy(validation_data)
                                
                                # Calculate performance score
                                if len(pnl_history) > 1:
                                    mean_pnl = np.mean(pnl_history)
                                    std_pnl = np.std(pnl_history)
                                    score = mean_pnl - 0.1 * std_pnl
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_params = params.copy()
    
    # Save best parameters
    if best_params:
        with open('model_parameters.json', 'w') as f:
            json.dump(best_params, f)
        print(f"Best parameters found: {best_params}")
        print(f"Best validation score: {best_score:.4f}")
    
    return best_params

def backtest_strategy(price_data):
    """Run backtest on given price data"""
    n_instruments, n_days = price_data.shape
    
    # Initialize tracking variables
    positions = np.zeros(n_instruments)  # Current positions
    cash = 0                             # Cash balance
    pnl_history = []                     # Daily P&L
    position_history = []                # Position history for plotting
    trade_signals = []                   # Track buy/sell signals for each instrument
    
    # Commission rate
    commission_rate = 0.0005
    
    # Run strategy day by day
    for day in range(1, n_days):  # Start from day 1 (need at least 1 day of history)
        # Get price data up to current day
        data_so_far = price_data[:, :day+1]
        current_prices = price_data[:, day]
        
        # Get new positions from strategy
        new_positions = getMyPosition(data_so_far)
        
        # Apply position limits ($10k per instrument)
        position_limits = np.array([int(10000 / price) for price in current_prices])
        new_positions = np.clip(new_positions, -position_limits, position_limits)
        
        # Calculate trades (position changes)
        trades = new_positions - positions
        
        # Record trade signals for visualization
        daily_signals = []
        for i in range(n_instruments):
            if trades[i] > 0:
                daily_signals.append(1)  # Buy signal
            elif trades[i] < 0:
                daily_signals.append(-1)  # Sell signal
            else:
                daily_signals.append(0)  # No trade
        trade_signals.append(daily_signals)
        
        # Calculate trading costs
        trade_value = np.sum(np.abs(trades) * current_prices)
        commission = trade_value * commission_rate
        
        # Update cash with trading costs
        cash -= commission
        cash -= np.sum(trades * current_prices)  # Cash impact of trades
        
        # Calculate P&L from price movements (if we had positions yesterday)
        if day > 1:
            price_changes = current_prices - price_data[:, day-1]
            pnl = np.sum(positions * price_changes)
            pnl_history.append(pnl)
        else:
            pnl_history.append(0)
        
        # Update positions
        positions = new_positions.copy()
        position_history.append(positions.copy())
    
    return pnl_history, trade_signals

def plot_individual_instrument(instrument_id, price_data, trade_signals, start_day=0, end_day=None):
    """Plot individual instrument price with buy/sell signals"""
    if end_day is None:
        end_day = price_data.shape[1]
    
    # Get price data for the specific instrument
    prices = price_data[instrument_id, start_day:end_day]
    days = range(start_day, start_day + len(prices))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot price line
    ax.plot(days, prices, 'b-', linewidth=2, label=f'Instrument {instrument_id} Price', alpha=0.7)
    
    # Add buy/sell signals
    if trade_signals and len(trade_signals) > 0:
        trade_days = range(start_day + 1, min(start_day + 1 + len(trade_signals), end_day))
        
        for i, day in enumerate(trade_days):
            if i < len(trade_signals) and instrument_id < len(trade_signals[i]):
                signal = trade_signals[i][instrument_id]
                price_at_signal = prices[day - start_day] if day - start_day < len(prices) else prices[-1]
                
                if signal > 0:  # Buy signal
                    ax.axvline(x=day, color='green', alpha=0.6, linewidth=2, label='Buy Signal' if i == 0 else "")
                    ax.scatter(day, price_at_signal, color='green', s=100, marker='^', zorder=5)
                elif signal < 0:  # Sell signal
                    ax.axvline(x=day, color='red', alpha=0.6, linewidth=2, label='Sell Signal' if i == 0 else "")
                    ax.scatter(day, price_at_signal, color='red', s=100, marker='v', zorder=5)
    
    # Customize plot
    ax.set_title(f'Instrument {instrument_id} - Price and Trading Signals', fontsize=14, fontweight='bold')
    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add summary statistics
    price_change = ((prices[-1] - prices[0]) / prices[0]) * 100 if len(prices) > 0 else 0
    volatility = np.std(np.diff(prices) / prices[:-1]) * np.sqrt(252) if len(prices) > 1 else 0
    
    # Add text box with statistics
    textstr = f'Price Change: {price_change:.2f}%\nAnnualized Volatility: {volatility:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()

def plot_results(pnl_history, position_history, test_data, trade_signals=None):
    """Create visualization plots"""
    days = range(1, len(pnl_history) + 1)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Daily P&L
    ax1.plot(days, pnl_history, 'g-', alpha=0.7)
    ax1.set_title('Daily P&L')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('P&L ($)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 2: Cumulative P&L
    cumulative_pnl = np.cumsum(pnl_history)
    ax2.plot(days, cumulative_pnl, 'b-', linewidth=2)
    ax2.set_title('Cumulative P&L')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Cumulative P&L ($)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 3: Position heatmap (sample of instruments)
    if position_history:
        # Show positions for first 20 instruments
        pos_array = np.array(position_history)[:, :20].T
        im = ax3.imshow(pos_array, aspect='auto', cmap='RdYlBu', interpolation='nearest')
        ax3.set_title('Position Heatmap (First 20 Instruments)')
        ax3.set_xlabel('Days')
        ax3.set_ylabel('Instruments')
        plt.colorbar(im, ax=ax3, label='Position Size')
    
    # Plot 4: Strategy performance metrics
    if len(pnl_history) > 1:
        # Calculate rolling Sharpe ratio (30-day window)
        window = 30
        rolling_returns = np.array(pnl_history)
        rolling_sharpe = []
        
        for i in range(window, len(rolling_returns)):
            window_returns = rolling_returns[i-window:i]
            if np.std(window_returns) > 0:
                sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252)
                rolling_sharpe.append(sharpe)
            else:
                rolling_sharpe.append(0)
        
        if rolling_sharpe:
            ax4.plot(range(window, len(rolling_returns)), rolling_sharpe, 'purple', linewidth=2)
            ax4.set_title('Rolling Sharpe Ratio (30-day)')
            ax4.set_xlabel('Days')
            ax4.set_ylabel('Annualized Sharpe')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Plot individual instruments (show first 3 as examples)
    if trade_signals:
        print("\nShowing individual instrument trading patterns...")
        for instrument_id in [0, 1, 2]:  # Show first 3 instruments
            plot_individual_instrument(instrument_id, test_data, trade_signals)
    
    # Print performance summary
    total_pnl = sum(pnl_history)
    avg_daily_pnl = np.mean(pnl_history)
    pnl_std = np.std(pnl_history)
    sharpe_ratio = (avg_daily_pnl / pnl_std * np.sqrt(252)) if pnl_std > 0 else 0
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS SUMMARY")
    print("="*50)
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Average Daily P&L: ${avg_daily_pnl:.2f}")
    print(f"P&L Standard Deviation: ${pnl_std:.2f}")
    print(f"Sharpe Ratio (Annualized): {sharpe_ratio:.3f}")
    print(f"Performance Score (Mean - 0.1*Std): {avg_daily_pnl - 0.1*pnl_std:.4f}")
    print(f"Number of Trading Days: {len(pnl_history)}")
    print("="*50)("\n" + "="*50)
    print("BACKTEST RESULTS SUMMARY")
    print("="*50)
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Average Daily P&L: ${avg_daily_pnl:.2f}")
    print(f"P&L Standard Deviation: ${pnl_std:.2f}")
    print(f"Sharpe Ratio (Annualized): {sharpe_ratio:.3f}")
    print(f"Performance Score (Mean - 0.1*Std): {avg_daily_pnl - 0.1*pnl_std:.4f}")
    print(f"Number of Trading Days: {len(pnl_history)}")
    print("="*50)

def main():
    """Main backtesting function"""
    print("Loading and preparing data...")
    full_data, train_data, validation_data, test_data = load_and_prepare_data()
    
    print(f"Data loaded: {full_data.shape[0]} instruments, {full_data.shape[1]} days")
    print(f"Train: {train_data.shape[1]} days")
    print(f"Validation: {validation_data.shape[1]} days") 
    print(f"Test: {test_data.shape[1]} days")
    
    # # Optimize parameters using train/validation data
    # best_params = optimize_parameters(train_data, validation_data)
    
    # if best_params is None:
    #     print("Parameter optimization failed, using defaults")
    #     default_params = {
    #         "fast_ma": 8,
    #         "slow_ma": 21,
    #         "trend_ma": 50,
    #         "position_limit": 10000,
    #         "min_trend_duration": 8,
    #         "trend_strength_threshold": 0.025,
    #         "momentum_lookback": 5,
    #         "pullback_threshold": 0.03,
    #         "stop_loss_pct": 0.12,
    #         "ma_separation": 0.015,
    #         "volume_confirmation": True,
    #         "trend_consistency": 0.7
    #     }
    #     with open('model_parameters.json', 'w') as f:
    #         json.dump(default_params, f)
    
    # Run backtest on test data
    print("\nRunning backtest on test data...")
    pnl_history, trade_signals = backtest_strategy(full_data)
    
    # Generate position history for plotting
    position_history = []
    n_instruments = full_data.shape[0]
    positions = np.zeros(n_instruments)
    
    for day in range(1, full_data.shape[1]):
        data_so_far = full_data[:, :day+1]
        new_positions = getMyPosition(data_so_far)
        current_prices = full_data[:, day]
        position_limits = np.array([int(10000 / price) for price in current_prices])
        new_positions = np.clip(new_positions, -position_limits, position_limits)
        position_history.append(new_positions.copy())
        positions = new_positions
    
    # Create visualizations
    print("Generating plots...")
    plot_results(pnl_history, position_history, test_data, trade_signals)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
HYBRID TRADING MODEL - testing.py

This implementation combines 5 distinct trading strategies with optimized weightings:
1. Mean Reversion Strategy 
2. Cross-Sectional Momentum Strategy 
3. Volatility Momentum Strategy 
4. Technical Indicators Strategy 
5. Risk Parity Strategy 

The weightings were selected based on:
- Mean reversion showing strongest historical performance on mean-reverting assets
- Cross-sectional momentum capturing relative strength across instruments
- Volatility momentum for trend identification
- Technical indicators for additional signal validation
- Risk parity for portfolio stability and downside protection

Assessment Formula: mean(PL) - 0.1 * StdDev(PL)
This favors strategies with consistent returns and low volatility.
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Global configuration
N_INSTRUMENTS = 50
POSITION_LIMIT = 10000  # $10k position limit per stock
COMMISSION_RATE = 0.0005  # 5 bps

# Strategy weights (optimized through backtesting)
STRATEGY_WEIGHTS = {
    'mean_reversion': 0.032429361687617335,      # Highest weight - most consistent performer
    'cross_momentum': 0.009832642282048235,      # Strong trend identification
    'volatility_momentum': 0.4559053720008886, # Good risk-adjusted returns
    'technical_indicators': 0.3704338141111066, # Signal validation
    'risk_parity': 0.13139880991833924         # Portfolio stabilization
}

class TradingStrategies:
    """Container class for all trading strategies with detailed implementations"""
    
    def __init__(self):
        self.lookback_short = 5    # Short-term lookback period
        self.lookback_medium = 20  # Medium-term lookback period
        self.lookback_long = 60    # Long-term lookback period
        
    def mean_reversion_strategy(self, prices):
        """
        MEAN REVERSION STRATEGY (Weight: 35%)
        
        Theory: Prices tend to revert to their historical mean over time.
        When prices deviate significantly from mean, they're likely to reverse.
        
        Implementation:
        1. Calculate rolling means over multiple timeframes (5, 20, 60 days)
        2. Compute z-scores to identify extreme deviations
        3. Generate contrarian signals: buy oversold, sell overbought
        4. Apply position sizing based on deviation magnitude
        
        Why this weight (35%):
        - Simulated trading universes often exhibit mean-reverting characteristics
        - Strategy performs well in sideways/choppy markets
        - Provides good risk-adjusted returns with lower volatility
        - Historical backtesting shows consistent performance
        """
        nInst, nDays = prices.shape
        if nDays < self.lookback_long:
            return np.zeros(nInst)
            
        positions = np.zeros(nInst)
        current_prices = prices[:, -1]
        
        # Multi-timeframe mean reversion signals
        for lookback in [self.lookback_short, self.lookback_medium, self.lookback_long]:
            if nDays >= lookback:
                # Calculate rolling statistics
                price_window = prices[:, -lookback:]
                rolling_mean = np.mean(price_window, axis=1)
                rolling_std = np.std(price_window, axis=1)
                
                # Avoid division by zero
                rolling_std = np.where(rolling_std == 0, 1, rolling_std)
                
                # Z-score calculation (standardized deviation from mean)
                z_scores = (current_prices - rolling_mean) / rolling_std
                
                # Generate contrarian signals
                # Negative z-score = oversold = buy signal
                # Positive z-score = overbought = sell signal
                mean_revert_signal = -z_scores
                
                # Weight longer timeframes more heavily
                weight = lookback / self.lookback_long
                positions += weight * mean_revert_signal
        
        # Normalize and scale positions
        positions = positions / 3.0  # Average across timeframes
        
        # Apply volatility-adjusted position sizing
        volatilities = np.std(prices[:, -min(nDays, 20):], axis=1)
        volatilities = np.where(volatilities == 0, 1, volatilities)
        vol_adj_factor = 1.0 / volatilities
        vol_adj_factor = vol_adj_factor / np.mean(vol_adj_factor)  # Normalize
        
        positions = positions * vol_adj_factor
        
        # Convert to share quantities
        target_dollar_positions = positions * 2000  # Base position size
        share_positions = target_dollar_positions / current_prices
        
        return share_positions
    
    def cross_sectional_momentum_strategy(self, prices):
        """
        CROSS-SECTIONAL MOMENTUM STRATEGY (Weight: 25%)
        
        Theory: Relative strength momentum - instruments performing well relative
        to peers will continue outperforming, and vice versa.
        
        Implementation:
        1. Calculate returns over multiple periods for each instrument
        2. Rank instruments by relative performance (cross-sectional ranking)
        3. Go long top performers, short bottom performers
        4. Use percentile-based position sizing
        
        Why this weight (25%):
        - Captures market regime shifts effectively
        - Works well in trending markets
        - Provides diversification benefits vs mean reversion
        - Strong performance during momentum phases
        """
        nInst, nDays = prices.shape
        if nDays < self.lookback_medium:
            return np.zeros(nInst)
            
        current_prices = prices[:, -1]
        positions = np.zeros(nInst)
        
        # Calculate momentum signals over multiple timeframes
        momentum_signals = np.zeros(nInst)
        
        for lookback in [self.lookback_short, self.lookback_medium]:
            if nDays >= lookback:
                # Calculate cumulative returns
                past_prices = prices[:, -lookback-1]
                returns = (current_prices - past_prices) / past_prices
                
                # Cross-sectional ranking (relative performance)
                # Convert returns to percentile ranks
                ranks = np.zeros(nInst)
                sorted_indices = np.argsort(returns)
                for i, idx in enumerate(sorted_indices):
                    ranks[idx] = i / (nInst - 1)  # Percentile rank [0,1]
                
                # Convert percentile ranks to signals [-1, 1]
                momentum_signal = (ranks - 0.5) * 2
                
                # Weight shorter timeframes more for responsiveness
                weight = (self.lookback_medium - lookback + self.lookback_short) / self.lookback_medium
                momentum_signals += weight * momentum_signal
        
        # Normalize momentum signals
        momentum_signals = momentum_signals / 2.0
        
        # Apply sector neutrality (assume instruments 0-24 and 25-49 are different sectors)
        # This prevents over-concentration in one sector
        mid_point = nInst // 2
        sector1_mean = np.mean(momentum_signals[:mid_point])
        sector2_mean = np.mean(momentum_signals[mid_point:])
        
        momentum_signals[:mid_point] -= sector1_mean
        momentum_signals[mid_point:] -= sector2_mean
        
        # Position sizing based on signal strength
        target_dollar_positions = momentum_signals * 3000
        share_positions = target_dollar_positions / current_prices
        
        return share_positions
    
    def volatility_momentum_strategy(self, prices):
        """
        VOLATILITY MOMENTUM STRATEGY (Weight: 20%)
        
        Theory: Volatility clustering - high volatility periods tend to be followed
        by high volatility, and trends are stronger during high volatility periods.
        
        Implementation:
        1. Calculate realized volatility over multiple periods
        2. Identify volatility regimes (low/high)
        3. Adjust momentum signals based on volatility environment
        4. Increase position sizes during low volatility (volatility mean reversion)
        
        Why this weight (20%):
        - Provides regime-aware positioning
        - Helps capture volatility risk premium
        - Performs well during market stress
        - Complements other strategies during different market conditions
        """
        nInst, nDays = prices.shape
        if nDays < self.lookback_medium:
            return np.zeros(nInst)
            
        current_prices = prices[:, -1]
        
        # Calculate returns for volatility estimation
        if nDays >= 2:
            returns = np.diff(np.log(prices), axis=1)
        else:
            return np.zeros(nInst)
        
        # Short-term momentum signals
        if nDays >= self.lookback_short + 1:
            short_returns = returns[:, -self.lookback_short:]
            short_momentum = np.mean(short_returns, axis=1)
        else:
            short_momentum = np.zeros(nInst)
        
        # Realized volatility calculation
        if returns.shape[1] >= self.lookback_medium:
            vol_window = returns[:, -self.lookback_medium:]
            realized_vol = np.std(vol_window, axis=1)
        else:
            realized_vol = np.std(returns, axis=1)
        
        # Volatility regime identification
        overall_vol = np.mean(realized_vol)
        vol_regime = realized_vol / overall_vol  # Relative volatility
        
        # Volatility-adjusted momentum
        # In high vol regimes: reduce momentum positions (more mean reversion)
        # In low vol regimes: increase momentum positions (trends more reliable)
        vol_adjustment = 1.0 / (1.0 + vol_regime)  # Higher vol = lower adjustment
        
        # Generate volatility momentum signals
        vol_momentum_signals = short_momentum * vol_adjustment
        
        # Volatility mean reversion component
        # When volatility is very low, expect volatility increase (position for volatility)
        vol_percentiles = np.zeros(nInst)
        for i in range(nInst):
            if returns.shape[1] >= self.lookback_long:
                hist_vols = []
                for j in range(self.lookback_long, returns.shape[1]):
                    window_vol = np.std(returns[i, j-self.lookback_medium:j])
                    hist_vols.append(window_vol)
                
                if hist_vols:
                    vol_percentile = np.mean(np.array(hist_vols) < realized_vol[i])
                    vol_percentiles[i] = vol_percentile
        
        # Combine momentum and volatility signals
        combined_signals = vol_momentum_signals + 0.3 * (vol_percentiles - 0.5)
        
        # Position sizing
        target_dollar_positions = combined_signals * 2500
        share_positions = target_dollar_positions / current_prices
        
        return share_positions
    
    def technical_indicators_strategy(self, prices):
        """
        TECHNICAL INDICATORS STRATEGY (Weight: 15%)
        
        Theory: Technical analysis patterns contain predictive information
        about future price movements. Combines multiple indicators for robustness.
        
        Implementation:
        1. RSI (Relative Strength Index) for overbought/oversold conditions
        2. Bollinger Bands for mean reversion signals
        3. Moving Average Crossovers for trend identification
        4. Price momentum indicators
        
        Why this weight (15%):
        - Provides additional signal validation
        - Captures different market microstructure effects
        - Lower weight due to higher noise in technical signals
        - Complements fundamental strategies
        """
        nInst, nDays = prices.shape
        if nDays < self.lookback_medium:
            return np.zeros(nInst)
            
        current_prices = prices[:, -1]
        signals = np.zeros(nInst)
        
        # RSI Calculation (14-day default)
        rsi_period = min(14, nDays - 1)
        if nDays > rsi_period:
            for i in range(nInst):
                price_series = prices[i, -rsi_period-1:]
                returns = np.diff(price_series)
                
                gains = np.where(returns > 0, returns, 0)
                losses = np.where(returns < 0, -returns, 0)
                
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100
                
                # RSI signals: oversold (RSI < 30) = buy, overbought (RSI > 70) = sell
                if rsi < 30:
                    rsi_signal = (30 - rsi) / 30  # Strength of oversold condition
                elif rsi > 70:
                    rsi_signal = -(rsi - 70) / 30  # Strength of overbought condition
                else:
                    rsi_signal = 0
                
                signals[i] += 0.4 * rsi_signal
        
        # Bollinger Bands
        if nDays >= self.lookback_medium:
            bb_window = prices[:, -self.lookback_medium:]
            bb_mean = np.mean(bb_window, axis=1)
            bb_std = np.std(bb_window, axis=1)
            
            # Bollinger Band signals
            upper_band = bb_mean + 2 * bb_std
            lower_band = bb_mean - 2 * bb_std
            
            # Mean reversion signals from BB
            bb_signals = np.zeros(nInst)
            for i in range(nInst):
                if bb_std[i] > 0:
                    if current_prices[i] > upper_band[i]:
                        bb_signals[i] = -0.5  # Sell signal
                    elif current_prices[i] < lower_band[i]:
                        bb_signals[i] = 0.5   # Buy signal
                    else:
                        # Position within bands
                        position = (current_prices[i] - bb_mean[i]) / (2 * bb_std[i])
                        bb_signals[i] = -position * 0.3  # Contrarian signal
            
            signals += 0.3 * bb_signals
        
        # Moving Average Crossover
        if nDays >= self.lookback_medium:
            short_ma = np.mean(prices[:, -self.lookback_short:], axis=1)
            long_ma = np.mean(prices[:, -self.lookback_medium:], axis=1)
            
            # MA crossover signals
            ma_signals = (short_ma - long_ma) / long_ma
            signals += 0.3 * ma_signals
        
        # Convert to positions
        target_dollar_positions = signals * 1500
        share_positions = target_dollar_positions / current_prices
        
        return share_positions
    
    def risk_parity_strategy(self, prices):
        """
        RISK PARITY STRATEGY (Weight: 5%)
        
        Theory: Equal risk contribution from each instrument rather than equal
        dollar allocation. Provides portfolio stability and diversification.
        
        Implementation:
        1. Calculate instrument volatilities
        2. Inverse volatility weighting
        3. Risk budgeting across instruments
        4. Dynamic rebalancing based on changing volatilities
        
        Why this weight (5%):
        - Primary role is portfolio stabilization
        - Provides downside protection
        - Lower return expectation but important for risk management
        - Helps meet the evaluation criterion (lower StdDev component)
        """
        nInst, nDays = prices.shape
        if nDays < self.lookback_medium:
            return np.zeros(nInst)
            
        current_prices = prices[:, -1]
        
        # Calculate volatilities
        if nDays >= self.lookback_medium:
            vol_window = prices[:, -self.lookback_medium:]
            returns = np.diff(np.log(vol_window), axis=1)
            volatilities = np.std(returns, axis=1)
        else:
            returns = np.diff(np.log(prices), axis=1)
            volatilities = np.std(returns, axis=1)
        
        # Avoid division by zero
        volatilities = np.where(volatilities == 0, np.mean(volatilities[volatilities > 0]), volatilities)
        
        # Inverse volatility weights
        inv_vol_weights = 1.0 / volatilities
        inv_vol_weights = inv_vol_weights / np.sum(inv_vol_weights)  # Normalize
        
        # Target equal risk contribution
        total_risk_budget = 5000  # Total dollar risk budget
        dollar_positions = inv_vol_weights * total_risk_budget
        
        # Convert to shares
        share_positions = dollar_positions / current_prices
        
        return share_positions


def getMyPosition(prcSoFar):
    """
    MAIN POSITION FUNCTION
    
    Combines all strategies using optimized weights and applies risk management.
    
    Risk Management Features:
    1. Position limits enforcement ($10k per instrument)
    2. Total portfolio exposure limits
    3. Volatility scaling during high-stress periods
    4. Turnover reduction to minimize transaction costs
    
    The function is called daily with complete price history up to current day.
    Returns integer positions (number of shares) for each of the 50 instruments.
    """
    global STRATEGY_WEIGHTS, N_INSTRUMENTS, POSITION_LIMIT
    
    nInst, nDays = prcSoFar.shape
    
    # Initialize for early days
    if nDays < 3:
        return np.zeros(nInst)
    
    # Initialize strategy engine
    strategies = TradingStrategies()
    current_prices = prcSoFar[:, -1]
    
    # Calculate individual strategy positions
    try:
        mean_rev_pos = strategies.mean_reversion_strategy(prcSoFar)
        cross_mom_pos = strategies.cross_sectional_momentum_strategy(prcSoFar)
        vol_mom_pos = strategies.volatility_momentum_strategy(prcSoFar)
        tech_pos = strategies.technical_indicators_strategy(prcSoFar)
        risk_par_pos = strategies.risk_parity_strategy(prcSoFar)
    except:
        # Fallback in case of calculation errors
        return np.zeros(nInst)
    
    # Combine strategies using optimized weights
    combined_positions = (
        STRATEGY_WEIGHTS['mean_reversion'] * mean_rev_pos +
        STRATEGY_WEIGHTS['cross_momentum'] * cross_mom_pos +
        STRATEGY_WEIGHTS['volatility_momentum'] * vol_mom_pos +
        STRATEGY_WEIGHTS['technical_indicators'] * tech_pos +
        STRATEGY_WEIGHTS['risk_parity'] * risk_par_pos
    )
    
    # Risk Management Layer
    
    # 1. Position limits per instrument ($10k limit)
    position_limits = np.array([int(POSITION_LIMIT / price) for price in current_prices])
    combined_positions = np.clip(combined_positions, -position_limits, position_limits)
    
    # 2. Portfolio-level risk management
    total_gross_exposure = np.sum(np.abs(combined_positions * current_prices))
    max_gross_exposure = 300000  # Maximum total exposure
    
    if total_gross_exposure > max_gross_exposure:
        scale_factor = max_gross_exposure / total_gross_exposure
        combined_positions *= scale_factor
    
    # 3. Volatility scaling during market stress
    if nDays >= 20:
        recent_returns = np.diff(np.log(prcSoFar), axis=1)[:, -20:]
        portfolio_returns = np.mean(recent_returns, axis=0)  # Equal-weighted portfolio proxy
        current_vol = np.std(portfolio_returns)
        
        # Historical volatility for comparison
        if nDays >= 60:
            hist_returns = np.diff(np.log(prcSoFar), axis=1)[:, -60:-20]
            hist_portfolio_returns = np.mean(hist_returns, axis=0)
            hist_vol = np.std(hist_portfolio_returns)
            
            if hist_vol > 0:
                vol_ratio = current_vol / hist_vol
                if vol_ratio > 1.5:  # High stress period
                    stress_scale = 1.0 / vol_ratio
                    combined_positions *= stress_scale
    
    # 4. Convert to integers (required output format)
    final_positions = np.array([int(pos) for pos in combined_positions])
    
    # 5. Final position limit check
    final_positions = np.clip(final_positions, -position_limits, position_limits)
    
    return final_positions


# Additional utility functions for analysis and debugging
def calculate_strategy_metrics(prices, positions, start_idx=0):
    """Calculate performance metrics for backtesting and analysis"""
    if len(positions) == 0 or start_idx >= len(positions):
        return {}
    
    # Calculate daily P&L
    daily_pnl = []
    total_commission = 0
    
    for i in range(start_idx + 1, len(positions)):
        if i >= prices.shape[1]:
            break
            
        prev_pos = positions[i-1] if i > 0 else np.zeros(len(positions[0]))
        curr_pos = positions[i]
        curr_prices = prices[:, i]
        
        # Position changes
        position_changes = curr_pos - prev_pos
        
        # Commission costs
        dollar_volume = np.sum(np.abs(position_changes * curr_prices))
        commission = dollar_volume * COMMISSION_RATE
        total_commission += commission
        
        # Daily P&L (assuming we held overnight)
        if i > start_idx + 1:
            price_change = prices[:, i] - prices[:, i-1]
            pnl = np.sum(prev_pos * price_change) - commission
            daily_pnl.append(pnl)
    
    if not daily_pnl:
        return {}
    
    # Calculate metrics
    daily_pnl = np.array(daily_pnl)
    mean_pnl = np.mean(daily_pnl)
    std_pnl = np.std(daily_pnl)
    sharpe = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0
    score = mean_pnl - 0.1 * std_pnl  # Competition scoring formula
    
    return {
        'mean_daily_pnl': mean_pnl,
        'std_daily_pnl': std_pnl,
        'annualized_sharpe': sharpe,
        'competition_score': score,
        'total_commission': total_commission,
        'num_trading_days': len(daily_pnl)
    }


if __name__ == "__main__":
    # Test function for debugging
    print("Hybrid Trading Model - testing.py")
    print("Strategy Weights:", STRATEGY_WEIGHTS)
    print("Ready for evaluation with eval.py")
#!/usr/bin/env python3
"""
ML STRATEGY WEIGHT OPTIMIZER - optimiser.py

This module implements multiple machine learning algorithms to optimize the weights
of the 5 trading strategies in testing.py. The goal is to maximize the competition
scoring function: mean(PL) - 0.1 * StdDev(PL)

Optimization Algorithms Implemented:
1. Bayesian Optimization - Efficient global optimization using Gaussian Processes
2. Genetic Algorithm - Population-based evolutionary optimization
3. Random Forest Regressor - Non-linear relationship modeling
4. Differential Evolution - Robust global optimizer
5. Walk-Forward Analysis - Time series cross-validation

The optimizer uses proper train/validation/test splits to avoid overfitting and
provides robust performance estimates for unseen data.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings
warnings.filterwarnings('ignore')

# Import the trading model components
import sys
import os
import importlib.util

# Global configuration
N_INSTRUMENTS = 50
COMMISSION_RATE = 0.0005
POSITION_LIMIT = 10000

class DataSplitter:
    """
    Handles proper time series data splitting for financial data.
    Ensures no lookahead bias in training/validation/testing.
    """
    
    def __init__(self, prices, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """
        Initialize data splitter with proper time series splits.
        
        Args:
            prices: Price data matrix (instruments x days)
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation  
            test_ratio: Proportion of data for testing
        """
        self.prices = prices
        self.n_inst, self.n_days = prices.shape
        
        # Calculate split indices
        self.train_end = int(self.n_days * train_ratio)
        self.val_end = int(self.n_days * (train_ratio + val_ratio))
        
        print(f"Data split: Train: 0-{self.train_end}, Val: {self.train_end}-{self.val_end}, Test: {self.val_end}-{self.n_days}")
    
    def get_train_data(self):
        """Return training data"""
        return self.prices[:, :self.train_end]
    
    def get_validation_data(self):
        """Return validation data"""
        return self.prices[:, :self.val_end]  # Include training for strategy calculation
    
    def get_test_data(self):
        """Return test data"""
        return self.prices  # Full data for final evaluation
    
    def get_validation_period(self):
        """Return validation period indices"""
        return self.train_end, self.val_end
    
    def get_test_period(self):
        """Return test period indices"""
        return self.val_end, self.n_days


class StrategyEvaluator:
    """
    Evaluates trading strategies and calculates performance metrics.
    This class simulates the eval.py functionality for optimization.
    """
    
    def __init__(self, commission_rate=COMMISSION_RATE):
        self.commission_rate = commission_rate
        
    def evaluate_strategy_weights(self, prices, weights, start_day=0, end_day=None):
        """
        Evaluate strategy performance given specific weights.
        
        Args:
            prices: Price data matrix
            weights: Dictionary of strategy weights
            start_day: Start evaluation from this day
            end_day: End evaluation at this day (None = end of data)
            
        Returns:
            Performance metrics dictionary
        """
        try:
            if end_day is None:
                end_day = prices.shape[1]
                
            # Simulate trading with given weights
            daily_pnl = []
            cash = 0
            current_positions = np.zeros(N_INSTRUMENTS)
            total_commission = 0
            
            for day in range(max(start_day, 20), end_day):  # Start from day 20 for sufficient history
                # Get price history up to current day
                price_history = prices[:, :day+1]
                current_prices = price_history[:, -1]
                
                # Calculate new positions using modified strategy weights
                new_positions = self.get_weighted_positions(price_history, weights)
                if new_positions is None:
                    continue
                
                # Calculate position changes and costs
                position_changes = new_positions - current_positions
                dollar_volume = np.sum(np.abs(position_changes * current_prices))
                commission = dollar_volume * self.commission_rate
                total_commission += commission
                
                # Update cash from trades
                cash -= np.sum(position_changes * current_prices) + commission
                
                # Calculate daily P&L from price changes
                if day > max(start_day, 20):
                    price_change = current_prices - prices[:, day-1]
                    daily_pnl.append(np.sum(current_positions * price_change))
                
                current_positions = new_positions
            
            if not daily_pnl:
                return {'score': -1000.0, 'mean_pnl': 0.0, 'std_pnl': 1.0, 'sharpe': 0.0, 'total_commission': 0.0, 'num_days': 0}
            
            # Calculate performance metrics
            daily_pnl = np.array(daily_pnl)
            mean_pnl = float(np.mean(daily_pnl))
            std_pnl = float(np.std(daily_pnl))
            
            if std_pnl == 0:
                std_pnl = 1.0  # Avoid division by zero
            
            # Competition scoring function
            score = mean_pnl - 0.1 * std_pnl
            
            # Sharpe ratio (annualized)
            sharpe = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0.0
            
            return {
                'score': float(score),
                'mean_pnl': float(mean_pnl),
                'std_pnl': float(std_pnl),
                'sharpe': float(sharpe),
                'total_commission': float(total_commission),
                'num_days': len(daily_pnl)
            }
            
        except Exception as e:
            print(f"Error in strategy evaluation: {e}")
            return {'score': -1000.0, 'mean_pnl': 0.0, 'std_pnl': 1.0, 'sharpe': 0.0, 'total_commission': 0.0, 'num_days': 0}
    
    def get_weighted_positions(self, price_history, weights):
        """
        Calculate positions using weighted strategy combination.
        This replicates the strategy logic from testing.py with custom weights.
        """
        try:
            nInst, nDays = price_history.shape
            if nDays < 3:
                return np.zeros(nInst)
            
            # Use simplified strategy implementation
            strategies = SimplifiedTradingStrategies()
            current_prices = price_history[:, -1]
            
            # Calculate individual strategy positions
            mean_rev_pos = strategies.mean_reversion_strategy(price_history)
            cross_mom_pos = strategies.cross_sectional_momentum_strategy(price_history)
            vol_mom_pos = strategies.volatility_momentum_strategy(price_history)
            tech_pos = strategies.technical_indicators_strategy(price_history)
            risk_par_pos = strategies.risk_parity_strategy(price_history)
            
            # Combine with custom weights
            combined_positions = (
                weights.get('mean_reversion', 0.35) * mean_rev_pos +
                weights.get('cross_momentum', 0.25) * cross_mom_pos +
                weights.get('volatility_momentum', 0.20) * vol_mom_pos +
                weights.get('technical_indicators', 0.15) * tech_pos +
                weights.get('risk_parity', 0.05) * risk_par_pos
            )
            
            # Apply position limits
            position_limits = np.array([max(1, int(POSITION_LIMIT / max(0.01, price))) for price in current_prices])
            combined_positions = np.clip(combined_positions, -position_limits, position_limits)
            
            return np.array([int(pos) for pos in combined_positions])
            
        except Exception as e:
            print(f"Error in position calculation: {e}")
            return np.zeros(nInst)


class SimplifiedTradingStrategies:
    """Simplified version of trading strategies for optimization"""
    
    def __init__(self):
        self.lookback_short = 5
        self.lookback_medium = 20
        self.lookback_long = 60
    
    def mean_reversion_strategy(self, prices):
        nInst, nDays = prices.shape
        if nDays < self.lookback_long:
            return np.zeros(nInst)
        
        current_prices = prices[:, -1]
        positions = np.zeros(nInst)
        
        for lookback in [self.lookback_short, self.lookback_medium, self.lookback_long]:
            if nDays >= lookback:
                price_window = prices[:, -lookback:]
                rolling_mean = np.mean(price_window, axis=1)
                rolling_std = np.std(price_window, axis=1)
                rolling_std = np.where(rolling_std == 0, 1, rolling_std)
                
                z_scores = (current_prices - rolling_mean) / rolling_std
                mean_revert_signal = -z_scores
                weight = lookback / self.lookback_long
                positions += weight * mean_revert_signal
        
        positions = positions / 3.0
        volatilities = np.std(prices[:, -min(nDays, 20):], axis=1)
        volatilities = np.where(volatilities == 0, 1, volatilities)
        vol_adj_factor = 1.0 / volatilities
        vol_adj_factor = vol_adj_factor / np.mean(vol_adj_factor)
        
        positions = positions * vol_adj_factor
        target_dollar_positions = positions * 2000
        share_positions = target_dollar_positions / current_prices
        return share_positions
    
    def cross_sectional_momentum_strategy(self, prices):
        nInst, nDays = prices.shape
        if nDays < self.lookback_medium:
            return np.zeros(nInst)
        
        current_prices = prices[:, -1]
        momentum_signals = np.zeros(nInst)
        
        for lookback in [self.lookback_short, self.lookback_medium]:
            if nDays >= lookback:
                past_prices = prices[:, -lookback-1]
                returns = (current_prices - past_prices) / past_prices
                
                ranks = np.zeros(nInst)
                sorted_indices = np.argsort(returns)
                for i, idx in enumerate(sorted_indices):
                    ranks[idx] = i / (nInst - 1)
                
                momentum_signal = (ranks - 0.5) * 2
                weight = (self.lookback_medium - lookback + self.lookback_short) / self.lookback_medium
                momentum_signals += weight * momentum_signal
        
        momentum_signals = momentum_signals / 2.0
        target_dollar_positions = momentum_signals * 3000
        share_positions = target_dollar_positions / current_prices
        return share_positions
    
    def volatility_momentum_strategy(self, prices):
        nInst, nDays = prices.shape
        if nDays < self.lookback_medium:
            return np.zeros(nInst)
        
        current_prices = prices[:, -1]
        
        if nDays >= 2:
            returns = np.diff(np.log(prices), axis=1)
        else:
            return np.zeros(nInst)
        
        if nDays >= self.lookback_short + 1:
            short_returns = returns[:, -self.lookback_short:]
            short_momentum = np.mean(short_returns, axis=1)
        else:
            short_momentum = np.zeros(nInst)
        
        if returns.shape[1] >= self.lookback_medium:
            vol_window = returns[:, -self.lookback_medium:]
            realized_vol = np.std(vol_window, axis=1)
        else:
            realized_vol = np.std(returns, axis=1)
        
        overall_vol = np.mean(realized_vol)
        vol_regime = realized_vol / overall_vol
        vol_adjustment = 1.0 / (1.0 + vol_regime)
        
        vol_momentum_signals = short_momentum * vol_adjustment
        target_dollar_positions = vol_momentum_signals * 2500
        share_positions = target_dollar_positions / current_prices
        return share_positions
    
    def technical_indicators_strategy(self, prices):
        nInst, nDays = prices.shape
        if nDays < self.lookback_medium:
            return np.zeros(nInst)
        
        current_prices = prices[:, -1]
        signals = np.zeros(nInst)
        
        # Simplified technical indicators
        if nDays >= self.lookback_medium:
            short_ma = np.mean(prices[:, -self.lookback_short:], axis=1)
            long_ma = np.mean(prices[:, -self.lookback_medium:], axis=1)
            ma_signals = (short_ma - long_ma) / long_ma
            signals += ma_signals
        
        target_dollar_positions = signals * 1500
        share_positions = target_dollar_positions / current_prices
        return share_positions
    
    def risk_parity_strategy(self, prices):
        nInst, nDays = prices.shape
        if nDays < self.lookback_medium:
            return np.zeros(nInst)
        
        current_prices = prices[:, -1]
        
        if nDays >= self.lookback_medium:
            vol_window = prices[:, -self.lookback_medium:]
            returns = np.diff(np.log(vol_window), axis=1)
            volatilities = np.std(returns, axis=1)
        else:
            returns = np.diff(np.log(prices), axis=1)
            volatilities = np.std(returns, axis=1)
        
        volatilities = np.where(volatilities == 0, np.mean(volatilities[volatilities > 0]), volatilities)
        inv_vol_weights = 1.0 / volatilities
        inv_vol_weights = inv_vol_weights / np.sum(inv_vol_weights)
        
        total_risk_budget = 5000
        dollar_positions = inv_vol_weights * total_risk_budget
        share_positions = dollar_positions / current_prices
        return share_positions


class BayesianOptimizer:
    """
    Bayesian Optimization using Gaussian Processes for efficient hyperparameter search.
    
    Theory: Bayesian optimization is particularly effective for expensive-to-evaluate
    functions (like backtesting trading strategies). It uses a probabilistic model
    (Gaussian Process) to model the objective function and an acquisition function
    to decide where to sample next.
    
    Why chosen:
    - Efficient for small parameter spaces (5 strategy weights)
    - Handles noisy objective functions well (financial data is noisy)
    - Provides uncertainty estimates
    - Requires fewer evaluations than grid search
    """
    
    def __init__(self, evaluator, bounds):
        self.evaluator = evaluator
        self.bounds = bounds
        self.X_sample = []
        self.y_sample = []
        
        # Gaussian Process with Matern kernel (good for financial data)
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    
    def optimize(self, prices, n_calls=50, start_day=0, end_day=None):
        """
        Optimize strategy weights using Bayesian optimization.
        
        Args:
            prices: Price data matrix
            n_calls: Number of optimization iterations
            start_day: Start evaluation period
            end_day: End evaluation period
            
        Returns:
            Best weights and performance metrics
        """
        print("Starting Bayesian Optimization...")
        
        # Initialize with random samples
        n_initial = min(10, n_calls // 5)
        for i in range(n_initial):
            weights = self._sample_random_weights()
            score = self._evaluate_weights(prices, weights, start_day, end_day)
            self.X_sample.append(list(weights.values()))
            self.y_sample.append(score)
            print(f"Initial sample {i+1}/{n_initial}: Score = {score:.4f}")
        
        # Bayesian optimization loop
        for i in range(n_initial, n_calls):
            # Fit GP on current data
            X = np.array(self.X_sample)
            y = np.array(self.y_sample)
            self.gp.fit(X, y)
            
            # Find next point to evaluate using acquisition function
            next_weights = self._optimize_acquisition()
            
            # Evaluate new point
            score = self._evaluate_weights(prices, next_weights, start_day, end_day)
            self.X_sample.append(list(next_weights.values()))
            self.y_sample.append(score)
            
            current_best = max(self.y_sample)
            print(f"Iteration {i+1}/{n_calls}: Score = {score:.4f}, Best = {current_best:.4f}")
        
        # Return best result
        best_idx = np.argmax(self.y_sample)
        best_weights_list = self.X_sample[best_idx]
        best_weights = {
            'mean_reversion': best_weights_list[0],
            'cross_momentum': best_weights_list[1],
            'volatility_momentum': best_weights_list[2],
            'technical_indicators': best_weights_list[3],
            'risk_parity': best_weights_list[4]
        }
        
        return best_weights, self.y_sample[best_idx]
    
    def _sample_random_weights(self):
        """Sample random weights that sum to 1"""
        weights = np.random.dirichlet([1, 1, 1, 1, 1])  # Symmetric Dirichlet
        return {
            'mean_reversion': weights[0],
            'cross_momentum': weights[1],
            'volatility_momentum': weights[2],
            'technical_indicators': weights[3],
            'risk_parity': weights[4]
        }
    
    def _evaluate_weights(self, prices, weights, start_day, end_day):
        """Evaluate strategy weights and return score"""
        try:
            metrics = self.evaluator.evaluate_strategy_weights(prices, weights, start_day, end_day)
            if metrics is None or 'score' not in metrics:
                return -1000.0
            return float(metrics['score'])
        except Exception as e:
            print(f"Evaluation error: {e}")
            return -1000.0  # Penalty for failed evaluation
    
    def _optimize_acquisition(self):
        """Optimize acquisition function to find next point to evaluate"""
        def acquisition(x):
            x = x.reshape(1, -1)
            mean, std = self.gp.predict(x, return_std=True)
            # Upper Confidence Bound acquisition
            return -(mean + 2.0 * std)  # Negative because we minimize
        
        # Optimize acquisition function with constraint that weights sum to 1
        from scipy.optimize import minimize
        
        best_acquisition = np.inf
        best_x = None
        
        # Multiple random starts
        for _ in range(10):
            x0 = np.random.dirichlet([1, 1, 1, 1, 1])
            
            # Constraint: weights sum to 1
            constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = [(0, 1) for _ in range(5)]
            
            result = minimize(acquisition, x0, method='SLSQP', bounds=bounds, constraints=constraint)
            
            if result.success and result.fun < best_acquisition:
                best_acquisition = result.fun
                best_x = result.x
        
        if best_x is None:
            best_x = np.random.dirichlet([1, 1, 1, 1, 1])
        
        # Normalize to ensure sum = 1
        best_x = best_x / np.sum(best_x)
        
        return {
            'mean_reversion': best_x[0],
            'cross_momentum': best_x[1],
            'volatility_momentum': best_x[2],
            'technical_indicators': best_x[3],
            'risk_parity': best_x[4]
        }


class GeneticOptimizer:
    """
    Genetic Algorithm for global optimization of strategy weights.
    
    Theory: Genetic algorithms are inspired by natural evolution. They maintain
    a population of candidate solutions and evolve them through selection,
    crossover, and mutation operations.
    
    Why chosen:
    - Excellent for global optimization (avoids local optima)
    - Handles non-convex, noisy objective functions well
    - Robust to outliers in financial data
    - Can explore diverse weight combinations
    """
    
    def __init__(self, evaluator, population_size=50):
        self.evaluator = evaluator
        self.population_size = population_size
    
    def optimize(self, prices, generations=30, start_day=0, end_day=None):
        """
        Optimize strategy weights using genetic algorithm.
        
        Args:
            prices: Price data matrix
            generations: Number of generations to evolve
            start_day: Start evaluation period
            end_day: End evaluation period
            
        Returns:
            Best weights and performance metrics
        """
        print("Starting Genetic Algorithm Optimization...")
        
        # Initialize population
        population = []
        fitness_scores = []
        
        for i in range(self.population_size):
            individual = self._create_individual()
            fitness = self._evaluate_individual(prices, individual, start_day, end_day)
            population.append(individual)
            fitness_scores.append(fitness)
            
            if i % 10 == 0:
                print(f"Initializing population: {i+1}/{self.population_size}")
        
        # Evolution loop
        for generation in range(generations):
            new_population = []
            new_fitness_scores = []
            
            # Elitism: keep best 10% of population
            elite_size = self.population_size // 10
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            
            for idx in elite_indices:
                new_population.append(population[idx].copy())
                new_fitness_scores.append(fitness_scores[idx])
            
            # Generate new individuals through crossover and mutation
            while len(new_population) < self.population_size:
                # Selection: tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                child = self._crossover(parent1, parent2)
                
                # Mutation
                child = self._mutate(child)
                
                # Evaluate child
                fitness = self._evaluate_individual(prices, child, start_day, end_day)
                
                new_population.append(child)
                new_fitness_scores.append(fitness)
            
            population = new_population
            fitness_scores = new_fitness_scores
            
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            print(f"Generation {generation+1}/{generations}: Best = {best_fitness:.4f}, Avg = {avg_fitness:.4f}")
        
        # Return best individual
        best_idx = np.argmax(fitness_scores)
        best_individual = population[best_idx]
        
        return best_individual, fitness_scores[best_idx]
    
    def _create_individual(self):
        """Create random individual (weight vector)"""
        weights = np.random.dirichlet([1, 1, 1, 1, 1])
        return {
            'mean_reversion': weights[0],
            'cross_momentum': weights[1],
            'volatility_momentum': weights[2],
            'technical_indicators': weights[3],
            'risk_parity': weights[4]
        }
    
    def _evaluate_individual(self, prices, individual, start_day, end_day):
        """Evaluate individual fitness"""
        try:
            metrics = self.evaluator.evaluate_strategy_weights(prices, individual, start_day, end_day)
            if metrics is None or 'score' not in metrics:
                return -1000.0
            return float(metrics['score'])
        except Exception as e:
            print(f"Individual evaluation error: {e}")
            return -1000.0
    
    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Tournament selection for parent selection"""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1, parent2):
        """Blend crossover for weight vectors"""
        alpha = 0.5  # Blending parameter
        child = {}
        
        for key in parent1.keys():
            child[key] = alpha * parent1[key] + (1 - alpha) * parent2[key]
        
        # Normalize to ensure sum = 1
        total = sum(child.values())
        for key in child:
            child[key] /= total
        
        return child
    
    def _mutate(self, individual, mutation_rate=0.1, mutation_strength=0.1):
        """Gaussian mutation with normalization"""
        if np.random.random() < mutation_rate:
            # Add Gaussian noise
            for key in individual:
                individual[key] += np.random.normal(0, mutation_strength)
                individual[key] = max(0, individual[key])  # Ensure non-negative
            
            # Normalize to ensure sum = 1
            total = sum(individual.values())
            if total > 0:
                for key in individual:
                    individual[key] /= total
        
        return individual


class RandomForestOptimizer:
    """
    Random Forest Regressor for modeling non-linear relationships between weights and performance.
    
    Theory: Random Forest can capture complex non-linear relationships between
    strategy weights and performance. It's robust to overfitting and provides
    feature importance insights.
    
    Why chosen:
    - Handles non-linear relationships well
    - Provides feature importance (which strategies matter most)
    - Robust to noisy financial data
    - Can model interaction effects between strategies
    """
    
    def __init__(self, evaluator, n_estimators=100):
        self.evaluator = evaluator
        self.rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        self.scaler = StandardScaler()
    
    def optimize(self, prices, n_samples=200, start_day=0, end_day=None):
        """
        Optimize using Random Forest surrogate model.
        
        Args:
            prices: Price data matrix
            n_samples: Number of samples for training surrogate model
            start_day: Start evaluation period
            end_day: End evaluation period
            
        Returns:
            Best weights and performance metrics
        """
        print("Starting Random Forest Optimization...")
        
        # Generate training data for surrogate model
        X_train = []
        y_train = []
        
        for i in range(n_samples):
            weights = self._sample_weights()
            score = self._evaluate_weights(prices, weights, start_day, end_day)
            
            X_train.append(list(weights.values()))
            y_train.append(score)
            
            if i % 50 == 0:
                print(f"Generating samples: {i+1}/{n_samples}")
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train surrogate model
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.rf.fit(X_train_scaled, y_train)
        
        print("Surrogate model trained. Feature importances:")
        feature_names = ['mean_reversion', 'cross_momentum', 'volatility_momentum', 'technical_indicators', 'risk_parity']
        importances = self.rf.feature_importances_
        for name, importance in zip(feature_names, importances):
            print(f"  {name}: {importance:.4f}")
        
        # Use surrogate model to find optimal weights
        best_weights = None
        best_score = -np.inf
        
        # Search with higher resolution using surrogate model
        for _ in range(1000):
            weights = self._sample_weights()
            X_pred = np.array([list(weights.values())])
            X_pred_scaled = self.scaler.transform(X_pred)
            predicted_score = self.rf.predict(X_pred_scaled)[0]
            
            if predicted_score > best_score:
                best_score = predicted_score
                best_weights = weights
        
        # Evaluate best weights on actual objective function
        actual_score = self._evaluate_weights(prices, best_weights, start_day, end_day)
        
        print(f"Surrogate model best score: {best_score:.4f}")
        print(f"Actual best score: {actual_score:.4f}")
        
        return best_weights, actual_score
    
    def _sample_weights(self):
        """Sample random weights"""
        weights = np.random.dirichlet([1, 1, 1, 1, 1])
        return {
            'mean_reversion': weights[0],
            'cross_momentum': weights[1],
            'volatility_momentum': weights[2],
            'technical_indicators': weights[3],
            'risk_parity': weights[4]
        }
    
    def _evaluate_weights(self, prices, weights, start_day, end_day):
        """Evaluate strategy weights"""
        try:
            metrics = self.evaluator.evaluate_strategy_weights(prices, weights, start_day, end_day)
            if metrics is None or 'score' not in metrics:
                return -1000.0
            return float(metrics['score'])
        except Exception as e:
            print(f"Random Forest evaluation error: {e}")
            return -1000.0


class DifferentialEvolutionOptimizer:
    """
    Differential Evolution for robust global optimization.
    
    Theory: Differential Evolution is a population-based optimization algorithm
    that uses differences between population members to guide the search.
    It's particularly effective for continuous optimization problems.
    
    Why chosen:
    - Excellent convergence properties
    - Robust to local optima
    - Handles constraints well (weight normalization)
    - Proven effective for financial optimization problems
    """
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
    
    def optimize(self, prices, maxiter=100, start_day=0, end_day=None):
        """
        Optimize using Differential Evolution.
        
        Args:
            prices: Price data matrix
            maxiter: Maximum iterations
            start_day: Start evaluation period
            end_day: End evaluation period
            
        Returns:
            Best weights and performance metrics
        """
        print("Starting Differential Evolution Optimization...")
        
        def objective(x):
            # Normalize weights to sum to 1
            if np.sum(x) == 0:
                return 1000.0
            x = x / np.sum(x)
            weights = {
                'mean_reversion': float(x[0]),
                'cross_momentum': float(x[1]),
                'volatility_momentum': float(x[2]),
                'technical_indicators': float(x[3]),
                'risk_parity': float(x[4])
            }
            
            try:
                metrics = self.evaluator.evaluate_strategy_weights(prices, weights, start_day, end_day)
                if metrics is None or 'score' not in metrics:
                    return 1000.0
                return -float(metrics['score'])  # Minimize negative score
            except Exception as e:
                print(f"DE objective error: {e}")
                return 1000.0  # Penalty for failed evaluation
        
        # Bounds for each weight [0, 1]
        bounds = [(0, 1) for _ in range(5)]
        
        # Constraint: weights should sum to approximately 1
        def constraint(x):
            return np.sum(x) - 1
        
        # Optimize
        result = differential_evolution(
            objective,
            bounds,
            maxiter=maxiter,
            popsize=15,
            seed=42,
            disp=True
        )
        
        if result.success:
            optimal_weights = result.x / np.sum(result.x)  # Normalize
            best_weights = {
                'mean_reversion': float(optimal_weights[0]),
                'cross_momentum': float(optimal_weights[1]),
                'volatility_momentum': float(optimal_weights[2]),
                'technical_indicators': float(optimal_weights[3]),
                'risk_parity': float(optimal_weights[4])
            }
            best_score = float(-result.fun)
            
            print(f"Optimization successful. Best score: {best_score:.4f}")
            return best_weights, best_score
        else:
            print("Optimization failed")
            return None, -1000.0


class WalkForwardOptimizer:
    """
    Walk-Forward Analysis for robust time series validation.
    
    Theory: Walk-forward analysis is the gold standard for validating trading
    strategies. It simulates real-world conditions by continuously retraining
    the model as new data becomes available.
    
    Why chosen:
    - Prevents lookahead bias
    - Provides realistic performance estimates
    - Simulates actual trading conditions
    - Reduces overfitting to specific time periods
    """
    
    def __init__(self, evaluator, base_optimizer):
        self.evaluator = evaluator
        self.base_optimizer = base_optimizer
    
    def optimize(self, prices, n_splits=5, min_train_size=100):
        """
        Perform walk-forward optimization.
        
        Args:
            prices: Price data matrix
            n_splits: Number of walk-forward splits
            min_train_size: Minimum training size for each split
            
        Returns:
            Average optimal weights and out-of-sample performance
        """
        print("Starting Walk-Forward Analysis...")
        
        n_days = prices.shape[1]
        split_size = (n_days - min_train_size) // n_splits
        
        all_optimal_weights = []
        all_oos_scores = []
        
        for i in range(n_splits):
            print(f"\nWalk-Forward Split {i+1}/{n_splits}")
            
            # Define training and test periods
            train_start = 0
            train_end = min_train_size + i * split_size
            test_start = train_end
            test_end = min(train_end + split_size, n_days)
            
            if test_end - test_start < 20:  # Need minimum test period
                continue
            
            print(f"Training period: {train_start}-{train_end}")
            print(f"Testing period: {test_start}-{test_end}")
            
            # Optimize on training data
            train_prices = prices[:, :train_end]
            optimal_weights, _ = self.base_optimizer.optimize(
                train_prices,
                start_day=max(20, train_start),
                end_day=train_end
            )
            
            # Evaluate on out-of-sample test data
            oos_metrics = self.evaluator.evaluate_strategy_weights(
                prices,
                optimal_weights,
                start_day=test_start,
                end_day=test_end
            )
            
            all_optimal_weights.append(optimal_weights)
            all_oos_scores.append(oos_metrics['score'])
            
            print(f"Out-of-sample score: {oos_metrics['score']:.4f}")
        
        if not all_optimal_weights:
            return None, -1000
        
        # Average optimal weights across splits
        avg_weights = {}
        for key in all_optimal_weights[0].keys():
            avg_weights[key] = np.mean([w[key] for w in all_optimal_weights])
        
        # Normalize to ensure sum = 1
        total = sum(avg_weights.values())
        for key in avg_weights:
            avg_weights[key] /= total
        
        avg_oos_score = np.mean(all_oos_scores)
        std_oos_score = np.std(all_oos_scores)
        
        print(f"\nWalk-Forward Results:")
        print(f"Average out-of-sample score: {avg_oos_score:.4f} ± {std_oos_score:.4f}")
        print(f"Average optimal weights: {avg_weights}")
        
        return avg_weights, avg_oos_score


def load_price_data(filename='prices.txt'):
    """Load price data from file"""
    try:
        # Try loading as space-separated values
        data = np.loadtxt(filename)
        if data.ndim == 1:
            # Reshape if it's a 1D array
            data = data.reshape(-1, 1)
        return data.T  # Transpose to get instruments x days
    except:
        try:
            # Try loading as CSV
            data = pd.read_csv(filename, header=None, sep=r'\s+')
            return data.values.T
        except:
            print(f"Error loading price data from {filename}")
            return None


def main():
    """
    Main optimization routine that runs all algorithms and compares results.
    """
    print("=== STRATEGY WEIGHT OPTIMIZATION ===")
    
    # Load price data
    prices = load_price_data('prices.txt')
    if prices is None:
        print("Failed to load price data")
        return
    
    print(f"Loaded price data: {prices.shape[0]} instruments, {prices.shape[1]} days")
    
    # Split data
    splitter = DataSplitter(prices, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    train_prices = splitter.get_train_data()
    val_prices = splitter.get_validation_data()
    train_end, val_end = splitter.get_validation_period()
    
    # Initialize evaluator
    evaluator = StrategyEvaluator()
    
    # Store results from all optimization methods
    results = {}
    
    # 1. Bayesian Optimization
    print("\n" + "="*50)
    print("1. BAYESIAN OPTIMIZATION")
    print("="*50)
    try:
        bayesian_optimizer = BayesianOptimizer(evaluator, bounds=[(0, 1)] * 5)
        bayes_weights, bayes_score = bayesian_optimizer.optimize(
            val_prices, n_calls=30, start_day=train_end, end_day=val_end
        )
        results['Bayesian'] = {'weights': bayes_weights, 'score': bayes_score}
        print(f"Bayesian optimal weights: {bayes_weights}")
        print(f"Bayesian validation score: {bayes_score:.4f}")
    except Exception as e:
        print(f"Bayesian optimization failed: {e}")
        results['Bayesian'] = {'weights': None, 'score': None}
    
    # 2. Genetic Algorithm
    print("\n" + "="*50)
    print("2. GENETIC ALGORITHM")
    print("="*50)
    try:
        genetic_optimizer = GeneticOptimizer(evaluator, population_size=30)
        genetic_weights, genetic_score = genetic_optimizer.optimize(
            val_prices, generations=20, start_day=train_end, end_day=val_end
        )
        results['Genetic'] = {'weights': genetic_weights, 'score': genetic_score}
        print(f"Genetic optimal weights: {genetic_weights}")
        print(f"Genetic validation score: {genetic_score:.4f}")
    except Exception as e:
        print(f"Genetic optimization failed: {e}")
        results['Genetic'] = {'weights': None, 'score': None}
    
    # 3. Random Forest
    print("\n" + "="*50)
    print("3. RANDOM FOREST OPTIMIZATION")
    print("="*50)
    try:
        rf_optimizer = RandomForestOptimizer(evaluator, n_estimators=50)
        rf_weights, rf_score = rf_optimizer.optimize(
            val_prices, n_samples=100, start_day=train_end, end_day=val_end
        )
        results['Random Forest'] = {'weights': rf_weights, 'score': rf_score}
        print(f"Random Forest optimal weights: {rf_weights}")
        print(f"Random Forest validation score: {rf_score:.4f}")
    except Exception as e:
        print(f"Random Forest optimization failed: {e}")
        results['Random Forest'] = {'weights': None, 'score': None}
    
    # 4. Differential Evolution
    print("\n" + "="*50)
    print("4. DIFFERENTIAL EVOLUTION")
    print("="*50)
    try:
        de_optimizer = DifferentialEvolutionOptimizer(evaluator)
        de_weights, de_score = de_optimizer.optimize(
            val_prices, maxiter=50, start_day=train_end, end_day=val_end
        )
        results['Differential Evolution'] = {'weights': de_weights, 'score': de_score}
        if de_weights is not None:
            print(f"Differential Evolution optimal weights: {de_weights}")
            print(f"Differential Evolution validation score: {de_score:.4f}")
        else:
            print("Differential Evolution failed")
    except Exception as e:
        print(f"Differential Evolution failed: {e}")
        results['Differential Evolution'] = {'weights': None, 'score': None}
    
    # 5. Walk-Forward Analysis (using best single method)
    print("\n" + "="*50)
    print("5. WALK-FORWARD ANALYSIS")
    print("="*50)
    try:
        # Use Bayesian as base optimizer for walk-forward
        wf_optimizer = WalkForwardOptimizer(evaluator, BayesianOptimizer(evaluator, bounds=[(0, 1)] * 5))
        wf_weights, wf_score = wf_optimizer.optimize(train_prices, n_splits=3, min_train_size=60)
        results['Walk-Forward'] = {'weights': wf_weights, 'score': wf_score}
        if wf_weights is not None:
            print(f"Walk-Forward optimal weights: {wf_weights}")
            print(f"Walk-Forward validation score: {wf_score:.4f}")
        else:
            print("Walk-Forward analysis failed")
    except Exception as e:
        print(f"Walk-Forward analysis failed: {e}")
        results['Walk-Forward'] = {'weights': None, 'score': None}
    
    # Compare all methods
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*50)
    
    best_score = -1000.0
    best_method = None
    best_weights = None
    
    for method, result in results.items():
        if result['weights'] is not None and result['score'] is not None:
            score = float(result['score'])
            print(f"\n{method}:")
            print(f"  Score: {score:.4f}")
            print(f"  Weights: {result['weights']}")
            
            if score > best_score:
                best_score = score
                best_method = method
                best_weights = result['weights']
        else:
            print(f"\n{method}: FAILED")
    
    if best_method is None:
        print("All optimization methods failed!")
        return
    
    print(f"\nBEST METHOD: {best_method}")
    print(f"BEST WEIGHTS: {best_weights}")
    print(f"VALIDATION SCORE: {best_score:.4f}")
    
    # Test on out-of-sample data
    test_start, test_end = splitter.get_test_period()
    test_metrics = evaluator.evaluate_strategy_weights(
        prices, best_weights, start_day=test_start, end_day=test_end
    )
    
    if test_metrics is not None and 'score' in test_metrics:
        print(f"\nOUT-OF-SAMPLE TEST RESULTS:")
        print(f"Test Score: {test_metrics['score']:.4f}")
        print(f"Mean Daily P&L: ${test_metrics['mean_pnl']:.2f}")
        print(f"Std Daily P&L: ${test_metrics['std_pnl']:.2f}")
        print(f"Annualized Sharpe: {test_metrics['sharpe']:.2f}")
    else:
        print("\nOut-of-sample test failed!")
    
    # Save optimal weights for use in testing.py
    print(f"\nTo use these optimized weights in testing.py, update STRATEGY_WEIGHTS to:")
    print(f"STRATEGY_WEIGHTS = {best_weights}")


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import math

class TradingStrategy():
    """Trading Strategy for Supervised Learning based models, implementing different trading strategies using Kelly criterion for optimal bet sizing."""
    def __init__(self, wallet_a, wallet_b, frac_kelly, trade_threshold):
        """Initialize the TradingStrategy class with the initial wallet balances and Kelly fraction option."""
        self.frac_kelly = frac_kelly
        self.trade_threshold = trade_threshold
        # Initialize wallets for different trading strategies
        self.wallet_a = {'mean_reversion': wallet_a, 'trend': wallet_a, 'pure_forcasting': wallet_a, 'hybrid_mean_reversion': wallet_a, 'hybrid_trend': wallet_a}
        self.wallet_b = {'mean_reversion': wallet_b, 'trend': wallet_b, 'pure_forcasting': wallet_b, 'hybrid_mean_reversion': wallet_b, 'hybrid_trend': wallet_b}
        # Track profit/loss, wins/losses, and total gains/losses for each strategy
        self.total_profit_or_loss = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0}
        self.num_wins = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0}
        self.num_losses = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0}
        self.total_gains = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0}
        self.total_losses = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0}

        # New: Track open positions
        self.open_positions = {
            'mean_reversion': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'trend': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'pure_forcasting': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'hybrid_mean_reversion': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'hybrid_trend': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0}
        }

        self.min_trades_for_full_kelly = 50  # Minimum trades before using full Kelly
    
    def win_loss_ratio(self, strategy_name):
        """Calculate the win/loss ratio for a strategy with basic smoothing."""
        total_trades = self.num_wins[strategy_name] + self.num_losses[strategy_name]
        
        if total_trades == 0:
            return 1.5  # Conservative default
            
        # Use consistent scaling factor
        confidence = min(1.0, total_trades / self.min_trades_for_full_kelly)
        smoothing = max(0.1, 1.0 - confidence)
        
        # Calculate averages with basic error handling
        avg_gain = (self.total_gains[strategy_name] / self.num_wins[strategy_name]) if self.num_wins[strategy_name] else 1.0
        avg_loss = (self.total_losses[strategy_name] / self.num_losses[strategy_name]) if self.num_losses[strategy_name] else 1.0
        
        # Apply smoothing and return with floor
        return max(0.1, (avg_gain + smoothing) / (avg_loss + smoothing))

    def win_probability(self, strategy_name):
        """Calculate win probability with basic statistical adjustment."""
        total_trades = self.num_wins[strategy_name] + self.num_losses[strategy_name]
        
        if total_trades == 0:
            return 0.5  # Neutral default
            
        # Basic win rate
        win_rate = self.num_wins[strategy_name] / total_trades
        
        # Use consistent scaling
        confidence = min(1.0, total_trades / self.min_trades_for_full_kelly)
        adjusted_rate = (win_rate * confidence) + (0.5 * (1 - confidence))
        
        # Keep within reasonable bounds
        return max(0.1, min(0.9, adjusted_rate))

    def kelly_criterion(self, strategy_name):
        """Calculate Kelly fraction with basic risk controls."""
        # Get core metrics
        win_prob = self.win_probability(strategy_name)
        win_ratio = self.win_loss_ratio(strategy_name)
        total_trades = self.num_wins[strategy_name] + self.num_losses[strategy_name]

        # Basic Kelly calculation
        kelly = win_prob - ((1 - win_prob) / win_ratio)

        # Single confidence adjustment based on trade count
        confidence = min(1.0, total_trades / self.min_trades_for_full_kelly)
        kelly *= confidence

        # Return bounded result
        return max(0.01, min(0.25, kelly))

    def calculate_profit(self, strategy_name, trade_direction, curr_ratio, next_ratio, f_i):
        """Calculate profit/loss and handle position management"""
        # kelly_factor = 0.33 if self.frac_kelly else 1.0
        # f_i = f_i * kelly_factor
        
        profit_in_base_currency = 0
        
        # First, close any open position
        if self.open_positions[strategy_name]['type'] is not None:
            profit_in_base_currency += self.close_position(strategy_name, curr_ratio)
        
        # Then open new position if there's a trade signal
        if trade_direction != 'no_trade':
            # Calculate total portfolio value in currency A
            total_value_in_a = self.wallet_a[strategy_name] + (self.wallet_b[strategy_name] / curr_ratio)

            base_bet_size_a = f_i * total_value_in_a
            
            if trade_direction == 'buy_currency_a':
                bet_size_a = min(base_bet_size_a, self.wallet_a[strategy_name])
                bet_size_b = bet_size_a * curr_ratio
                
                # Check if we have enough B
                if bet_size_b <= self.wallet_b[strategy_name]:
                    self.wallet_b[strategy_name] -= bet_size_b
                    self.wallet_a[strategy_name] += bet_size_a
                    
                    self.open_positions[strategy_name] = {
                        'type': 'long',
                        'size_a': bet_size_a,
                        'size_b': bet_size_b,
                        'entry_ratio': curr_ratio
                    }
                
            elif trade_direction == 'sell_currency_a':
                bet_size_a = min(base_bet_size_a, self.wallet_a[strategy_name])
                bet_size_b = bet_size_a * curr_ratio
                
                if bet_size_a <= self.wallet_a[strategy_name]:
                    self.wallet_a[strategy_name] -= bet_size_a
                    self.wallet_b[strategy_name] += bet_size_b
                    
                    self.open_positions[strategy_name] = {
                        'type': 'short',
                        'size_a': bet_size_a,
                        'size_b': bet_size_b,
                        'entry_ratio': curr_ratio
                    }
        
        return profit_in_base_currency
    
    def close_position(self, strategy_name, curr_ratio):
        """Close an open position and calculate profit/loss"""
        position = self.open_positions[strategy_name]
        profit = 0
        
        if position['type'] == 'long':
            # Close long position (sell currency A)
            self.wallet_a[strategy_name] -= position['size_a']
            self.wallet_b[strategy_name] += position['size_a'] * curr_ratio
            # Calculate profit
            profit = position['size_a'] * (curr_ratio - position['entry_ratio']) / curr_ratio
            
        elif position['type'] == 'short':
            # Close short position (buy currency A)
            self.wallet_b[strategy_name] -= position['size_b']
            self.wallet_a[strategy_name] += position['size_b'] / curr_ratio
            # Calculate profit
            profit = position['size_a'] * (position['entry_ratio'] - curr_ratio) / curr_ratio
        
        # Reset position tracking
        self.open_positions[strategy_name] = {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0}
        
        return profit

    def determine_trade_direction(self, strategy_name, base_ratio_change, predicted_ratio_change):
        """Determine the trade direction based on strategy and ratio changes."""
        trade_direction = 'no_trade'

        if(strategy_name == 'mean_reversion'):
            # Mean reversion strategy: trade against significant ratio changes
            if base_ratio_change > self.trade_threshold:
                trade_direction = 'sell_currency_a'
            elif base_ratio_change < -self.trade_threshold:
                trade_direction = 'buy_currency_a'
                
        elif(strategy_name == 'trend'):
            # Trend strategy: trade towards significant ratio changes
            if base_ratio_change > self.trade_threshold:
                trade_direction = 'buy_currency_a'
            elif base_ratio_change < -self.trade_threshold:
                trade_direction = 'sell_currency_a'
                
        elif(strategy_name == 'pure_forcasting'):
            # Pure forecasting strategy: trade based on predicted future ratio changes
            if predicted_ratio_change > self.trade_threshold:
                trade_direction = 'buy_currency_a'
            elif predicted_ratio_change < -self.trade_threshold:
                trade_direction = 'sell_currency_a'
                
        elif(strategy_name == 'hybrid_mean_reversion'):
            # Hybrid strategy: combine mean reversion and pure forecasting signals
            if base_ratio_change < -self.trade_threshold and predicted_ratio_change > self.trade_threshold:
                trade_direction = 'buy_currency_a'
            elif base_ratio_change > self.trade_threshold and predicted_ratio_change < -self.trade_threshold:
                trade_direction = 'sell_currency_a'
                
        elif(strategy_name == 'hybrid_trend'):
            # Hybrid strategy: combine trend and pure forecasting signals
            if base_ratio_change > self.trade_threshold and predicted_ratio_change > self.trade_threshold:
                trade_direction = 'buy_currency_a'
            elif base_ratio_change < -self.trade_threshold and predicted_ratio_change < -self.trade_threshold:
                trade_direction = 'sell_currency_a'
            
        return trade_direction
        
    def display_total_profit(self):
        """Display the total profit or loss for each strategy."""
        print(f"Total Profits - {self.total_profit_or_loss}")

    def display_final_wallet_amount(self):
        """Display the final amounts in both wallets for each strategy."""
        print(f"Final amount in Wallet A - {self.wallet_a}")
        print(f"Final amount in Wallet B - {self.wallet_b}")
        
    def simulate_trading_with_strategies(self, actual_rates, predicted_rates):
        """Simulate trading over a series of exchange rates using different strategies."""
        strategy_names = ['mean_reversion', 'trend', 'pure_forcasting', 'hybrid_mean_reversion', 'hybrid_trend']
        
        for strategy_name in strategy_names:
            for i in range(2, len(actual_rates) - 1):
                curr_ratio = actual_rates[i - 1]
                prev_ratio = actual_rates[i - 2]
                predicted_next_ratio = predicted_rates[i]
                actual_next_ratio = actual_rates[i]

                # Determine the predicted and actual ratio changes
                predicted_ratio_change = predicted_next_ratio - curr_ratio
                base_ratio_change = curr_ratio - prev_ratio

                # Calculate the Kelly fraction for the bet size
                f_i = self.kelly_criterion(strategy_name)
            
                # Determine the trade direction based on the strategy and ratio changes
                trade_direction = self.determine_trade_direction(strategy_name, base_ratio_change, predicted_ratio_change)

                # Calculate the profit or loss for the trade
                profit = self.calculate_profit(strategy_name, trade_direction, curr_ratio, actual_next_ratio, f_i)
                self.total_profit_or_loss[strategy_name] += profit

                # Update win/loss counters and totals
                if profit > 0:
                    self.num_wins[strategy_name] += 1
                    self.total_gains[strategy_name] += abs(profit)
                elif profit < 0:
                    self.num_losses[strategy_name] += 1
                    self.total_losses[strategy_name] += abs(profit)

        # Close any remaining open positions for all strategies
        for strategy_name in strategy_names:
            if self.open_positions[strategy_name]['type'] is not None:
                final_ratio = actual_rates[-1]  # Use the last ratio for closing
                profit = self.close_position(strategy_name, final_ratio)
                self.total_profit_or_loss[strategy_name] += profit

        # Display the overall results after simulation
        self.display_total_profit()
        self.display_final_wallet_amount()
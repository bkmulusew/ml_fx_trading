import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression, LinearRegression

class TradingStrategy():
    """Trading Strategy for Supervised Learning based models, implementing different trading strategies using Kelly criterion for optimal bet sizing."""
    def __init__(self, wallet_a, wallet_b, frac_kelly, trade_threshold):
        """Initialize the TradingStrategy class with the initial wallet balances and Kelly fraction option."""
        self.frac_kelly = frac_kelly
        self.trade_threshold = trade_threshold
        # Initialize wallets for different trading strategies
        self.wallet_a = {'mean_reversion': wallet_a, 'trend': wallet_a, 'pure_forcasting': wallet_a, 'hybrid_mean_reversion': wallet_a, 'hybrid_trend': wallet_a, 'ensamble': wallet_a}
        self.wallet_b = {'mean_reversion': wallet_b, 'trend': wallet_b, 'pure_forcasting': wallet_b, 'hybrid_mean_reversion': wallet_b, 'hybrid_trend': wallet_b, 'ensamble': wallet_b}
        # Track profit/loss, wins/losses, and total gains/losses for each strategy
        self.total_profit_or_loss = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensamble': 0}
        self.num_trades = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensamble': 0}
        self.num_wins = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensamble': 0}
        self.num_losses = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensamble': 0}
        self.total_gains = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensamble': 0}
        self.total_losses = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensamble': 0}

        # New: Track open positions
        self.open_positions = {
            'mean_reversion': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'trend': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'pure_forcasting': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'hybrid_mean_reversion': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'hybrid_trend': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'ensamble': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0}
        }

        self.min_trades_for_full_kelly = 50  # Minimum trades before using full Kelly
        self.fixed_position_size = 1000  # Fixed position size for training
        
        # Initialize linear regression model
        self.linear_model = LinearRegression()
        self.trained = False

        self.trend_coeff = 0
        self.forecasting_coeff = 0

    def calculate_profit_for_signals(self, curr_ratio, next_ratio):
        """Calculate potential profit for given signals using fixed position size."""
        max_profit = float('-inf')
        best_direction = 'no_trade'
        
        # Try both possible trade directions
        for direction in ['buy_currency_a', 'sell_currency_a']:
            # Calculate profit for this direction
            if direction == 'buy_currency_a':
                profit = self.fixed_position_size * (next_ratio - curr_ratio) / next_ratio
            else:  # sell_currency_a
                profit = self.fixed_position_size * (curr_ratio - next_ratio) / next_ratio
            
            # Update best direction if this profit is higher
            if profit > max_profit:
                max_profit = profit
                best_direction = direction
        
        return max_profit, best_direction
    
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

    def calculate_profit(self, strategy_name, trade_direction, curr_ratio, f_i, use_kelly):
        """Calculate profit/loss and handle position management"""        
        profit_in_base_currency = 0
        
        # First, close any open position
        if self.open_positions[strategy_name]['type'] is not None:
            profit_in_base_currency += self.close_position(strategy_name, curr_ratio)
        
        # Then open new position if there's a trade signal
        if trade_direction != 'no_trade':
            # Calculate total portfolio value in currency A
            total_value_in_a = self.wallet_a[strategy_name] + (self.wallet_b[strategy_name] / curr_ratio)

            if(use_kelly):
                base_bet_size_a = f_i * total_value_in_a
            else:
                base_bet_size_a = 1000
            
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
    
    def get_strategy_signals(self, base_ratio_change, predicted_ratio_change, curr_ratio, next_ratio):
        """Get the signals (+1, 0, -1) for each strategy."""
        signals = {}
        for strategy_name in ['trend', 'pure_forcasting']:
            trade_direction = self.determine_trade_direction(strategy_name, base_ratio_change, predicted_ratio_change)
            if trade_direction == 'buy_currency_a':
                signals[strategy_name] = self.fixed_position_size * (next_ratio - curr_ratio) / next_ratio
            elif trade_direction == 'sell_currency_a':
                signals[strategy_name] = self.fixed_position_size * (curr_ratio - next_ratio) / next_ratio
            else:
                signals[strategy_name] = 0
        return signals
    
    def train_linear_regression(self, historical_data):
        """Train the linear regression model on historical data."""
        X = []
        y = []
        for features, label in historical_data:
            # Only include data where label is +1 or -1
            X.append(features)
            y.append(label)
        if X:
            self.linear_model.fit(X, y)
            self.trained = True
            print("Linear regression model trained.")

            # Output the weights
            strategy_names = ['trend', 'pure_forcasting']
            # weights = self.logistic_model.coef_[0]
            # intercept = self.logistic_model.intercept_[0]
            weights = self.linear_model.coef_
            intercept = self.linear_model.intercept_
            print("Linear Regression Coefficients:")
            for name, weight in zip(strategy_names, weights):
                print(f"  {name}: {weight:.4f}")
            print(f"Intercept: {intercept:.4f}\n")
        else:
            print("Not enough data to train linear regression model.")
        
    def display_total_profit(self):
        """Display the total profit or loss for each strategy."""
        print(f"Total Profits - {self.total_profit_or_loss}")

    def display_profit_per_trade(self):
        """Display the average profit or loss per trade for each strategy."""
        print("Average Profit Per Trade:")
        for strategy_name in self.total_profit_or_loss.keys():
            total_trades = self.num_trades[strategy_name]
            if total_trades > 0:
                avg_profit = self.total_profit_or_loss[strategy_name] / total_trades
            else:
                avg_profit = 0
            print(f"Profit Per Trade - {strategy_name}: {avg_profit:.2f}")

    def display_final_wallet_amount(self):
        """Display the final amounts in both wallets for each strategy."""
        print(f"Final amount in Wallet A - {self.wallet_a}")
        print(f"Final amount in Wallet B - {self.wallet_b}")

    def simulate_trading_with_strategies(self, actual_rates, predicted_rates, use_kelly=True):
        """Simulate trading over a series of exchange rates using different strategies."""
        strategy_name = "ensamble"
        
        historical_data = []
        
        split_idx = len(actual_rates)//2
        print(f"split_idx : {split_idx}")
        actual_rates_first_half = actual_rates[:split_idx]
        actual_rates_second_half = actual_rates[split_idx:]

        trend_cumulative_profit = 0
        forecast_cumulative_profit = 0
        max_cumulative_profit = 0

        for i in range(2, len(actual_rates_first_half) - 1):
            curr_ratio = actual_rates[i - 1]
            prev_ratio = actual_rates[i - 2]
            predicted_next_ratio = predicted_rates[i]
            actual_next_ratio = actual_rates[i]

            # Determine the predicted and actual ratio changes
            # predicted_ratio_change = predicted_next_ratio - curr_ratio
            # base_ratio_change = curr_ratio - prev_ratio

            # Avoid division by zero
            if prev_ratio != 0 and curr_ratio != 0:
                # Calculate percentage increase instead of ratio change
                predicted_percentage_increase = ((predicted_next_ratio - curr_ratio) / curr_ratio) * 100
                actual_percentage_increase = ((actual_next_ratio - curr_ratio) / curr_ratio) * 100
                base_percentage_increase = ((curr_ratio - prev_ratio) / prev_ratio) * 100
            else:
                print("Skipping iteration due to zero division risk.")

            # signals = self.get_strategy_signals(base_ratio_change, predicted_ratio_change, curr_ratio, actual_next_ratio)
            signals = self.get_strategy_signals(base_percentage_increase, predicted_percentage_increase, curr_ratio, actual_next_ratio)
            trend_cumulative_profit += signals['trend']
            forecast_cumulative_profit += signals['pure_forcasting']

            # Calculate profit using fixed position size
            profit, _ = self.calculate_profit_for_signals(curr_ratio, actual_next_ratio)
            max_cumulative_profit += profit
            feature_vector = [trend_cumulative_profit, forecast_cumulative_profit]

            # Store the data point
            historical_data.append((feature_vector, max_cumulative_profit))

        # Train linear regression model
        self.train_linear_regression(historical_data)

        self.trend_coeff = self.linear_model.coef_[0]
        self.forecasting_coeff = self.linear_model.coef_[1]

        # # Testing/Trading phase
        # trend_cumulative_profit = 0
        # forecast_cumulative_profit = 0

        if(self.trend_coeff >= self.forecasting_coeff):
            trade_direction_strategy = "trend"
        else:
            trade_direction_strategy = "pure_forcasting"

        for i in range(2, len(actual_rates_second_half) - 1):
            curr_ratio = actual_rates[i - 1]
            prev_ratio = actual_rates[i - 2]
            predicted_next_ratio = predicted_rates[i]
            actual_next_ratio = actual_rates[i]

            # Determine the predicted and actual ratio changes
            # predicted_ratio_change = predicted_next_ratio - curr_ratio
            # base_ratio_change = curr_ratio - prev_ratio

            # Avoid division by zero
            if prev_ratio != 0 and curr_ratio != 0:
                # Calculate percentage increase instead of ratio change
                predicted_percentage_increase = ((predicted_next_ratio - curr_ratio) / curr_ratio) * 100
                actual_percentage_increase = ((actual_next_ratio - curr_ratio) / curr_ratio) * 100
                base_percentage_increase = ((curr_ratio - prev_ratio) / prev_ratio) * 100
            else:
                print("Skipping iteration due to zero division risk.")

            # # Collect signals
            # signals = self.get_strategy_signals(base_ratio_change, predicted_ratio_change, curr_ratio, actual_next_ratio)
            # trend_cumulative_profit += signals['trend']
            # forecast_cumulative_profit += signals['pure_forcasting']
            # feature_vector = [trend_cumulative_profit, forecast_cumulative_profit]

            # # Use linear regression model to predict trade direction
            # if self.trained:
            #     prediction = self.linear_model.predict([feature_vector])[0]
            #     print(f"Prediction: {prediction}")
            #     if prediction == 1:
            #         trade_direction = 'buy_currency_a'
            #     elif prediction == -1:
            #         trade_direction = 'sell_currency_a'
            #     else:
            #         trade_direction = 'no_trade'
            # else:
            #     trade_direction = 'no_trade'

            # Determine the trade direction based on the strategy and ratio changes
            # trade_direction = self.determine_trade_direction(trade_direction_strategy, base_ratio_change, predicted_ratio_change)
            trade_direction = self.determine_trade_direction(trade_direction_strategy, base_percentage_increase, predicted_percentage_increase)

            if(trade_direction != "no_trade"):
                self.num_trades[strategy_name] += 1

            # Calculate the Kelly fraction for the bet size
            f_i = self.kelly_criterion(strategy_name)

            # Calculate the profit or loss for the trade
            profit = self.calculate_profit(strategy_name, trade_direction, curr_ratio, f_i, use_kelly)
            self.total_profit_or_loss[strategy_name] += profit

            # Update win/loss counters and totals
            if profit > 0:
                self.num_wins[strategy_name] += 1
                self.total_gains[strategy_name] += abs(profit)
            elif profit < 0:
                self.num_losses[strategy_name] += 1
                self.total_losses[strategy_name] += abs(profit)

        # Close any remaining open positions for all strategies
        if self.open_positions[strategy_name]['type'] is not None:
            final_ratio = actual_rates[-1]  # Use the last ratio for closing
            profit = self.close_position(strategy_name, final_ratio)
            self.total_profit_or_loss[strategy_name] += profit

        strategy_names = ['mean_reversion', 'trend', 'pure_forcasting', 'hybrid_mean_reversion', 'hybrid_trend']
        
        for strategy_name in strategy_names:
            for i in range(2, len(actual_rates_second_half) - 1):
                curr_ratio = actual_rates[i - 1]
                prev_ratio = actual_rates[i - 2]
                predicted_next_ratio = predicted_rates[i]

                # Determine the predicted and actual ratio changes
                # predicted_ratio_change = predicted_next_ratio - curr_ratio
                # base_ratio_change = curr_ratio - prev_ratio

                # Avoid division by zero
                if prev_ratio != 0 and curr_ratio != 0:
                    # Calculate percentage increase instead of ratio change
                    predicted_percentage_increase = ((predicted_next_ratio - curr_ratio) / curr_ratio) * 100
                    actual_percentage_increase = ((actual_next_ratio - curr_ratio) / curr_ratio) * 100
                    base_percentage_increase = ((curr_ratio - prev_ratio) / prev_ratio) * 100
                else:
                    print("Skipping iteration due to zero division risk.")

                # Calculate the Kelly fraction for the bet size
                f_i = self.kelly_criterion(strategy_name)
            
                # Determine the trade direction based on the strategy and ratio changes
                # trade_direction = self.determine_trade_direction(strategy_name, base_ratio_change, predicted_ratio_change)

                trade_direction = self.determine_trade_direction(strategy_name, base_percentage_increase, predicted_percentage_increase)

                if(trade_direction != "no_trade"):
                    self.num_trades[strategy_name] += 1

                # Calculate the profit or loss for the trade
                profit = self.calculate_profit(strategy_name, trade_direction, curr_ratio, f_i, use_kelly)
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
        self.display_profit_per_trade()
        
    # def simulate_trading_with_strategies(self, actual_rates, predicted_rates, use_kelly=True):
    #     """Simulate trading over a series of exchange rates using different strategies."""
    #     strategy_names = ['mean_reversion', 'trend', 'pure_forcasting', 'hybrid_mean_reversion', 'hybrid_trend']
        
    #     for strategy_name in strategy_names:
    #         for i in range(2, len(actual_rates) - 1):
    #             curr_ratio = actual_rates[i - 1]
    #             prev_ratio = actual_rates[i - 2]
    #             predicted_next_ratio = predicted_rates[i]

    #             # Determine the predicted and actual ratio changes
    #             predicted_ratio_change = predicted_next_ratio - curr_ratio
    #             base_ratio_change = curr_ratio - prev_ratio

    #             # Calculate the Kelly fraction for the bet size
    #             f_i = self.kelly_criterion(strategy_name)
            
    #             # Determine the trade direction based on the strategy and ratio changes
    #             trade_direction = self.determine_trade_direction(strategy_name, base_ratio_change, predicted_ratio_change)

    #             if(trade_direction != "no_trade"):
    #                 self.num_trades[strategy_name] += 1

    #             # Calculate the profit or loss for the trade
    #             profit = self.calculate_profit(strategy_name, trade_direction, curr_ratio, f_i, use_kelly)
    #             self.total_profit_or_loss[strategy_name] += profit

    #             # Update win/loss counters and totals
    #             if profit > 0:
    #                 self.num_wins[strategy_name] += 1
    #                 self.total_gains[strategy_name] += abs(profit)
    #             elif profit < 0:
    #                 self.num_losses[strategy_name] += 1
    #                 self.total_losses[strategy_name] += abs(profit)

    #     # Close any remaining open positions for all strategies
    #     for strategy_name in strategy_names:
    #         if self.open_positions[strategy_name]['type'] is not None:
    #             final_ratio = actual_rates[-1]  # Use the last ratio for closing
    #             profit = self.close_position(strategy_name, final_ratio)
    #             self.total_profit_or_loss[strategy_name] += profit

    #     # Display the overall results after simulation
    #     self.display_total_profit()
    #     self.display_final_wallet_amount()
    #     self.display_profit_per_trade()
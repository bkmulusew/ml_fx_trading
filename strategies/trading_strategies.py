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
        self.wallet_a = {'mean_reversion': wallet_a, 'trend': wallet_a, 'pure_forecasting': wallet_a, 'ensemble_with_llm_mean_reversion': wallet_a, 'ensemble_with_llm_trend': wallet_a}
        self.wallet_b = {'mean_reversion': wallet_b, 'trend': wallet_b, 'pure_forecasting': wallet_b, 'ensemble_with_llm_mean_reversion': wallet_b, 'ensemble_with_llm_trend': wallet_b}
        # Track profit/loss, wins/losses, and total gains/losses for each strategy
        self.total_profit_or_loss = {'mean_reversion': 0, 'trend': 0, 'pure_forecasting': 0, 'ensemble_with_llm_mean_reversion': 0, 'ensemble_with_llm_trend': 0}
        self.num_trades = {'mean_reversion': 0, 'trend': 0, 'pure_forecasting': 0, 'ensemble_with_llm_mean_reversion': 0, 'ensemble_with_llm_trend': 0}
        self.num_wins = {'mean_reversion': 0, 'trend': 0, 'pure_forecasting': 0, 'ensemble_with_llm_mean_reversion': 0, 'ensemble_with_llm_trend': 0}
        self.num_losses = {'mean_reversion': 0, 'trend': 0, 'pure_forecasting': 0, 'ensemble_with_llm_mean_reversion': 0, 'ensemble_with_llm_trend': 0}
        self.total_gains = {'mean_reversion': 0, 'trend': 0, 'pure_forecasting': 0, 'ensemble_with_llm_mean_reversion': 0, 'ensemble_with_llm_trend': 0}
        self.total_losses = {'mean_reversion': 0, 'trend': 0, 'pure_forecasting': 0, 'ensemble_with_llm_mean_reversion': 0, 'ensemble_with_llm_trend': 0}

        # New: Track open positions
        self.open_positions = {
            'mean_reversion': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'trend': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'pure_forecasting': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'ensemble_with_llm_mean_reversion': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'ensemble_with_llm_trend': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0}
        }

        self.open_positions_signal = {
            'mean_reversion': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'trend': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'pure_forecasting': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'ensemble_with_llm_mean_reversion': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'ensemble_with_llm_trend': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0}
        }

        self.min_trades_for_full_kelly = 50  # Minimum trades before using full Kelly
        self.fixed_position_size = 1000  # Fixed position size for training
        
        # Initialize linear regression models
        self.linear_model_llm_trend = LinearRegression()
        self.linear_model_llm_mean_reversion = LinearRegression()

        # Reset all coefficients

        self.trend_coeff_llm = 0
        self.mean_reversion_coeff_llm = 0
        self.forecasting_coeff_llm = 0
        self.llm_sentiment_coeff = 0

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

    def calculate_profit(self, strategy_name, trade_direction, bid_price, ask_price, f_i, use_kelly):
        """Calculate profit/loss and handle position management"""        
        profit_in_base_currency = 0
        
        # Check if there's an open position
        if self.open_positions[strategy_name]['type'] is not None:
            # If new trade direction is different from current position type, close the position
            current_position_type = 'long' if self.open_positions[strategy_name]['type'] == 'long' else 'short'
            new_position_type = 'long' if trade_direction == 'buy_currency_a' else 'short' if trade_direction == 'sell_currency_a' else None
            
            if new_position_type is not None and new_position_type != current_position_type:
                profit_in_base_currency += self.close_position(strategy_name, bid_price, ask_price)
            else:
                # If same type or no trade, don't make a new trade
                return profit_in_base_currency
        
        # Then open new position if there's a trade signal and no matching position type
        if trade_direction != 'no_trade':
            # Calculate total portfolio value in currency A
            total_value_in_a = self.wallet_a[strategy_name] + (self.wallet_b[strategy_name] / ask_price)

            if(use_kelly):
                base_bet_size_a = f_i * total_value_in_a
            else:
                base_bet_size_a = self.fixed_position_size
            
            if trade_direction == 'buy_currency_a':
                bet_size_a = min(base_bet_size_a, self.wallet_a[strategy_name])
                bet_size_b = bet_size_a * ask_price
                
                # Check if we have enough B
                if bet_size_b <= self.wallet_b[strategy_name]:
                    self.wallet_b[strategy_name] -= bet_size_b
                    self.wallet_a[strategy_name] += bet_size_a
                    
                    self.open_positions[strategy_name] = {
                        'type': 'long',
                        'size_a': bet_size_a,
                        'size_b': bet_size_b,
                        'entry_ratio': ask_price
                    }
                
            elif trade_direction == 'sell_currency_a':
                bet_size_a = min(base_bet_size_a, self.wallet_a[strategy_name])
                bet_size_b = bet_size_a * bid_price
                
                if bet_size_a <= self.wallet_a[strategy_name]:
                    self.wallet_a[strategy_name] -= bet_size_a
                    self.wallet_b[strategy_name] += bet_size_b
                    
                    self.open_positions[strategy_name] = {
                        'type': 'short',
                        'size_a': bet_size_a,
                        'size_b': bet_size_b,
                        'entry_ratio': bid_price
                    }
        
        return profit_in_base_currency
    
    def close_position(self, strategy_name, bid_price, ask_price):
        """Close an open position and calculate profit/loss"""
        position = self.open_positions[strategy_name]
        profit_in_base_currency = 0
        
        if position['type'] == 'long':
            # Close long position (sell currency A)
            # Entry: BUY at ASK (entry_ratio)
            # Exit: SELL at BID
            exit_amount_b = position['size_a'] * bid_price
            self.wallet_a[strategy_name] -= position['size_a']
            self.wallet_b[strategy_name] += exit_amount_b
            profit_in_base_currency = position['size_a'] * (bid_price - position['entry_ratio']) / bid_price
            
        elif position['type'] == 'short':
            # Close short position (buy currency A)
            # Entry: SELL at BID (entry_ratio)
            # Exit: BUY at ASK
            exit_amount_a = position['size_b'] / ask_price
            self.wallet_b[strategy_name] -= position['size_b']
            self.wallet_a[strategy_name] += exit_amount_a
            profit_in_base_currency = position['size_a'] * (position['entry_ratio'] - ask_price) / ask_price
        
        # Reset position tracking
        self.open_positions[strategy_name] = {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0}
        
        return profit_in_base_currency

    def determine_trade_direction(self, strategy_name, base_ratio_change, predicted_ratio_change, llm_sentiment):
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
                
        elif(strategy_name == 'pure_forecasting'):
            # Pure forecasting strategy: trade based on predicted future ratio changes
            if predicted_ratio_change > self.trade_threshold:
                trade_direction = 'buy_currency_a'
            elif predicted_ratio_change < -self.trade_threshold:
                trade_direction = 'sell_currency_a'

        elif(strategy_name == 'llm'):
            if(llm_sentiment == -1):
                trade_direction = 'sell_currency_a'
            elif(llm_sentiment == 1):
                trade_direction = 'buy_currency_a'
            
        return trade_direction
    
    '''
    def close_position_signal(self, strategy_name, curr_ratio, next_ratio):
        """Close a signal position and calculate the profit/loss"""
        position = self.open_positions_signal[strategy_name]
        profit = 0
        
        if position['type'] == 'long':
            # Close long position (sell currency A)
            # Entry: BUY at entry_ratio
            # Exit: SELL at curr_ratio
            profit = self.fixed_position_size * (curr_ratio - position['entry_ratio']) / curr_ratio
            
        elif position['type'] == 'short':
            # Close short position (buy currency A)
            # Entry: SELL at entry_ratio
            # Exit: BUY at curr_ratio
            profit = self.fixed_position_size * (position['entry_ratio'] - curr_ratio) / curr_ratio
        
        # Reset position tracking
        self.open_positions_signal[strategy_name] = {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0}
        
        return profit
    '''

    def get_strategy_signals(self, base_ratio_change, predicted_ratio_change, curr_ratio, next_ratio, llm_sentiment):
        """Get the signals (+1, 0, -1) for each strategy."""
        signals = {}
        strategy_names = ['mean_reversion', 'trend', 'pure_forecasting', 'llm']
        for strategy_name in strategy_names:
            trade_direction = self.determine_trade_direction(strategy_name, base_ratio_change, predicted_ratio_change, llm_sentiment)
            if trade_direction == 'buy_currency_a':
                signals[strategy_name] = self.fixed_position_size * (next_ratio - curr_ratio) / next_ratio
            elif trade_direction == 'sell_currency_a':
                signals[strategy_name] = self.fixed_position_size * (curr_ratio - next_ratio) / next_ratio
            else:
                signals[strategy_name] = 0
        return signals
    
    def train_linear_regression(self, historical_data, linear_model):
        """Train the linear regression model on historical data."""
        X = []
        y = []
        for features, label in historical_data:
            X.append(features)
            y.append(label)
        if X:
            linear_model.fit(X, y)
            print("Linear regression model trained.")
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

    def diplay_num_trades(self):
        """Display the number of trades for each strategy."""
        print(f"Number of trades - {self.num_trades}")

    def normalize_coefficients(self):
        """Normalize the regression coefficients to get proper weights."""
        # Calculate total magnitude for normalization
        total_magnitude = abs(self.mean_reversion_coeff_llm) + abs(self.trend_coeff_llm) + \
                         abs(self.forecasting_coeff_llm) + abs(self.llm_sentiment_coeff)
        
        # Keep signs but normalize magnitude
        weights = {
            'mean_reversion': self.mean_reversion_coeff_llm / total_magnitude,
            'trend': self.trend_coeff_llm / total_magnitude,
            'forecasting': self.forecasting_coeff_llm / total_magnitude,
            'llm': self.llm_sentiment_coeff / total_magnitude
        }
        
        return weights

    def get_weighted_trade_direction(self, base_percentage_increase, predicted_percentage_increase, llm_sentiment):
        """Determine trade direction by weighting all strategy signals."""
        # Get individual trade directions
        mean_rev_dir = self.determine_trade_direction("mean_reversion", base_percentage_increase, predicted_percentage_increase, 0)
        trend_dir = self.determine_trade_direction("trend", base_percentage_increase, predicted_percentage_increase, 0)
        forecast_dir = self.determine_trade_direction("pure_forecasting", base_percentage_increase, predicted_percentage_increase, 0)
        llm_dir = self.determine_trade_direction("llm", base_percentage_increase, predicted_percentage_increase, llm_sentiment)
        
        # Convert directions to numerical values
        dir_to_value = {
            'buy_currency_a': 1,
            'sell_currency_a': -1,
            'no_trade': 0
        }
        
        # Get weights
        weights = self.normalize_coefficients()
        
        # Calculate weighted sum
        # Positive coefficients: follow the strategy's signal
        # Negative coefficients: do opposite of strategy's signal
        weighted_signal = (
            weights['mean_reversion'] * dir_to_value[mean_rev_dir] +
            weights['trend'] * dir_to_value[trend_dir] +
            weights['forecasting'] * dir_to_value[forecast_dir] +
            weights['llm'] * dir_to_value[llm_dir]
        )
        
        # Convert weighted sum back to trade direction using thresholds
        if weighted_signal > self.trade_threshold:
            return 'buy_currency_a'
        elif weighted_signal < -self.trade_threshold:
            return 'sell_currency_a'
        else:
            return 'no_trade'

    def simulate_trading_with_strategies(self, actual_rates, predicted_rates, bid_prices, ask_prices, llm_sentiments, use_kelly=True):
        """Simulate trading over a series of exchange rates using different strategies."""

        ''' 
        -----------------------------------------
        Ensemble with LLM Trend

        '''
        print("-----------------------------------------")
        print("Ensemble with LLM Trend")
        print("-----------------------------------------")
        strategy_name = "ensemble_with_llm_trend"
        
        historical_data = []
        
        split_idx = len(actual_rates)//2

        trend_cumulative_profit = 0
        forecast_cumulative_profit = 0
        llm_sentiment_cumulative_profit = 0
        max_cumulative_profit = 0

        for i in range(2, split_idx - 1):
            curr_ratio = actual_rates[i - 1]
            prev_ratio = actual_rates[i - 2]
            predicted_next_ratio = predicted_rates[i]
            actual_next_ratio = actual_rates[i]
            llm_sentiment = llm_sentiments[i - 1]

            # Avoid division by zero
            if prev_ratio != 0 and curr_ratio != 0:
                # Calculate percentage increase instead of ratio change
                predicted_percentage_increase = ((predicted_next_ratio - curr_ratio) / curr_ratio) * 100
                actual_percentage_increase = ((actual_next_ratio - curr_ratio) / curr_ratio) * 100
                base_percentage_increase = ((curr_ratio - prev_ratio) / prev_ratio) * 100
            else:
                print("Skipping iteration due to zero division risk.")

            signals = self.get_strategy_signals(base_percentage_increase, predicted_percentage_increase, curr_ratio, actual_next_ratio, llm_sentiment)
            trend_cumulative_profit += signals['trend']
            forecast_cumulative_profit += signals['pure_forecasting']
            llm_sentiment_cumulative_profit += signals['llm']

            # Calculate profit using fixed position size
            profit, _ = self.calculate_profit_for_signals(curr_ratio, actual_next_ratio)
            max_cumulative_profit += profit
            feature_vector = [trend_cumulative_profit, forecast_cumulative_profit, llm_sentiment_cumulative_profit]

            # Store the data point
            historical_data.append((feature_vector, max_cumulative_profit))

        # Train linear regression model
        self.train_linear_regression(historical_data, self.linear_model_llm_trend)

        self.mean_reversion_coeff_llm = 0
        self.trend_coeff_llm = self.linear_model_llm_trend.coef_[0]
        self.forecasting_coeff_llm = self.linear_model_llm_trend.coef_[1]
        self.llm_sentiment_coeff = self.linear_model_llm_trend.coef_[2]

        print(f"mean_reversion_coeff_llm : {self.mean_reversion_coeff_llm}")
        print(f"trend_coeff_llm : {self.trend_coeff_llm}")
        print(f"forecasting_coeff_llm : {self.forecasting_coeff_llm}")
        print(f"llm_sentiment_coeff : {self.llm_sentiment_coeff}")

        for i in range(split_idx + 2, len(actual_rates) - 1):
            curr_ratio = actual_rates[i - 1]
            prev_ratio = actual_rates[i - 2]
            predicted_next_ratio = predicted_rates[i]
            actual_next_ratio = actual_rates[i]
            llm_sentiment = llm_sentiments[i - 1]
            bid_price = bid_prices[i]
            ask_price = ask_prices[i]

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

            # Determine the trade direction based on the strategy and ratio changes
            trade_direction = self.get_weighted_trade_direction(base_percentage_increase, predicted_percentage_increase, llm_sentiment)

            if(trade_direction != "no_trade"):
                self.num_trades[strategy_name] += 1

            # Calculate the Kelly fraction for the bet size
            f_i = self.kelly_criterion(strategy_name)

            # Calculate the profit or loss for the trade
            profit = self.calculate_profit(strategy_name, trade_direction, bid_price, ask_price, f_i, use_kelly)
            
            self.total_profit_or_loss[strategy_name] += profit

            # Update win/loss counters and totals
            if profit > 0:
                self.num_wins[strategy_name] += 1
                self.total_gains[strategy_name] += abs(profit)
            elif profit < 0:
                self.num_losses[strategy_name] += 1
                self.total_losses[strategy_name] += abs(profit)

        if(self.num_trades[strategy_name] == 0):
            self.num_trades[strategy_name] = 1

        # Close any remaining open positions for all strategies
        if self.open_positions[strategy_name]['type'] is not None:
            final_ratio = actual_rates[-1]  # Use the last ratio for closing
            profit = self.close_position(strategy_name, bid_price, ask_price)
            self.total_profit_or_loss[strategy_name] += profit

        ''' 
        -----------------------------------------
        Ensemble with LLM Mean Reversion

        '''
        print("-----------------------------------------")
        print("Ensemble with LLM Mean Reversion")
        print("-----------------------------------------")
        strategy_name = "ensemble_with_llm_mean_reversion"
        
        historical_data = []
        
        split_idx = len(actual_rates)//2

        mean_reversion_cumulative_profit = 0
        forecast_cumulative_profit = 0
        llm_sentiment_cumulative_profit = 0
        max_cumulative_profit = 0

        for i in range(2, split_idx - 1):
            curr_ratio = actual_rates[i - 1]
            prev_ratio = actual_rates[i - 2]
            predicted_next_ratio = predicted_rates[i]
            actual_next_ratio = actual_rates[i]
            llm_sentiment = llm_sentiments[i - 1]

            # Avoid division by zero
            if prev_ratio != 0 and curr_ratio != 0:
                # Calculate percentage increase instead of ratio change
                predicted_percentage_increase = ((predicted_next_ratio - curr_ratio) / curr_ratio) * 100
                actual_percentage_increase = ((actual_next_ratio - curr_ratio) / curr_ratio) * 100
                base_percentage_increase = ((curr_ratio - prev_ratio) / prev_ratio) * 100
            else:
                print("Skipping iteration due to zero division risk.")

            signals = self.get_strategy_signals(base_percentage_increase, predicted_percentage_increase, curr_ratio, actual_next_ratio, llm_sentiment)
            mean_reversion_cumulative_profit += signals['mean_reversion']
            forecast_cumulative_profit += signals['pure_forecasting']
            llm_sentiment_cumulative_profit += signals['llm']

            # Calculate profit using fixed position size
            profit, _ = self.calculate_profit_for_signals(curr_ratio, actual_next_ratio)
            max_cumulative_profit += profit
            feature_vector = [mean_reversion_cumulative_profit, forecast_cumulative_profit, llm_sentiment_cumulative_profit]

            # Store the data point
            historical_data.append((feature_vector, max_cumulative_profit))

        # Train linear regression model and update coefficients only if training succeeds
        self.train_linear_regression(historical_data, self.linear_model_llm_mean_reversion)

        self.mean_reversion_coeff_llm = self.linear_model_llm_mean_reversion.coef_[0]
        self.trend_coeff_llm = 0
        self.forecasting_coeff_llm = self.linear_model_llm_mean_reversion.coef_[1]
        self.llm_sentiment_coeff = self.linear_model_llm_mean_reversion.coef_[2]

        print(f"mean_reversion_coeff_llm : {self.mean_reversion_coeff_llm}")
        print(f"trend_coeff_llm : {self.trend_coeff_llm}")
        print(f"forecasting_coeff_llm : {self.forecasting_coeff_llm}")
        print(f"llm_sentiment_coeff : {self.llm_sentiment_coeff}")

        for i in range(split_idx + 2, len(actual_rates) - 1):
            curr_ratio = actual_rates[i - 1]
            prev_ratio = actual_rates[i - 2]
            predicted_next_ratio = predicted_rates[i]
            actual_next_ratio = actual_rates[i]
            llm_sentiment = llm_sentiments[i - 1]
            bid_price = bid_prices[i]
            ask_price = ask_prices[i]

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

            # Determine the trade direction based on the strategy and ratio changes
            trade_direction = self.get_weighted_trade_direction(base_percentage_increase, predicted_percentage_increase, llm_sentiment)

            if(trade_direction != "no_trade"):
                self.num_trades[strategy_name] += 1

            # Calculate the Kelly fraction for the bet size
            f_i = self.kelly_criterion(strategy_name)

            # Calculate the profit or loss for the trade
            profit = self.calculate_profit(strategy_name, trade_direction, bid_price, ask_price, f_i, use_kelly)
            
            self.total_profit_or_loss[strategy_name] += profit

            # Update win/loss counters and totals
            if profit > 0:
                self.num_wins[strategy_name] += 1
                self.total_gains[strategy_name] += abs(profit)
            elif profit < 0:
                self.num_losses[strategy_name] += 1
                self.total_losses[strategy_name] += abs(profit)

        if(self.num_trades[strategy_name] == 0):
            self.num_trades[strategy_name] = 1

        # Close any remaining open positions for all strategies
        if self.open_positions[strategy_name]['type'] is not None:
            final_ratio = actual_rates[-1]  # Use the last ratio for closing
            profit = self.close_position(strategy_name, bid_price, ask_price)
            self.total_profit_or_loss[strategy_name] += profit

        ''' 
        -----------------------------------------
        Other Strategies

        '''

        strategy_names = ['mean_reversion', 'trend', 'pure_forecasting']
        
        for strategy_name in strategy_names:
            for i in range(split_idx + 2, len(actual_rates) - 1):
                curr_ratio = actual_rates[i - 1]
                prev_ratio = actual_rates[i - 2]
                predicted_next_ratio = predicted_rates[i]
                bid_price = bid_prices[i]
                ask_price = ask_prices[i]

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
                trade_direction = self.determine_trade_direction(strategy_name, base_percentage_increase, predicted_percentage_increase, 0)

                if(trade_direction != "no_trade"):
                    self.num_trades[strategy_name] += 1

                # Calculate the profit or loss for the trade
                profit = self.calculate_profit(strategy_name, trade_direction, bid_price, ask_price, f_i, use_kelly)

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
                profit = self.close_position(strategy_name, bid_price, ask_price)
                self.total_profit_or_loss[strategy_name] += profit

        # Display the overall results after simulation
        self.display_total_profit()
        self.display_final_wallet_amount()
        self.display_profit_per_trade()
        self.diplay_num_trades()
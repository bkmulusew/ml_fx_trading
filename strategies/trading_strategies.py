import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from utils import TradingUtils

class TradingStrategy():
    """Trading Strategy for Supervised Learning based models, implementing different trading strategies using Kelly criterion for optimal bet sizing."""
    def __init__(self, wallet_a, wallet_b, frac_kelly):
        """Initialize the TradingStrategy class with the initial wallet balances and Kelly fraction option."""
        self.frac_kelly = frac_kelly
        # Initialize wallets for different trading strategies
        self.wallet_a = {'mean_reversion': wallet_a, 'trend': wallet_a, 'pure_forcasting': wallet_a, 'hybrid_mean_reversion': wallet_a, 'hybrid_trend': wallet_a, 'ensemble': wallet_a}
        self.wallet_b = {'mean_reversion': wallet_b, 'trend': wallet_b, 'pure_forcasting': wallet_b, 'hybrid_mean_reversion': wallet_b, 'hybrid_trend': wallet_b, 'ensemble': wallet_b}
        # Track profit/loss, wins/losses, and total gains/losses for each strategy
        self.total_profit_or_loss = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensemble': 0}
        self.num_trades = {'mean_reversion': 1, 'trend': 1, 'pure_forcasting': 1, 'hybrid_mean_reversion': 1, 'hybrid_trend': 1, 'ensemble': 1}
        self.num_wins = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensemble': 0}
        self.num_losses = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensemble': 0}
        self.total_gains = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensemble': 0}
        self.total_losses = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensemble': 0}
        self.trade_returns = {
            'mean_reversion': [],
            'trend': [],
            'pure_forcasting': [],
            'hybrid_mean_reversion': [],
            'hybrid_trend': [],
            'ensemble': []
        }
        self.sharpe_ratios = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensemble': 0}

        # New: Track open positions
        self.open_positions = {
            'mean_reversion': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'trend': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'pure_forcasting': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'hybrid_mean_reversion': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'hybrid_trend': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'ensemble': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
        }

        self.min_trades_for_full_kelly = 50  # Minimum trades before using full Kelly
        self.fixed_position_size = 1000  # Fixed position size for training
        
        # Initialize XGBoost models with appropriate parameters
        # self.ensemble_model = LogisticRegression(multi_class='multinomial')
        self.ensemble_model = XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            eval_metric='mlogloss',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        self.ensemble_scaler = StandardScaler()
        self.default_ensemble_prediction = None

        self.label_mapping = {
            'no_trade': 0,
            'buy_currency_a': 1,
            'sell_currency_a': 2
        }
    
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
        win_loss_ratio = self.win_loss_ratio(strategy_name)
        total_trades = self.num_wins[strategy_name] + self.num_losses[strategy_name]

        # Basic Kelly calculation
        kelly = win_prob - ((1 - win_prob) / win_loss_ratio)

        # Single confidence adjustment based on trade count
        confidence = min(1.0, total_trades / self.min_trades_for_full_kelly)
        kelly *= confidence

        # Return bounded result
        return max(0.01, min(0.25, kelly))

    def calculate_profit(self, strategy_name, trade_direction, bid_price, ask_price, f_i, use_kelly, enable_transaction_costs, hold_position):
        """Calculate profit/loss and handle position management"""
        # Determine pricing based on transaction costs setting
        if enable_transaction_costs:
            buy_price = ask_price
            sell_price = bid_price
        else:
            mid_price = (bid_price + ask_price) / 2
            buy_price = sell_price = mid_price

        profit_in_base_currency = 0.0
        
        # Check if there's an open position
        if self.open_positions[strategy_name]['type'] is not None:
            if(hold_position):
                # If new trade direction is different from current position type, close the position
                current_position_type = 'long' if self.open_positions[strategy_name]['type'] == 'long' else 'short'
                new_position_type = 'long' if trade_direction == 'buy_currency_a' else 'short' if trade_direction == 'sell_currency_a' else None
                
                if new_position_type is not None and new_position_type != current_position_type:
                    profit_in_base_currency += self.close_position(strategy_name, sell_price, buy_price)
                else:
                    # If same type or no trade, don't make a new trade
                    return profit_in_base_currency
            else:
                # If hold position is not enabled, close the position
                profit_in_base_currency += self.close_position(strategy_name, sell_price, buy_price)
        
        # Then open new position if there's a trade signal and no matching position type
        if trade_direction != 'no_trade':
            # Calculate total portfolio value in currency A
            total_value_in_a = self.wallet_a[strategy_name] + (self.wallet_b[strategy_name] / buy_price)

            if(use_kelly):
                base_bet_size_a = f_i * total_value_in_a
            else:
                base_bet_size_a = self.fixed_position_size
            
            if trade_direction == 'buy_currency_a':
                bet_size_a = min(base_bet_size_a, self.wallet_a[strategy_name])
                bet_size_b = bet_size_a * buy_price
                
                # Check if we have enough B
                if bet_size_b <= self.wallet_b[strategy_name]:
                    self.wallet_a[strategy_name] += bet_size_a
                    self.wallet_b[strategy_name] -= bet_size_b
                    
                    self.open_positions[strategy_name] = {
                        'type': 'long',
                        'size_a': bet_size_a,
                        'size_b': bet_size_b,
                        'entry_ratio': buy_price
                    }

            elif trade_direction == 'sell_currency_a':
                bet_size_a = min(base_bet_size_a, self.wallet_a[strategy_name])
                bet_size_b = bet_size_a * sell_price
                
                if bet_size_a <= self.wallet_a[strategy_name]:
                    self.wallet_a[strategy_name] -= bet_size_a
                    self.wallet_b[strategy_name] += bet_size_b
                    
                    self.open_positions[strategy_name] = {
                        'type': 'short',
                        'size_a': bet_size_a,
                        'size_b': bet_size_b,
                        'entry_ratio': sell_price
                    }
        
        return profit_in_base_currency
    
    def close_position(self, strategy_name, sell_price, buy_price):
        """Close an open position and calculate profit/loss"""
        position = self.open_positions[strategy_name]
        profit_in_base_currency = 0.0
        
        if position['type'] == 'long':
            # Close long position (sell currency A)
            # Entry: BUY at ASK (entry_ratio)
            # Exit: SELL at BID
            exit_amount_b = position['size_a'] * sell_price
            self.wallet_a[strategy_name] -= position['size_a']
            self.wallet_b[strategy_name] += exit_amount_b
            profit_in_base_currency = position['size_a'] * (sell_price - position['entry_ratio']) / buy_price
            
        elif position['type'] == 'short':
            # Close short position (buy currency A)
            # Entry: SELL at BID (entry_ratio)
            # Exit: BUY at ASK
            exit_amount_a = position['size_b'] / buy_price
            self.wallet_b[strategy_name] -= position['size_b']
            self.wallet_a[strategy_name] += exit_amount_a
            profit_in_base_currency = position['size_a'] * (position['entry_ratio'] - buy_price) / buy_price
        
        # Reset position tracking
        self.open_positions[strategy_name] = {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0}
        self.num_trades[strategy_name] += 1
        
        return profit_in_base_currency

    def determine_trade_direction(self, strategy_name, base_ratio_change, predicted_ratio_change, base_lower_threshold, 
                                  base_upper_threshold, predicted_lower_threshold, predicted_upper_threshold, llm_sentiment):
        """Determine the trade direction based on strategy and ratio changes."""
        trade_direction = 'no_trade'

        if(strategy_name == "mean_reversion"):
            if base_ratio_change < base_lower_threshold:
                trade_direction = 'buy_currency_a'
            elif base_ratio_change > base_upper_threshold:
                trade_direction = 'sell_currency_a'

        elif(strategy_name == "trend"):
            if base_ratio_change < base_lower_threshold:
                trade_direction = 'sell_currency_a'
            elif base_ratio_change > base_upper_threshold:
                trade_direction = 'buy_currency_a'

        elif(strategy_name == "pure_forcasting"):
            if predicted_ratio_change < predicted_lower_threshold:
                trade_direction = 'sell_currency_a'
            elif predicted_ratio_change > predicted_upper_threshold:
                trade_direction = 'buy_currency_a'

        elif(strategy_name == "hybrid_mean_reversion"):
            if base_ratio_change < base_lower_threshold and predicted_ratio_change > predicted_upper_threshold:
                trade_direction = 'buy_currency_a'
            elif base_ratio_change > base_upper_threshold and predicted_ratio_change < predicted_lower_threshold:
                trade_direction = 'sell_currency_a'

        elif(strategy_name == "hybrid_trend"):
            if base_ratio_change < base_lower_threshold and predicted_ratio_change < predicted_lower_threshold:
                trade_direction = 'sell_currency_a'
            elif base_ratio_change > base_upper_threshold and predicted_ratio_change > predicted_upper_threshold:
                trade_direction = 'buy_currency_a'

        elif(strategy_name == 'llm'):
            if(llm_sentiment == -1):
                trade_direction = 'sell_currency_a'
            elif(llm_sentiment == 1):
                trade_direction = 'buy_currency_a'
            
        return trade_direction
    
    def train_ensemble_model(self, historical_data):
        """Train the XGBoost model on historical data."""
        if not historical_data:
            print("No historical data provided.")
            self.default_ensemble_prediction = 0  # or None, depending on your system
            return
    
        X, y = zip(*historical_data)
        X = list(X)
        y = list(y)

        unique_labels = sorted(list(set(y)))
        num_unique_labels = len(unique_labels)
        print(f"Classes found in ensemble training data: {unique_labels}")

        if num_unique_labels == 1:
            # Not enough class diversity, fallback to majority class
            majority_class = unique_labels[0]
            print(f"Only one class ({majority_class}) present — defaulting to that.")
            self.default_ensemble_prediction = majority_class
            return
        
        # Add dummy data for the ensemble model to train on so that all the classes are represented
        X.append([0, 0])
        y.append(0)
        X.append([1, 1])
        y.append(1)
        X.append([2, 2])
        y.append(2)
        
        # Proceed with model training
        X = np.array(X)
        y = np.array(y)
        # X_scaled = self.ensemble_scaler.fit_transform(X)

        # self.ensemble_model.fit(X_scaled, y)
        self.ensemble_model.fit(X, y)
        self.default_ensemble_prediction = None

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
    
    def _generate_training_data(self, actual_rates_train, predicted_rates_train, llm_sentiments_train):
        """Helper method to generate training data for ensemble model."""
        historical_data = []
        base_percentage_increases, predicted_percentage_increases = TradingUtils.calculate_percentage_increases(
            actual_rates_train, predicted_rates_train
        )
        
        # Calculate Bollinger bands
        _, _, base_upper_bands, base_lower_bands = TradingUtils.calculate_bollinger_bands_for_percentages(base_percentage_increases)
        _, _, pred_upper_bands, pred_lower_bands = TradingUtils.calculate_bollinger_bands_for_percentages(predicted_percentage_increases)

        for i in range(1, len(actual_rates_train) - 1):
            curr_ratio = actual_rates_train[i]
            actual_next_ratio = actual_rates_train[i + 1]

            base_percentage_increase = base_percentage_increases[i]
            predicted_percentage_increase = predicted_percentage_increases[i]
            base_lower_threshold = base_lower_bands[i]
            base_upper_threshold = base_upper_bands[i]
            pred_lower_threshold = pred_lower_bands[i]
            pred_upper_threshold = pred_upper_bands[i]

            # Generate predictions for all base strategies
            predicted_trades = []
            for strategy in ['mean_reversion', 'pure_forcasting']:
                trade_direction = self.determine_trade_direction(
                    strategy, base_percentage_increase, predicted_percentage_increase, 
                    base_lower_threshold, base_upper_threshold, pred_lower_threshold, pred_upper_threshold, 0
                )
                predicted_trades.append(self.label_mapping[trade_direction])

            # Determine correct trade direction
            correct_trade_direction = self.label_mapping["no_trade"]
            if actual_next_ratio > curr_ratio:
                correct_trade_direction = self.label_mapping["buy_currency_a"]
            elif actual_next_ratio < curr_ratio:
                correct_trade_direction = self.label_mapping["sell_currency_a"]

            historical_data.append((predicted_trades, correct_trade_direction))

        return historical_data
        
    def _execute_trading_strategy(self, strategy_name, actual_rates, predicted_rates, bid_prices, ask_prices, 
                                 use_kelly, enable_transaction_costs, hold_position):
        """Helper method to execute trading for a specific strategy."""
        base_percentage_increases, predicted_percentage_increases = TradingUtils.calculate_percentage_increases(
            actual_rates, predicted_rates
        )
        
        # Calculate Bollinger bands
        _, _, base_upper_bands, base_lower_bands = TradingUtils.calculate_bollinger_bands_for_percentages(base_percentage_increases)
        _, _, pred_upper_bands, pred_lower_bands = TradingUtils.calculate_bollinger_bands_for_percentages(predicted_percentage_increases)
        
        for i in range(1, len(actual_rates) - 1):
            curr_bid_price = bid_prices[i]
            curr_ask_price = ask_prices[i]

            base_percentage_increase = base_percentage_increases[i]
            predicted_percentage_increase = predicted_percentage_increases[i]
            base_lower_threshold = base_lower_bands[i]
            base_upper_threshold = base_upper_bands[i]
            pred_lower_threshold = pred_lower_bands[i]
            pred_upper_threshold = pred_upper_bands[i]

            # Calculate Kelly fraction
            f_i = self.kelly_criterion(strategy_name)
            
            # Determine trade direction
            trade_direction = self.determine_trade_direction(
                strategy_name, base_percentage_increase, predicted_percentage_increase,
                base_lower_threshold, base_upper_threshold, pred_lower_threshold, pred_upper_threshold, 0
            )

            # Execute trade and calculate profit
            profit = self.calculate_profit(strategy_name, trade_direction, curr_bid_price, curr_ask_price, 
                                         f_i, use_kelly, enable_transaction_costs, hold_position)

            # Update tracking variables
            self._update_strategy_metrics(strategy_name, profit)
        
    def _execute_ensemble_strategy(self, actual_rates_test, predicted_rates_test, bid_prices_test, 
                                  ask_prices_test, use_kelly, enable_transaction_costs, hold_position):
        """Helper method to execute ensemble trading strategy."""
        strategy_name = "ensemble"
        base_percentage_increases, predicted_percentage_increases = TradingUtils.calculate_percentage_increases(
            actual_rates_test, predicted_rates_test
        )
        
        # Use pre-calculated bands from training (simplified for this example)
        _, _, base_upper_bands, base_lower_bands = TradingUtils.calculate_bollinger_bands_for_percentages(base_percentage_increases)
        _, _, pred_upper_bands, pred_lower_bands = TradingUtils.calculate_bollinger_bands_for_percentages(predicted_percentage_increases)

        for i in range(1, len(actual_rates_test) - 1):
            curr_bid_price = bid_prices_test[i]
            curr_ask_price = ask_prices_test[i]

            base_percentage_increase = base_percentage_increases[i]
            predicted_percentage_increase = predicted_percentage_increases[i]
            base_lower_threshold = base_lower_bands[i]
            base_upper_threshold = base_upper_bands[i]
            pred_lower_threshold = pred_lower_bands[i]
            pred_upper_threshold = pred_upper_bands[i]

            # Generate predictions for all base strategies
            predicted_trades = []
            for strategy in ['mean_reversion', 'pure_forcasting']:
                trade_direction = self.determine_trade_direction(
                    strategy, base_percentage_increase, predicted_percentage_increase,
                    base_lower_threshold, base_upper_threshold, pred_lower_threshold, pred_upper_threshold, 0
                )
                predicted_trades.append(self.label_mapping[trade_direction])

            # Get ensemble prediction
            if self.default_ensemble_prediction is not None:
                predicted_trade_direction = self.default_ensemble_prediction
            else:
                feature_vector = np.array(predicted_trades).reshape(1, -1)
                # feature_vector = self.ensemble_scaler.transform(feature_vector)
                predicted_trade_direction = self.ensemble_model.predict(feature_vector)[0]

            # Convert to trade direction
            if predicted_trade_direction == self.label_mapping["buy_currency_a"]:
                trade_direction = 'buy_currency_a'
            elif predicted_trade_direction == self.label_mapping["sell_currency_a"]:
                trade_direction = 'sell_currency_a'
            else:
                trade_direction = 'no_trade'

            # Calculate Kelly fraction and execute trade
            f_i = self.kelly_criterion(strategy_name)
            profit = self.calculate_profit(strategy_name, trade_direction, curr_bid_price, curr_ask_price, 
                                         f_i, use_kelly, enable_transaction_costs, hold_position)
            
            # Update tracking variables
            self._update_strategy_metrics(strategy_name, profit)
        
    def _update_strategy_metrics(self, strategy_name, profit):
        """Helper method to update strategy performance metrics."""
        self.total_profit_or_loss[strategy_name] += profit
        self.trade_returns[strategy_name].append(profit)

        # Update win/loss counters and totals
        if profit > 0:
            self.num_wins[strategy_name] += 1
            self.total_gains[strategy_name] += abs(profit)
        elif profit < 0:
            self.num_losses[strategy_name] += 1
            self.total_losses[strategy_name] += abs(profit)
        
    def _close_all_remaining_positions(self, strategy_names, bid_prices, ask_prices, enable_transaction_costs):
        """Helper method to close any remaining open positions for all strategies."""
        for strategy_name in strategy_names:
            if self.open_positions[strategy_name]['type'] is not None:
                sell_price = bid_prices[-1]
                buy_price = ask_prices[-1]
                
                if not enable_transaction_costs:
                    mid_price = (sell_price + buy_price) / 2
                    buy_price = sell_price = mid_price
                    
                profit = self.close_position(strategy_name, sell_price, buy_price)
                self.total_profit_or_loss[strategy_name] += profit
                self.trade_returns[strategy_name].append(profit)

    def simulate_trading_with_strategies(self, actual_rates, predicted_rates, bid_prices, ask_prices, llm_sentiments, use_kelly=True, enable_transaction_costs=False, hold_position=False):
        """Simulate trading over a series of exchange rates using different strategies."""

        strategy_name = "ensemble"
        
        # Phase 1: Data preparation and splitting
        split_idx = len(actual_rates) // 2
        
        actual_rates_train = actual_rates[:split_idx]
        predicted_rates_train = predicted_rates[:split_idx]
        llm_sentiments_train = llm_sentiments[:split_idx]
        
        actual_rates_test = actual_rates[split_idx:]
        predicted_rates_test = predicted_rates[split_idx:]
        bid_prices_test = bid_prices[split_idx:]
        ask_prices_test = ask_prices[split_idx:]
        llm_sentiments_test = llm_sentiments[split_idx:]

        # Phase 2: Train ensemble model
        print("Training ensemble model...")
        historical_data = self._generate_training_data(actual_rates_train, predicted_rates_train, llm_sentiments_train)
        print(f"Length of historical_data: {len(historical_data)}")
        self.train_ensemble_model(historical_data)

        # Phase 3: Execute trading strategies
        print("Executing ensemble strategy...")
        self._execute_ensemble_strategy(actual_rates_test, predicted_rates_test, bid_prices_test, 
                                       ask_prices_test, use_kelly, enable_transaction_costs, hold_position)
        
        print("Executing base strategies...")
        base_strategy_names = ['mean_reversion', 'trend', 'pure_forcasting', 'hybrid_mean_reversion', 'hybrid_trend']
        for strategy_name in base_strategy_names:
            self._execute_trading_strategy(strategy_name, actual_rates_test, predicted_rates_test, 
                                         bid_prices_test, ask_prices_test, use_kelly, enable_transaction_costs, hold_position)
            
         # Phase 4: Close remaining positions and calculate results
        all_strategy_names = base_strategy_names + ['ensemble']
        self._close_all_remaining_positions(all_strategy_names, bid_prices_test, ask_prices_test, enable_transaction_costs)

        # Calculate Sharpe ratios for selected strategies
        selected_strategies = ['mean_reversion', 'trend', 'pure_forcasting', 'ensemble']
        for strategy_name in selected_strategies:
            self.sharpe_ratios[strategy_name] = TradingUtils.calculate_sharpe_ratio(self.trade_returns[strategy_name])

        # Display results
        print("=== Trading Simulation Results ===")
        self.display_total_profit()
        self.diplay_num_trades()
        print("\n")
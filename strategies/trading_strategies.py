import numpy as np
from xgboost import XGBClassifier
from utils import TradingUtils
from collections import Counter

class TradingStrategy():
    """Trading Strategy for Supervised Learning based models, implementing different trading strategies using Kelly criterion for optimal bet sizing."""
    def __init__(self, wallet_a, wallet_b):
        """Initialize the TradingStrategy class with the initial wallet balances and Kelly fraction option."""
        # Initialize wallets for different trading strategies
        self.wallet_a = {'mean_reversion': wallet_a, 'trend': wallet_a, 'pure_forcasting': wallet_a, 'hybrid_mean_reversion': wallet_a, 'hybrid_trend': wallet_a, 'news_sentiment': wallet_a, 'ensemble': wallet_a}
        self.wallet_b = {'mean_reversion': wallet_b, 'trend': wallet_b, 'pure_forcasting': wallet_b, 'hybrid_mean_reversion': wallet_b, 'hybrid_trend': wallet_b, 'news_sentiment': wallet_b, 'ensemble': wallet_b}
        # Track profit/loss, wins/losses, and total gains/losses for each strategy
        self.total_profit_or_loss = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'news_sentiment': 0, 'ensemble': 0}
        self.num_trades = {'mean_reversion': 1, 'trend': 1, 'pure_forcasting': 1, 'hybrid_mean_reversion': 1, 'hybrid_trend': 1, 'news_sentiment': 1, 'ensemble': 1}
        self.num_wins = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'news_sentiment': 0, 'ensemble': 0}
        self.num_losses = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'news_sentiment': 0, 'ensemble': 0}
        self.total_gains = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'news_sentiment': 0, 'ensemble': 0}
        self.total_losses = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'news_sentiment': 0, 'ensemble': 0}
        self.trade_returns = {
            'mean_reversion': [],
            'trend': [],
            'pure_forcasting': [],
            'hybrid_mean_reversion': [],
            'hybrid_trend': [],
            'news_sentiment': [],
            'ensemble': []
        }
        self.sharpe_ratios = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'news_sentiment': 0, 'ensemble': 0}

        # New: Track open positions
        self.open_positions = {
            'mean_reversion': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'trend': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'pure_forcasting': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'hybrid_mean_reversion': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'hybrid_trend': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'news_sentiment': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'ensemble': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
        }

        self.min_trades_for_full_kelly = 30  # Minimum trades before using full Kelly
        self.fixed_position_size = 10000  # Fixed position size for training
        self.kelly_fraction = 0.5 # Fraction of Kelly to use
        
        # Initialize XGBoost models with appropriate parameters
        self.ensemble_model = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=42,
        )

        self.default_ensemble_pred = None

        self.label_mapping = {
            'no_trade': 0,
            'buy_currency_a': 1,
            'sell_currency_a': 2
        }
    
    def win_loss_ratio(self, strategy_name):
        """Calculate the win/loss ratio for a strategy with basic smoothing."""
        total_trades = self.num_wins[strategy_name] + self.num_losses[strategy_name]
        
        if total_trades < self.min_trades_for_full_kelly:
            return 1.05  # Conservative default
        
        # Calculate averages with basic error handling
        avg_gain = (self.total_gains[strategy_name] / self.num_wins[strategy_name]) if self.num_wins[strategy_name] else 1.0
        avg_loss = (self.total_losses[strategy_name] / self.num_losses[strategy_name]) if self.num_losses[strategy_name] else 1.0
        win_loss_ratio = avg_gain / avg_loss

        return win_loss_ratio

    def win_probability(self, strategy_name):
        """Calculate win probability with basic statistical adjustment."""
        total_trades = self.num_wins[strategy_name] + self.num_losses[strategy_name]
        
        if total_trades < self.min_trades_for_full_kelly:
            return 0.5  # Conservative default
            
        # Calculate win rate
        win_rate = self.num_wins[strategy_name] / total_trades
        
        return win_rate

    def kelly_criterion(self, strategy_name):
        """Calculate Kelly fraction with basic risk controls."""
        # Get core metrics
        win_prob = self.win_probability(strategy_name)
        win_loss_ratio = self.win_loss_ratio(strategy_name)

        f = win_prob - ((1 - win_prob) / win_loss_ratio) # Basic Kelly calculation
        f = max(0.005, f) # Ensure kelly is non-negative
        f *= self.kelly_fraction # Fractional Kelly

        return f

    def execute_trade(self, strategy_name, trade_direction, bid_price, ask_price, f_i, use_kelly, enable_transaction_costs, hold_position):
        """Calculate profit/loss and handle position management"""
        # Determine pricing based on transaction costs setting
        if enable_transaction_costs:
            buy_price = ask_price
            sell_price = bid_price
        else:
            mid_price = (bid_price + ask_price) / 2
            buy_price = sell_price = mid_price
        
        # Check if there's an open position
        if self.open_positions[strategy_name]['type'] is not None:
            if hold_position:
                # If new trade direction is different from current position type, close the position
                current_position_type = self.open_positions[strategy_name]['type']
                new_position_type = 'long' if trade_direction == 'buy_currency_a' else 'short' if trade_direction == 'sell_currency_a' else None
                
                if new_position_type is not None and new_position_type != current_position_type:
                    self.close_position(strategy_name, sell_price, buy_price)
                else:
                    # If same type or no trade, don't make a new trade
                    return
            else:
                # If hold position is not enabled, close the position
                self.close_position(strategy_name, sell_price, buy_price)
        
        # Then open new position if there's a trade signal and no matching position type
        if trade_direction != 'no_trade':
            # Calculate total portfolio value in currency A
            total_value_in_a = self.wallet_a[strategy_name] + (self.wallet_b[strategy_name] / buy_price)

            if(use_kelly):
                base_bet_size_a = f_i * total_value_in_a
            else:
                base_bet_size_a = self.fixed_position_size
            
            if trade_direction == 'buy_currency_a':
                bet_size_a = min(base_bet_size_a, self.wallet_b[strategy_name] / buy_price)
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
                else:
                    print(f"Not enough B to buy {bet_size_a} currency A")

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
                else:
                    print(f"Not enough A to sell {bet_size_a} currency A")
    
    def close_position(self, strategy_name, sell_price, buy_price):
        """Close an open position and calculate profit/loss"""
        position = self.open_positions[strategy_name]
        profit_in_curr_b = 0.0
        
        if position['type'] == 'long':
            # Close long position: sell currency A for currency B
            exit_amount_b = position['size_a'] * sell_price

            # Calculate profit in currency B terms
            profit_in_curr_b = exit_amount_b - position['size_b']  # What we got vs what we paid
    
            # Update wallets
            self.wallet_a[strategy_name] -= position['size_a']
            self.wallet_b[strategy_name] += exit_amount_b
            
        elif position['type'] == 'short':
            # Close short position: buy back currency A with currency B
            cost_to_buyback_a = position['size_a'] * buy_price
            
            # Calculate profit in currency B terms
            profit_in_curr_b = position['size_b'] - cost_to_buyback_a
            
            # Update wallets
            self.wallet_b[strategy_name] -= cost_to_buyback_a
            self.wallet_a[strategy_name] += position['size_a']
        
        # Reset position tracking
        self.open_positions[strategy_name] = {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0}

        # Update profit tracking
        self.num_trades[strategy_name] += 1
        self.total_profit_or_loss[strategy_name] += profit_in_curr_b
        self.trade_returns[strategy_name].append(profit_in_curr_b)

        if profit_in_curr_b > 0:
            self.num_wins[strategy_name] += 1
            self.total_gains[strategy_name] += profit_in_curr_b
        elif profit_in_curr_b < 0:
            self.num_losses[strategy_name] += 1
            self.total_losses[strategy_name] += abs(profit_in_curr_b)
    
    def determine_trade_direction(self, strategy_name, base_pct_change, pred_pct_change, llm_sentiment):
        """Determine the trade direction based on strategy and ratio changes."""
        trade_direction = 'no_trade'

        if(strategy_name == "mean_reversion"):
            if base_pct_change < 0:
                trade_direction = 'buy_currency_a'
            elif base_pct_change > 0:
                trade_direction = 'sell_currency_a'

        elif(strategy_name == "trend"):
            if base_pct_change < 0:
                trade_direction = 'sell_currency_a'
            elif base_pct_change > 0:
                trade_direction = 'buy_currency_a'

        elif(strategy_name == "pure_forcasting"):
            if pred_pct_change < 0:
                trade_direction = 'sell_currency_a'
            elif pred_pct_change > 0:
                trade_direction = 'buy_currency_a'

        elif(strategy_name == "hybrid_mean_reversion"):
            if base_pct_change < 0 and pred_pct_change > 0:
                trade_direction = 'buy_currency_a'
            elif base_pct_change > 0 and pred_pct_change < 0:
                trade_direction = 'sell_currency_a'

        elif(strategy_name == "hybrid_trend"):
            if base_pct_change < 0 and pred_pct_change < 0:
                trade_direction = 'sell_currency_a'
            elif base_pct_change > 0 and pred_pct_change > 0:
                trade_direction = 'buy_currency_a'

        elif(strategy_name == 'news_sentiment'):
            if(llm_sentiment == -1):
                trade_direction = 'buy_currency_a'
            elif(llm_sentiment == 1):
                trade_direction = 'sell_currency_a'
            
        return trade_direction

    def train_ensemble_model(self, historical_data):
        """Train XGB on 3 classes; inject tiny-weight synthetic samples for any missing classes."""
        if not historical_data:
            print("No historical data provided.")
            self.default_ensemble_pred = 0
            return

        # Unpack
        X, y = zip(*historical_data)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)

        # We deploy with 3 classes: 0=no_trade, 1=buy, 2=sell
        expected_classes = [0, 1, 2]
        present = set(np.unique(y))
        missing = [c for c in expected_classes if c not in present]

        # If any class missing, create a synthetic feature vector per missing class
        # Use the feature-wise median as a neutral point, then add tiny jitter
        if len(missing) > 0:
            med = np.median(X, axis=0)
            jitter_scale = 1e-6  # tiny so it won't matter numerically
            synth_X = []
            synth_y = []
            for c in missing:
                synth = med + np.random.normal(0.0, jitter_scale, size=med.shape)
                synth_X.append(synth.astype(np.float32))
                synth_y.append(c)
            X = np.vstack([X, np.vstack(synth_X)])
            y = np.concatenate([y, np.array(synth_y, dtype=int)])

        # ----- Sample weights -----
        # Base weights: class-balanced on the *real* samples only
        real_mask = np.ones(len(y), dtype=bool)
        if len(missing) > 0:
            real_mask[-len(missing):] = False  # last entries are synthetic

        y_real = y[real_mask]
        cls_counts = Counter(y_real)
        n_classes = 3
        total_real = len(y_real)

        # ----- Sample weights -----
        # Only assign tiny weights to synthetic samples, leave real samples at default weight (1.0)
        base_w = np.ones(len(y), dtype=np.float32)  # Default weight = 1.0 for all

        if len(missing) > 0:
            # Give tiny weights only to the synthetic samples (last len(missing) entries)
            base_w[-len(missing):] = 1e-6

        # Fit
        self.ensemble_model.fit(X, y, sample_weight=base_w)
        self.default_ensemble_pred = None

        # Print diagnostics
        print(f"Classes present after augmentation: {sorted(np.unique(y))} "
            f"(synthetic added for: {missing})")

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
    
    def _generate_training_data(self, actual_rates, pred_rates, llm_sentiments):
        X, y = [], []
        base_pct_incs, pred_pct_incs = TradingUtils.calculate_pct_inc(actual_rates, pred_rates)

        for i in range(1, len(actual_rates)-1):
            feature = [
                self.label_mapping["buy_currency_a"] if base_pct_incs[i] < 0 else self.label_mapping["sell_currency_a"] if base_pct_incs[i] > 0 else self.label_mapping["no_trade"],
                self.label_mapping["buy_currency_a"] if pred_pct_incs[i] > 0 else self.label_mapping["sell_currency_a"] if pred_pct_incs[i] < 0 else self.label_mapping["no_trade"],
            ]
            X.append(feature)

            # Direction label by next price (or use triple-barrier—recommended)
            y.append(self.label_mapping["buy_currency_a"] if actual_rates[i+1] > actual_rates[i] 
                     else self.label_mapping["sell_currency_a"] if actual_rates[i+1] < actual_rates[i] 
                     else self.label_mapping["no_trade"])

        return list(zip(X, y))
        
    def _execute_trading_strategy(self, strategy_name, actual_rates, pred_rates, bid_prices, ask_prices, 
                                 llm_sentiments, use_kelly, enable_transaction_costs, hold_position):
        """Helper method to execute trading for a specific strategy."""
        base_pct_incs, pred_pct_incs = TradingUtils.calculate_pct_inc(actual_rates, pred_rates)
        
        for i in range(1, len(actual_rates) - 1):
            curr_bid_price = bid_prices[i]
            curr_ask_price = ask_prices[i]

            base_pct_inc = base_pct_incs[i]
            pred_pct_inc = pred_pct_incs[i]
            llm_sentiment = llm_sentiments[i]

            # Calculate Kelly fraction
            f_i = self.kelly_criterion(strategy_name)
            
            # Determine trade direction
            trade_direction = self.determine_trade_direction(
                strategy_name, base_pct_inc, pred_pct_inc, llm_sentiment
            )

            # Execute trade
            self.execute_trade(strategy_name, trade_direction, curr_bid_price, curr_ask_price, 
                                         f_i, use_kelly, enable_transaction_costs, hold_position)

    def _execute_ensemble_strategy(self, actual_rates, pred_rates, bid_prices, ask_prices, use_kelly, enable_transaction_costs, hold_position, min_conf=0.0):
        """Helper method to execute ensemble trading strategy."""
        strategy_name = "ensemble"
        base_pct_incs, pred_pct_incs = TradingUtils.calculate_pct_inc(actual_rates, pred_rates)

        classes = [0, 1, 2]

        for i in range(1, len(actual_rates) - 1):
            feature = [
                self.label_mapping["buy_currency_a"] if base_pct_incs[i] < 0 else self.label_mapping["sell_currency_a"] if base_pct_incs[i] > 0 else self.label_mapping["no_trade"],
                self.label_mapping["buy_currency_a"] if pred_pct_incs[i] > 0 else self.label_mapping["sell_currency_a"] if pred_pct_incs[i] < 0 else self.label_mapping["no_trade"],
            ]

            # Get ensemble prediction
            if self.default_ensemble_pred is not None:
                pred_trade_direction = self.default_ensemble_pred
            else:
                feature_vector = np.array(feature).reshape(1, -1)
                pred_probs = self.ensemble_model.predict_proba(feature_vector)[0]
                pred_best_idx = int(np.argmax(pred_probs))
                pred_best_prob = float(pred_probs[pred_best_idx])
                pred_trade_direction = int(classes[pred_best_idx])

            # Convert to trade direction
            if pred_best_prob < min_conf:
                trade_direction = 'no_trade'
            elif pred_trade_direction == self.label_mapping["buy_currency_a"]:
                trade_direction = 'buy_currency_a'
            elif pred_trade_direction == self.label_mapping["sell_currency_a"]:
                trade_direction = 'sell_currency_a'
            else:
                trade_direction = 'no_trade'

            curr_bid_price = bid_prices[i]
            curr_ask_price = ask_prices[i]

            # Calculate Kelly fraction and execute trade
            f_i = self.kelly_criterion(strategy_name)
            self.execute_trade(strategy_name, trade_direction, curr_bid_price, curr_ask_price, 
                                         f_i, use_kelly, enable_transaction_costs, hold_position)
        
    def _close_all_remaining_positions(self, strategy_names, bid_prices, ask_prices, enable_transaction_costs):
        """Helper method to close any remaining open positions for all strategies."""
        for strategy_name in strategy_names:
            if self.open_positions[strategy_name]['type'] is not None:
                sell_price = bid_prices[-1]
                buy_price = ask_prices[-1]
                
                if not enable_transaction_costs:
                    mid_price = (sell_price + buy_price) / 2
                    buy_price = sell_price = mid_price
                    
                self.close_position(strategy_name, sell_price, buy_price)

    def simulate_trading_with_strategies(self, actual_rates, pred_rates, bid_prices, ask_prices, llm_sentiments, use_kelly=True, enable_transaction_costs=False, hold_position=False):
        """Simulate trading over a series of exchange rates using different strategies."""

        strategy_name = "ensemble"
        
        # Phase 1: Data preparation and splitting
        split_idx = len(actual_rates) // 2
        
        actual_rates_train = actual_rates[:split_idx]
        pred_rates_train = pred_rates[:split_idx]
        llm_sentiments_train = llm_sentiments[:split_idx]
        
        actual_rates_test = actual_rates[split_idx:]
        pred_rates_test = pred_rates[split_idx:]
        bid_prices_test = bid_prices[split_idx:]
        ask_prices_test = ask_prices[split_idx:]
        llm_sentiments_test = llm_sentiments[split_idx:]

        # Phase 2: Train ensemble model
        print("Training ensemble model...")
        historical_data = self._generate_training_data(actual_rates_train, pred_rates_train, llm_sentiments_train)
        print(f"Length of historical_data: {len(historical_data)}")
        self.train_ensemble_model(historical_data)

        # Phase 3: Execute trading strategies
        print("Executing ensemble strategy...")
        self._execute_ensemble_strategy(actual_rates_test, pred_rates_test, bid_prices_test, 
                                       ask_prices_test, use_kelly, enable_transaction_costs, hold_position)
        
        print("Executing base strategies...")
        base_strategy_names = ['mean_reversion', 'trend', 'pure_forcasting', 'hybrid_mean_reversion', 'hybrid_trend', 'news_sentiment']
        for strategy_name in base_strategy_names:
            self._execute_trading_strategy(strategy_name, actual_rates_test, pred_rates_test, 
                                         bid_prices_test, ask_prices_test, llm_sentiments_test, use_kelly, enable_transaction_costs, hold_position)
            
         # Phase 4: Close remaining positions and calculate results
        all_strategy_names = base_strategy_names + ['ensemble']
        self._close_all_remaining_positions(all_strategy_names, bid_prices_test, ask_prices_test, enable_transaction_costs)

        # Calculate Sharpe ratios for selected strategies
        selected_strategies = ['mean_reversion', 'trend', 'pure_forcasting', 'news_sentiment', 'ensemble']
        for strategy_name in selected_strategies:
            self.sharpe_ratios[strategy_name] = TradingUtils.calculate_sharpe_ratio(self.trade_returns[strategy_name])

        # Display results
        print("=== Trading Simulation Results ===")
        self.display_total_profit()
        self.diplay_num_trades()
        print("\n")
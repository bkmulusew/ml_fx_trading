import numpy as np
from xgboost import XGBClassifier
from utils import TradingUtils
from collections import Counter
import datetime
import csv
import os

class TradingStrategy():
    """Trading Strategy for Supervised Learning based models, implementing different trading strategies using Kelly criterion for optimal bet sizing."""
    def __init__(self, wallet_a, wallet_b, hold_minutes, use_kelly, enable_transaction_costs):
        self.use_kelly = use_kelly
        self.enable_transaction_costs = enable_transaction_costs
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
            'mean_reversion': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0, 'entry_timestamp': None, 'hold_minutes': None},
            'trend': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0, 'entry_timestamp': None, 'hold_minutes': None},
            'pure_forcasting': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0, 'entry_timestamp': None, 'hold_minutes': None},
            'hybrid_mean_reversion': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0, 'entry_timestamp': None, 'hold_minutes': None},
            'hybrid_trend': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0, 'entry_timestamp': None, 'hold_minutes': None},
            'news_sentiment': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0, 'entry_timestamp': None, 'hold_minutes': None},
            'ensemble': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0, 'entry_timestamp': None, 'hold_minutes': None},
        }

        # Minimum number of minutes to hold a position before allowing exit for news sentiment strategy
        self.hold_minutes = hold_minutes

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

    def execute_trade(self, strategy_name, fx_timestamp, trade_direction, bid_price, ask_price):
        """Calculate profit/loss and handle position management"""
        # Determine pricing based on transaction costs setting
        if self.enable_transaction_costs:
            buy_price = ask_price
            sell_price = bid_price
        else:
            mid_price = (bid_price + ask_price) / 2
            buy_price = sell_price = mid_price
        
        position = self.open_positions[strategy_name]

        # Check if there's an open position
        if position['type'] is not None:
            if self.hold_minutes == -1 or (fx_timestamp - position['entry_timestamp']) >= datetime.timedelta(minutes=position['hold_minutes']):
                # Close the position
                self.close_position(strategy_name, sell_price, buy_price)
            else:
                return
        
        # Then open new position if there's a trade signal and no matching position type
        if trade_direction != 'no_trade':
            # Calculate total portfolio value in currency A
            total_value_in_a = self.wallet_a[strategy_name] + (self.wallet_b[strategy_name] / buy_price)

            if(self.use_kelly):
                f_i = self.kelly_criterion(strategy_name)
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
                        'entry_ratio': buy_price,
                        'entry_timestamp': fx_timestamp,
                        'hold_minutes': self.hold_minutes,
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
                        'entry_ratio': sell_price,
                        'entry_timestamp': fx_timestamp,
                        'hold_minutes': self.hold_minutes,
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

            # Convert realized B-PnL to A at the close (sell) rate
            profit_in_curr_a = profit_in_curr_b / sell_price
    
            # Update wallets
            self.wallet_a[strategy_name] -= position['size_a']
            self.wallet_b[strategy_name] += exit_amount_b
            
        elif position['type'] == 'short':
            # Close short position: buy back currency A with currency B
            cost_to_buyback_a = position['size_a'] * buy_price
            
            # Calculate profit in currency B terms
            profit_in_curr_b = position['size_b'] - cost_to_buyback_a

            # Convert realized B-PnL to A at the close (buy) rate
            profit_in_curr_a = profit_in_curr_b / buy_price
            
            # Update wallets
            self.wallet_b[strategy_name] -= cost_to_buyback_a
            self.wallet_a[strategy_name] += position['size_a']
        
        # Reset position tracking
        self.open_positions[strategy_name] = {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0, 'entry_timestamp': None, 'hold_minutes': None}

        # Update profit tracking
        self.num_trades[strategy_name] += 1
        self.total_profit_or_loss[strategy_name] += profit_in_curr_a
        self.trade_returns[strategy_name].append(profit_in_curr_a)

        if profit_in_curr_a > 0:
            self.num_wins[strategy_name] += 1
            self.total_gains[strategy_name] += profit_in_curr_a
        elif profit_in_curr_a < 0:
            self.num_losses[strategy_name] += 1
            self.total_losses[strategy_name] += abs(profit_in_curr_a)
    
    def determine_trade_direction(self, strategy_name, base_pct_change, pred_pct_change):
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
            
        return trade_direction

    def determine_news_sentiment_trade_direction(self, news_sentiment):
        """Determine the trade direction based on news sentiment."""
        trade_direction = 'no_trade'
        
        if(news_sentiment == -1):
            trade_direction = 'buy_currency_a'
        elif(news_sentiment == 1):
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
    
    def _generate_training_data(self, actual_rates, pred_rates):
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
        
    def _execute_trading_strategy(self, strategy_name, fx_timestamps, actual_rates, pred_rates, bid_prices, ask_prices):
        """Helper method to execute trading for a specific strategy."""
        base_pct_incs, pred_pct_incs = TradingUtils.calculate_pct_inc(actual_rates, pred_rates)
        
        for i in range(1, len(actual_rates) - 1):
            curr_fx_timestamp = fx_timestamps[i]
            curr_bid_price = bid_prices[i]
            curr_ask_price = ask_prices[i]

            base_pct_inc = base_pct_incs[i]
            pred_pct_inc = pred_pct_incs[i]
            # llm_sentiment = llm_sentiments[i]
            
            # Determine trade direction
            trade_direction = self.determine_trade_direction(
                strategy_name, base_pct_inc, pred_pct_inc
            )

            # Execute trade
            self.execute_trade(strategy_name, curr_fx_timestamp, trade_direction, curr_bid_price, curr_ask_price)

    def _execute_ensemble_strategy(self, fx_timestamps, actual_rates, pred_rates, bid_prices, ask_prices):
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
                pred_trade_direction = int(classes[pred_best_idx])

            # Convert to trade direction
            if pred_trade_direction == self.label_mapping["buy_currency_a"]:
                trade_direction = 'buy_currency_a'
            elif pred_trade_direction == self.label_mapping["sell_currency_a"]:
                trade_direction = 'sell_currency_a'
            else:
                trade_direction = 'no_trade'

            curr_fx_timestamp = fx_timestamps[i]
            curr_bid_price = bid_prices[i]
            curr_ask_price = ask_prices[i]

            # Execute trade
            self.execute_trade(strategy_name, curr_fx_timestamp, trade_direction, curr_bid_price, curr_ask_price)

    def _maybe_close_position(self, strategy_name, fx_timestamp, bid_price, ask_price):
        if self.enable_transaction_costs:
            buy_price = ask_price
            sell_price = bid_price
        else:
            mid_price = (bid_price + ask_price) / 2
            buy_price = sell_price = mid_price
        
        position = self.open_positions[strategy_name]

        # Check if there's an open position
        if position['type'] is not None:
            if self.hold_minutes == -1 or (fx_timestamp - position['entry_timestamp']) >= datetime.timedelta(minutes=position['hold_minutes']):
                # Close the position
                self.close_position(strategy_name, sell_price, buy_price)

    def _execute_news_sentiment_strategy(self, fx_timestamps, bid_prices, ask_prices, news_timestamps, news_sentiments):
        strategy_name = "news_sentiment"
        MAX_DIFF = datetime.timedelta(seconds=90)

        j = 0  # pointer over news arrays
        n = len(news_timestamps)

        for i, curr_fx_timestamp in enumerate(fx_timestamps):
            curr_bid_price = bid_prices[i]
            curr_ask_price = ask_prices[i]

            # 1) Time-based close: fire as soon as the hold window has elapsed
            self._maybe_close_position(strategy_name, curr_fx_timestamp, curr_bid_price, curr_ask_price)

            # 2) Consume all news that occurred up to (and including) this tick
            while j < n and news_timestamps[j] <= curr_fx_timestamp:
                diff = curr_fx_timestamp - news_timestamps[j]

                if diff <= MAX_DIFF:
                    curr_news_sentiment = news_sentiments[j]
                    trade_direction = self.determine_news_sentiment_trade_direction(curr_news_sentiment)
                    # Execute trade
                    self.execute_trade(strategy_name, curr_fx_timestamp, trade_direction, curr_bid_price, curr_ask_price)

                j += 1
        
    def _close_all_remaining_positions(self, strategy_names, bid_prices, ask_prices):
        """Helper method to close any remaining open positions for all strategies."""
        for strategy_name in strategy_names:
            position = self.open_positions[strategy_name]
            if position['type'] is not None:
                sell_price = bid_prices[-1]
                buy_price = ask_prices[-1]
                
                if not self.enable_transaction_costs:
                    mid_price = (sell_price + buy_price) / 2
                    buy_price = sell_price = mid_price
                    
                # Close the position
                self.close_position(strategy_name, sell_price, buy_price)

    def simulate_trading_with_strategies(self, fx_timestamps, actual_rates, pred_rates, bid_prices, ask_prices, news_timestamps, news_sentiments):
        """Simulate trading over a series of exchange rates using different strategies."""

        # Identify all FX timestamps that fall within normal market hours (09:00–17:00)
        in_window_ts = [
            (i, fx_timestamp)
            for i, fx_timestamp in enumerate(fx_timestamps)
            if datetime.time(9, 0) <= fx_timestamp.time() <= datetime.time(17, 0)
        ]

        # Identify all news timestamps that fall within normal market hours (09:00–17:00)
        in_window_news = [
            (i, news_timestamp)
            for i, news_timestamp in enumerate(news_timestamps)
            if datetime.time(9, 0) <= news_timestamp.time() <= datetime.time(17, 0)
        ]

        # Require at least a start and end point within the market hours for FX data
        if len(in_window_ts) < 2:
            print(f"Not enough fx data found in the market hours for date {fx_timestamps[0].date()}.")
            print("\n")
            return

        # Require at least a start and end point within the market hours for news data
        news_enabled = True
        if len(in_window_news) < 2:
            print(f"Not enough news data found in the market hours for date {fx_timestamps[0].date()}. Disabling news strategy.")
            news_enabled = False

        print(f"Starting trading simulation for {fx_timestamps[0].date()}...")
        print(f"({len(in_window_ts)} FX points, {len(in_window_news)} news events in market hours)")
        
        (first_fx_idx, first_fx_timestamp), (last_fx_idx, last_fx_timestamp) = in_window_ts[0], in_window_ts[-1]

        # If news is disabled, set news slices to empty lists
        if news_enabled:
            (first_news_idx, first_news_timestamp), (last_news_idx, last_news_timestamp) = in_window_news[0], in_window_news[-1]
            news_timestamps_train = news_timestamps[:first_news_idx]
            news_sentiments_train = news_sentiments[:first_news_idx]
            news_timestamps_test = news_timestamps[first_news_idx:last_news_idx + 1]
            news_sentiments_test = news_sentiments[first_news_idx:last_news_idx + 1]
        else:
            news_timestamps_train, news_sentiments_train = [], []
            news_timestamps_test, news_sentiments_test = [], []

        # Phase 1: Train/test splits for FX data for training ensemble model
        actual_rates_train = actual_rates[:first_fx_idx]
        pred_rates_train = pred_rates[:first_fx_idx]
        
        fx_timestamps_test = fx_timestamps[first_fx_idx:last_fx_idx + 1]
        actual_rates_test = actual_rates[first_fx_idx:last_fx_idx + 1]
        pred_rates_test = pred_rates[first_fx_idx:last_fx_idx + 1]
        bid_prices_test = bid_prices[first_fx_idx:last_fx_idx + 1]
        ask_prices_test = ask_prices[first_fx_idx:last_fx_idx + 1]

        # Phase 2: Train ensemble model
        print("Training ensemble model...")
        historical_data = self._generate_training_data(actual_rates_train, pred_rates_train)
        self.train_ensemble_model(historical_data)

        # Phase 3: Execute trading strategies
        print("Executing ensemble strategy...")
        self._execute_ensemble_strategy(fx_timestamps_test, actual_rates_test, pred_rates_test, bid_prices_test, ask_prices_test)
        
        print("Executing base strategies...")
        base_strategy_names = ['mean_reversion', 'trend', 'pure_forcasting', 'hybrid_mean_reversion', 'hybrid_trend']
        for strategy_name in base_strategy_names:
            self._execute_trading_strategy(strategy_name, fx_timestamps_test, actual_rates_test, pred_rates_test, 
                                         bid_prices_test, ask_prices_test)

        if news_enabled:
            print("Executing news sentiment strategy...")
            self._execute_news_sentiment_strategy(fx_timestamps_test, bid_prices_test, ask_prices_test, news_timestamps_test, news_sentiments_test)
            
         # Phase 4: Close remaining positions and calculate results
        all_strategy_names = base_strategy_names + ['ensemble']
        if news_enabled:
            all_strategy_names.append('news_sentiment')
        self._close_all_remaining_positions(all_strategy_names, bid_prices_test, ask_prices_test)

        # Calculate Sharpe ratios for selected strategies
        for strategy_name in all_strategy_names:
            self.sharpe_ratios[strategy_name] = TradingUtils.calculate_sharpe_ratio(self.trade_returns[strategy_name])

        # Display results
        print("=== Trading Simulation Results ===")
        self.display_total_profit()
        self.diplay_num_trades()
        print("\n")
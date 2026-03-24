import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import optuna
from utils import TradingUtils
import datetime

class TradingStrategy():
    """Implements different trading strategies using Kelly criterion for optimal bet sizing."""
    def __init__(self, wallet_a, wallet_b, news_hold_minutes, bet_sizing, enable_transaction_costs, 
                 allow_news_overlap=False, optimize_ensemble=True, n_trials=50, reoptimize_interval=7,
                 kelly_window_days=None):
        self.bet_sizing = bet_sizing
        self.enable_transaction_costs = enable_transaction_costs
        self.allow_news_overlap = allow_news_overlap
        self.optimize_ensemble = optimize_ensemble
        self.n_trials = n_trials
        self.reoptimize_interval = reoptimize_interval  # Days between optimizations
        
        # Minimum number of minutes to hold a position before allowing exit for news sentiment strategy
        self.news_hold_minutes = news_hold_minutes
        self.no_hold = -1
        
        # Initialize wallets for different trading strategies
        self.wallet_a = {'mean_reversion': wallet_a, 'trend': wallet_a, 'model_driven': wallet_a, 'news_sentiment': wallet_a, 'ensemble': wallet_a}
        self.wallet_b = {'mean_reversion': wallet_b, 'trend': wallet_b, 'model_driven': wallet_b, 'news_sentiment': wallet_b, 'ensemble': wallet_b}
        
        # Track profit/loss, wins/losses, and total gains/losses for each strategy
        self.pnl = {
            'mean_reversion': [],
            'trend': [],
            'model_driven': [],
            'news_sentiment': [],
            'ensemble': []
        }
        self.num_trades = {'mean_reversion': 0, 'trend': 0, 'model_driven': 0, 'news_sentiment': 0, 'ensemble': 0}
        
        # Tracks open positions
        self.single_slot_positions = {
            'mean_reversion': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0, 'entry_timestamp': None, 'hold_minutes': self.no_hold},
            'trend': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0, 'entry_timestamp': None, 'hold_minutes': self.no_hold},
            'model_driven': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0, 'entry_timestamp': None, 'hold_minutes': self.no_hold},
            'news_sentiment': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0, 'entry_timestamp': None, 'hold_minutes': self.news_hold_minutes},
            'ensemble': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0, 'entry_timestamp': None, 'hold_minutes': self.no_hold},
        }
        self.multi_slot_positions = {
            'news_sentiment': []  # list of open positions (same shape as a single slot)
        }
        
        # Kelly criterion parameters
        self.min_trades_for_full_kelly = 120  # Minimum trades before using full Kelly
        self.fixed_position_size = 10000  # Fixed position size for training
        self.min_kelly_fraction = 0.005 # Minimum Kelly fraction to use (0.5% of portfolio value)
        self.kelly_window_days = kelly_window_days  # None = use all history
        self._kelly_day_index = 0

        strategies = ['mean_reversion', 'trend', 'model_driven', 'news_sentiment', 'ensemble']
        self._kelly_actual_outcomes = {s: [] for s in strategies}
        self._kelly_potential_outcomes = {s: [] for s in strategies}
        self._kelly_estimated_gains = {s: [] for s in strategies}

        self.dir_map = {
            'buy_currency_a': 1,
            'sell_currency_a': -1,
            'no_trade': 0,
        }

        # Default ensemble model (Ridge Regression)
        self.ensemble_model = Ridge(alpha=1.0)
        
        # Scalers for feature and target normalization
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.global_historical_data = []
        
        # Optimization tracking
        self.current_day_index = 0  # Track which day we're on
        self.last_optimization_day = -1  # Track when we last optimized
        self.best_alpha = None  # Store best alpha hyperparameter
        self.min_samples_for_optimization = 500  # Minimum samples needed before first optimization
        self.ensemble_model_trained = False  # Track if ensemble model has been trained

    def _optimize_ridge_alpha(self, X_train, y_train, seed):
        """
        Use Optuna to find best alpha hyperparameter for Ridge regression.
        Data is assumed to already be scaled.
        """
        print(f"\n🔍 Starting alpha optimization (Day {self.current_day_index}, {len(X_train)} samples)...")
        
        def objective(trial):
            alpha = trial.suggest_float('alpha', 1e-4, 100.0, log=True)
            
            # Time series cross-validation (walk-forward)
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                model = Ridge(alpha=alpha)
                model.fit(X_tr, y_tr)
                
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                scores.append(rmse)
            
            return np.mean(scores)
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=seed)
        )
        
        # Suppress Optuna's verbose output
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        self.best_alpha = study.best_params['alpha']
        
        print(f"✅ Optimization complete! Best RMSE: {study.best_value:.6f}")
        print(f"📊 Best alpha: {self.best_alpha:.6f}")
        
        return self.best_alpha

    def train_ensemble_model(self, historical_data, seed=42):
        """
        Train Ridge Regression ensemble model with feature and target scaling.
        Re-optimizes alpha hyperparameter every N days (configurable via reoptimize_interval).
        """
        n_samples = len(historical_data)
        
        # Skip training if no historical data - ensemble will make no trades
        if n_samples == 0:
            print("⚠️ No historical data available. Ensemble model will make no trades this day.")
            return
        
        # Unpack (X, y) from historical_data
        X, y = zip(*historical_data)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        
        # Fit scalers and transform data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y).ravel()
        
        # ===== PERIODIC HYPERPARAMETER OPTIMIZATION =====
        should_optimize = (
            self.optimize_ensemble and 
            n_samples >= self.min_samples_for_optimization and
            (self.current_day_index - self.last_optimization_day) >= self.reoptimize_interval
        )
        
        if should_optimize:
            print(f"\n{'='*60}")
            print(f"🔄 Re-optimization triggered!")
            print(f"   Current day: {self.current_day_index}")
            print(f"   Last optimization: Day {self.last_optimization_day}")
            print(f"   Days since last optimization: {self.current_day_index - self.last_optimization_day}")
            print(f"{'='*60}")
            
            # Use 80% of data for optimization (already scaled)
            opt_split = int(n_samples * 0.8)
            X_opt, y_opt = X_scaled[:opt_split], y_scaled[:opt_split]
            
            # Run optimization on scaled data
            best_alpha = self._optimize_ridge_alpha(X_opt, y_opt, seed)
            
            # Update model with best alpha
            self.ensemble_model = Ridge(alpha=best_alpha)
            
            # Update tracking
            self.last_optimization_day = self.current_day_index
        
        # Train model on all scaled data
        self.ensemble_model.fit(X_scaled, y_scaled)
        self.ensemble_model_trained = True

    def _get_prices(self, bid_price, ask_price):
        """Get buy and sell prices based on transaction costs setting."""
        if self.enable_transaction_costs:
            buy_price = ask_price
            sell_price = bid_price
        else:
            mid = (bid_price + ask_price) / 2
            buy_price = sell_price = mid
        return buy_price, sell_price

    def advance_kelly_day(self):
        """Call at the start of each new trading day to advance the rolling window."""
        self._kelly_day_index += 1
        self._prune_kelly_history()

    def _prune_kelly_history(self):
        """Remove records older than the rolling window to bound memory usage."""
        if self.kelly_window_days is None:
            return
        cutoff = self._kelly_day_index - self.kelly_window_days
        for s in self._kelly_actual_outcomes:
            self._kelly_actual_outcomes[s] = [(d, p) for d, p in self._kelly_actual_outcomes[s] if d > cutoff]
            self._kelly_potential_outcomes[s] = [(d, w) for d, w in self._kelly_potential_outcomes[s] if d > cutoff]
            self._kelly_estimated_gains[s] = [(d, e) for d, e in self._kelly_estimated_gains[s] if d > cutoff]

    def _windowed_kelly_stats(self, strategy_name):
        """Compute Kelly input stats from the rolling window."""
        wins, losses = 0, 0
        total_gain, total_loss = 0.0, 0.0
        for _, profit in self._kelly_actual_outcomes[strategy_name]:
            if profit > 0:
                wins += 1
                total_gain += profit
            elif profit < 0:
                losses += 1
                total_loss += abs(profit)

        potential_wins, potential_losses = 0, 0
        for _, is_win in self._kelly_potential_outcomes[strategy_name]:
            if is_win:
                potential_wins += 1
            else:
                potential_losses += 1

        est_gain_sum, est_gain_count = 0.0, 0
        for _, eg in self._kelly_estimated_gains[strategy_name]:
            est_gain_sum += eg
            est_gain_count += 1

        return wins, losses, total_gain, total_loss, potential_wins, potential_losses, est_gain_sum, est_gain_count

    def kelly_criterion(self, strategy_name, estimated_gain):
        """Calculate Kelly fraction using a rolling window of recent days."""
        wins, losses, total_gain, total_loss, potential_wins, potential_losses, est_gain_sum, est_gain_count = \
            self._windowed_kelly_stats(strategy_name)

        total_actual_trades = wins + losses

        if total_actual_trades < self.min_trades_for_full_kelly:
            return self.min_kelly_fraction

        if self.bet_sizing == 'active_kelly':
            p = wins / total_actual_trades
        elif self.bet_sizing == 'passive_kelly':
            total_potential_trades = potential_wins + potential_losses
            if total_potential_trades == 0:
                return self.min_kelly_fraction
            p = potential_wins / total_potential_trades

        q = 1 - p

        avg_gain = (total_gain / wins) if wins else 1.0
        avg_loss = (total_loss / losses) if losses else 1.0
        win_loss_ratio = avg_gain / avg_loss

        if estimated_gain == 0.0:
            return 0.0

        if est_gain_count == 0:
            return self.min_kelly_fraction

        h = abs(estimated_gain) / (est_gain_sum / est_gain_count)

        f = p - (q / (h * win_loss_ratio))

        return max(f, 0.0)

    def _track_potential_outcome(self, strategy_name, prev_trade_direction, actual_pct_change):
        """
        Track whether a hypothetical trade from the previous timestep would have been a win.
        This is called at each timestep to evaluate what would have happened if we had traded.
        """
        if prev_trade_direction == 'no_trade':
            return
        
        # Buy currency A (long) wins if price goes up (positive pct change)
        # Sell currency A (short) wins if price goes down (negative pct change)
        if (prev_trade_direction == 'buy_currency_a' and actual_pct_change > 0) or \
           (prev_trade_direction == 'sell_currency_a' and actual_pct_change < 0):
            self._kelly_potential_outcomes[strategy_name].append((self._kelly_day_index, True))
        elif (prev_trade_direction == 'buy_currency_a' and actual_pct_change < 0) or \
             (prev_trade_direction == 'sell_currency_a' and actual_pct_change > 0):
            self._kelly_potential_outcomes[strategy_name].append((self._kelly_day_index, False))

    def execute_trade(self, strategy_name, fx_timestamp, trade_direction, bid_price, ask_price, estimated_gain=None):
        """Calculate profit/loss and handle position management"""
        # Determine pricing based on transaction costs setting
        buy_price, sell_price = self._get_prices(bid_price, ask_price)

        # -------- NEWS (overlap path) --------
        if strategy_name == 'news_sentiment' and self.allow_news_overlap:
            # 1) close any expired news positions first
            still_open = []
            for pos in self.multi_slot_positions[strategy_name]:
                if pos['hold_minutes'] == self.no_hold or (fx_timestamp - pos['entry_timestamp']) >= datetime.timedelta(minutes=pos['hold_minutes']):
                    self._close_single_position(strategy_name, pos, sell_price, buy_price)
                else:
                    still_open.append(pos)
            self.multi_slot_positions[strategy_name] = still_open
            self._open_single_position(strategy_name, fx_timestamp, trade_direction, buy_price, sell_price, estimated_gain)
            return

        # -------- DEFAULT (single-slot) for all other strategies (and news when overlap disabled) --------
        position = self.single_slot_positions[strategy_name]

        # Check if there's an open position
        if position['type'] is not None:
            if position['hold_minutes'] == self.no_hold or (fx_timestamp - position['entry_timestamp']) >= datetime.timedelta(minutes=position['hold_minutes']):
                # Close the position
                self.close_position(strategy_name, sell_price, buy_price)
            else:
                return

        self._open_single_position(strategy_name, fx_timestamp, trade_direction, buy_price, sell_price, estimated_gain)

    def _open_single_position(self, strategy_name, fx_timestamp, trade_direction, buy_price, sell_price, estimated_gain):
        """Open a single-slot position for a strategy."""
        if trade_direction == 'no_trade':
            self.pnl[strategy_name].append(0.0)
            return

        # 1) Total portfolio value in currency A
        total_value_in_a = self.wallet_a[strategy_name] + (self.wallet_b[strategy_name] / buy_price)

        # 2) Position sizing (Kelly or fixed)
        if self.bet_sizing != 'fixed' and strategy_name != 'news_sentiment':
            f_i = self.kelly_criterion(strategy_name, estimated_gain)
            if f_i <= 0:
                self.pnl[strategy_name].append(0.0)
                return
            base_bet_size_a = f_i * total_value_in_a
        else:
            base_bet_size_a = self.fixed_position_size

        # 3) Direction-specific bet sizes and balance checks
        if trade_direction == 'buy_currency_a':  # LONG
            bet_size_a = min(base_bet_size_a, self.wallet_b[strategy_name] / buy_price)
            bet_size_b = bet_size_a * buy_price

            if bet_size_b > self.wallet_b[strategy_name] or bet_size_a <= 0:
                return

            # Update wallets
            self.wallet_a[strategy_name] += bet_size_a
            self.wallet_b[strategy_name] -= bet_size_b

            pos_type = 'long'
            entry_ratio = buy_price

        elif trade_direction == 'sell_currency_a':  # SHORT
            bet_size_a = min(base_bet_size_a, self.wallet_a[strategy_name])
            bet_size_b = bet_size_a * sell_price

            if bet_size_a > self.wallet_a[strategy_name] or bet_size_a <= 0:
                return

            # Update wallets
            self.wallet_a[strategy_name] -= bet_size_a
            self.wallet_b[strategy_name] += bet_size_b

            pos_type = 'short'
            entry_ratio = sell_price

        # 4) Record open position (single-slot store)
        hold_mins = (self.news_hold_minutes if strategy_name == 'news_sentiment' else self.no_hold)

        if strategy_name != 'news_sentiment':
            self._kelly_estimated_gains[strategy_name].append((self._kelly_day_index, abs(estimated_gain)))

        if strategy_name == 'news_sentiment' and self.allow_news_overlap:
            self.multi_slot_positions[strategy_name].append({
                'type': pos_type,
                'size_a': bet_size_a,
                'size_b': bet_size_b,
                'entry_ratio': entry_ratio,
                'entry_timestamp': fx_timestamp,
                'hold_minutes': hold_mins,
            })
        else:
            self.single_slot_positions[strategy_name] = {
                'type': pos_type,
                'size_a': bet_size_a,
                'size_b': bet_size_b,
                'entry_ratio': entry_ratio,
                'entry_timestamp': fx_timestamp,
                'hold_minutes': hold_mins,
            }

    def _close_single_position(self, strategy_name, position, sell_price, buy_price):
        """Close an open position and calculate profit/loss"""
        profit_in_curr_b = 0.0

        if position['type'] == 'long':
            # Close long position: sell currency A for currency B
            exit_amount_b = position['size_a'] * sell_price

            # Calculate profit in currency B terms
            profit_in_curr_b = exit_amount_b - position['size_b']

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

        if profit_in_curr_a != 0:
            self._kelly_actual_outcomes[strategy_name].append((self._kelly_day_index, profit_in_curr_a))

        # Update profit tracking
        self.num_trades[strategy_name] += 1
        self.pnl[strategy_name].append(profit_in_curr_a)

    def close_position(self, strategy_name, sell_price, buy_price):
        position = self.single_slot_positions[strategy_name]
        if position['type'] is None:
            return
        self._close_single_position(strategy_name, position, sell_price, buy_price)
        self.single_slot_positions[strategy_name] = {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0, 'entry_timestamp': None,
                                            'hold_minutes': self.news_hold_minutes if strategy_name == 'news_sentiment' else self.no_hold}

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

        elif(strategy_name == "model_driven"):
            if pred_pct_change < 0:
                trade_direction = 'sell_currency_a'
            elif pred_pct_change > 0:
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

    def _generate_training_data(self, actual_rates, pred_rates):
        X, y = [], []
        base_pct_incs, pred_pct_incs = TradingUtils.calculate_pct_inc(actual_rates, pred_rates)
        model_keys = list(pred_pct_incs.keys())

        for i in range(1, len(actual_rates) - 2):
            pred_features_i = [pred_pct_incs[m][i] for m in model_keys]
            feature = [
                base_pct_incs[i],
                *pred_features_i,
            ]
            X.append(feature)

            # Direction label by next price
            y.append(base_pct_incs[i + 1])

        return list(zip(X, y))

    def _execute_trading_strategy(self, strategy_name, fx_timestamps, actual_rates, pred_rates, bid_prices, ask_prices):
        """Helper method to execute trading for a specific strategy."""
        base_pct_incs, pred_pct_incs = TradingUtils.calculate_pct_inc(actual_rates, pred_rates)

        prev_trade_direction = None

        for i in range(1, len(actual_rates) - 1):
            curr_fx_timestamp = fx_timestamps[i]
            curr_bid_price = bid_prices[i]
            curr_ask_price = ask_prices[i]

            base_pct_inc = base_pct_incs[i]
            pred_pct_inc = pred_pct_incs[i]

            # Track potential outcome from previous timestep's hypothetical trade
            if prev_trade_direction is not None:
                self._track_potential_outcome(strategy_name, prev_trade_direction, base_pct_inc)

            # Determine trade direction
            trade_direction = self.determine_trade_direction(
                strategy_name, base_pct_inc, pred_pct_inc
            )

            # Store for next iteration's potential outcome tracking
            prev_trade_direction = trade_direction

            estimated_gain = base_pct_inc
            if strategy_name == 'model_driven':
                estimated_gain = pred_pct_inc

            # Execute trade
            self.execute_trade(strategy_name, curr_fx_timestamp, trade_direction, curr_bid_price, curr_ask_price, estimated_gain)

    def _execute_ensemble_strategy(self, fx_timestamps, actual_rates, pred_rates, bid_prices, ask_prices):
        """Helper method to execute ensemble trading strategy."""
        strategy_name = "ensemble"
        base_pct_incs, pred_pct_incs = TradingUtils.calculate_pct_inc(actual_rates, pred_rates)
        model_keys = list(pred_pct_incs.keys())

        prev_trade_direction = None

        for i in range(1, len(actual_rates) - 1):
            base_pct_inc = base_pct_incs[i]

            # Track potential outcome from previous timestep's hypothetical trade
            if prev_trade_direction is not None:
                self._track_potential_outcome(strategy_name, prev_trade_direction, base_pct_inc)

            # If model not trained yet, make no trades
            if not self.ensemble_model_trained:
                trade_direction = 'no_trade'
                estimated_gain = 0.0
            else:
                pred_features_i = [pred_pct_incs[m][i] for m in model_keys]
                feature = [
                    base_pct_inc,
                    *pred_features_i,
                ]

                feature_vector = np.array(feature).reshape(1, -1)
                # Scale features using fitted scaler
                feature_vector_scaled = self.scaler_X.transform(feature_vector)
                # Predict in scaled space
                y_pred_scaled = self.ensemble_model.predict(feature_vector_scaled)[0]
                # Inverse transform to get prediction in original scale
                y_pred = self.scaler_y.inverse_transform([[y_pred_scaled]])[0, 0]
                
                if y_pred < 0:
                    trade_direction = 'sell_currency_a'
                elif y_pred > 0:
                    trade_direction = 'buy_currency_a'
                else:
                    trade_direction = 'no_trade'
                estimated_gain = y_pred

            # Store for next iteration's potential outcome tracking
            prev_trade_direction = trade_direction

            curr_fx_timestamp = fx_timestamps[i]
            curr_bid_price = bid_prices[i]
            curr_ask_price = ask_prices[i]

            # Execute trade
            self.execute_trade(strategy_name, curr_fx_timestamp, trade_direction, curr_bid_price, curr_ask_price, estimated_gain)

    def _maybe_close_position(self, strategy_name, fx_timestamp, bid_price, ask_price):
        buy_price, sell_price = self._get_prices(bid_price, ask_price)

        if strategy_name == 'news_sentiment' and self.allow_news_overlap:
            # close any expired stacked positions
            still_open = []
            for pos in self.multi_slot_positions['news_sentiment']:
                if pos['hold_minutes'] == self.no_hold or (fx_timestamp - pos['entry_timestamp']) >= datetime.timedelta(minutes=pos['hold_minutes']):
                    self._close_single_position('news_sentiment', pos, sell_price, buy_price)
                else:
                    still_open.append(pos)
            self.multi_slot_positions['news_sentiment'] = still_open
            return

        position = self.single_slot_positions[strategy_name]

        # Check if there's an open position
        if position['type'] is not None:
            if position['hold_minutes'] == self.no_hold or (fx_timestamp - position['entry_timestamp']) >= datetime.timedelta(minutes=position['hold_minutes']):
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
            while j < n and news_timestamps[j] < curr_fx_timestamp:
                diff = curr_fx_timestamp - news_timestamps[j]

                if diff <= MAX_DIFF:
                    curr_news_sentiment = news_sentiments[j]
                    trade_direction = self.determine_news_sentiment_trade_direction(curr_news_sentiment)
                    # Execute trade
                    self.execute_trade(strategy_name, curr_fx_timestamp, trade_direction, curr_bid_price, curr_ask_price)

                j += 1

    def _close_all_remaining_positions(self, strategy_names, bid_price, ask_price):
        """Helper method to close any remaining open positions for all strategies."""
        buy_price, sell_price = self._get_prices(bid_price, ask_price)

        for strategy_name in strategy_names:
            if strategy_name == 'news_sentiment' and self.allow_news_overlap:
                # close every stacked position
                for pos in list(self.multi_slot_positions['news_sentiment']):
                    self._close_single_position('news_sentiment', pos, sell_price, buy_price)
                self.multi_slot_positions['news_sentiment'].clear()
            else:
                self.close_position(strategy_name, sell_price, buy_price)

    def simulate_trading_with_strategies(
        self,
        fx_timestamps,
        actual_rates,
        pred_rates,
        bid_prices,
        ask_prices,
        news_timestamps,
        news_sentiments,
        strategy_names=None,
    ):
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
            return

        # Require at least a start and end point within the market hours for news data
        news_enabled = True
        if len(in_window_news) < 2:
            news_enabled = False

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

        fx_timestamps_test = fx_timestamps[first_fx_idx:last_fx_idx + 1]
        actual_rates_test = actual_rates[first_fx_idx:last_fx_idx + 1]
        pred_rates_test = pred_rates[first_fx_idx:last_fx_idx + 1]
        bid_prices_test = bid_prices[first_fx_idx:last_fx_idx + 1]
        ask_prices_test = ask_prices[first_fx_idx:last_fx_idx + 1]

        # Default to all base strategies unless a subset is explicitly provided
        if strategy_names is None:
            base_strategy_names = ['mean_reversion', 'trend', 'model_driven']
        else:
            base_strategy_names = strategy_names

        for strategy_name in base_strategy_names:
            self._execute_trading_strategy(
                strategy_name,
                fx_timestamps_test,
                actual_rates_test,
                pred_rates_test,
                bid_prices_test,
                ask_prices_test,
            )

        if news_enabled:
            self._execute_news_sentiment_strategy(
                fx_timestamps_test,
                bid_prices_test,
                ask_prices_test,
                news_timestamps_test,
                news_sentiments_test,
            )

        # Phase 4: Close remaining positions and calculate results
        all_strategy_names = list(base_strategy_names)
        if news_enabled:
            all_strategy_names.append('news_sentiment')
        self._close_all_remaining_positions(all_strategy_names, bid_prices_test[-1], ask_prices_test[-1])

    def simulate_trading_with_ensemble_strategy(self, fx_timestamps, actual_rates, pred_rates, bid_prices, ask_prices, seed=42):
        """Simulate trading over a series of exchange rates using ensemble strategy."""

        # Identify all FX timestamps that fall within normal market hours (09:00–17:00)
        in_window_ts = [
            (i, fx_timestamp)
            for i, fx_timestamp in enumerate(fx_timestamps)
            if datetime.time(9, 0) <= fx_timestamp.time() <= datetime.time(17, 0)
        ]

        # Require at least a start and end point within the market hours for FX data
        if len(in_window_ts) < 2:
            return

        (first_fx_idx, first_fx_timestamp), (last_fx_idx, last_fx_timestamp) = in_window_ts[0], in_window_ts[-1]

        # Phase 1: Train/test splits for FX data for training ensemble model
        actual_rates_train_before = actual_rates[:first_fx_idx]
        pred_rates_train_before = {model_name: preds[:first_fx_idx] for model_name, preds in pred_rates.items()}

        actual_rates_train_rest = actual_rates[first_fx_idx:]
        pred_rates_train_rest = {model_name: preds[first_fx_idx:] for model_name, preds in pred_rates.items()}

        fx_timestamps_test = fx_timestamps[first_fx_idx:last_fx_idx + 1]
        actual_rates_test = actual_rates[first_fx_idx:last_fx_idx + 1]
        pred_rates_test = {model_name: preds[first_fx_idx:last_fx_idx + 1] for model_name, preds in pred_rates.items()}
        bid_prices_test = bid_prices[first_fx_idx:last_fx_idx + 1]
        ask_prices_test = ask_prices[first_fx_idx:last_fx_idx + 1]

        # Phase 2: Train ensemble model
        prior_training_data = self._generate_training_data(actual_rates_train_before, pred_rates_train_before)
        self.global_historical_data.extend(prior_training_data)
        self.train_ensemble_model(self.global_historical_data, seed)

        # Phase 3: Execute trading strategies
        self._execute_ensemble_strategy(fx_timestamps_test, actual_rates_test, pred_rates_test, bid_prices_test, ask_prices_test)

        # Phase 4: Close remaining positions and calculate results
        all_strategy_names = ['ensemble']
        self._close_all_remaining_positions(all_strategy_names, bid_prices_test[-1], ask_prices_test[-1])

        later_training_data = self._generate_training_data(actual_rates_train_rest, pred_rates_train_rest)
        self.global_historical_data.extend(later_training_data)
        
        # Increment day counter for next simulation
        self.current_day_index += 1
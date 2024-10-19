import numpy as np

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
    
    def win_loss_ratio(self, strategy_name):
        """Calculate the win/loss ratio for a given strategy."""
        if self.num_wins[strategy_name] == 0 and self.num_losses[strategy_name] == 0:
            # Initialize to 1 when there are no trades yet
            return 1
        average_gain = self.total_gains[strategy_name] / self.num_wins[strategy_name] if self.num_wins[strategy_name] else 0
        average_loss = self.total_losses[strategy_name] / self.num_losses[strategy_name] if self.num_losses[strategy_name] else 0

        return average_gain / average_loss if average_loss != 0 else float('inf')
    
    def win_probability(self, strategy_name):
        """Calculate the win probability for a given strategy."""
        if self.num_wins[strategy_name] == 0 and self.num_losses[strategy_name] == 0:
            # Initialize to 0.5 when there are no trades yet
            return 0.5
        return self.num_wins[strategy_name] / (self.num_wins[strategy_name] + self.num_losses[strategy_name])

    def kelly_criterion(self, p, q, win_loss_ratio):
        """
        Calculate the Kelly fraction using the probability of winning (p),
        probability of losing (q), and the win loss ratio.
        """
        return p - (q / win_loss_ratio) if win_loss_ratio != 0 else 0

    def calculate_profit(self, strategy_name, trade_direction, curr_ratio, next_ratio, f_i):
        """Calculate the profit or loss of a trade based on the trade direction and currency ratios."""

        # Adjust Kelly factor if fractional Kelly is chosen
        kelly_factor = 0.33 if self.frac_kelly else 1.0
        profit = 0
        if trade_direction == 'buy_currency_a':
            # Execute a 'buy_currency_a' trade, converting currency B to currency A
            bet_size = f_i * self.wallet_b[strategy_name] * kelly_factor
            currency_a_bought = bet_size / curr_ratio # Convert bet_size from currency B to currency A
            profit = (currency_a_bought * next_ratio) - bet_size # Calculate profit by selling currency A at next time step
            self.wallet_b[strategy_name] += profit # Update wallet B with the profit/loss
        elif trade_direction == 'sell_currency_a':
            # Execute a 'sell_currency_a' trade, converting currency A to currency B
            bet_size = f_i * self.wallet_a[strategy_name] * kelly_factor
            currency_b_bought = bet_size * curr_ratio # Convert bet_size from currency A to currency B
            profit = (currency_b_bought / next_ratio) - bet_size # Calculate profit by buying back currency A at next time step
            self.wallet_a[strategy_name] += profit # Update wallet A with the profit/loss
        return profit

    def determine_trade_direction(self, strategy_name, base_ratio_change, predicted_ratio_change):
        """Determine the trade direction based on strategy and ratio changes."""
        if(strategy_name == 'mean_reversion'):
            # Mean reversion strategy: trade against significant ratio changes
            if base_ratio_change > self.trade_threshold:
                return 'sell_currency_a'
            elif base_ratio_change < -self.trade_threshold:
                return 'buy_currency_a'
            else:
                return 'no_trade'
        elif(strategy_name == 'trend'):
            # Trend strategy: trade towards significant ratio changes
            if base_ratio_change > self.trade_threshold:
                return 'buy_currency_a'
            elif base_ratio_change < -self.trade_threshold:
                return 'sell_currency_a'
            else:
                return 'no_trade'
        elif(strategy_name == 'pure_forcasting'):
            # Pure forecasting strategy: trade based on predicted future ratio changes
            if predicted_ratio_change > self.trade_threshold:
                return 'buy_currency_a'
            elif predicted_ratio_change < -self.trade_threshold:
                return 'sell_currency_a'
            else:
                return 'no_trade'
        elif(strategy_name == 'hybrid_mean_reversion'):
            # Hybrid strategy: combine mean reversion and pure forecasting signals
            if base_ratio_change < -self.trade_threshold and predicted_ratio_change > self.trade_threshold:
                return 'buy_currency_a'
            elif base_ratio_change > self.trade_threshold and predicted_ratio_change < -self.trade_threshold:
                return 'sell_currency_a'
            else:
                return 'no_trade'
        elif(strategy_name == 'hybrid_trend'):
            # Hybrid strategy: combine trend and pure forecasting signals
            if base_ratio_change > self.trade_threshold and predicted_ratio_change > self.trade_threshold:
                return 'buy_currency_a'
            elif base_ratio_change < -self.trade_threshold and predicted_ratio_change < -self.trade_threshold:
                return 'sell_currency_a'
            else:
                return 'no_trade'
        else:
            return 'no_trade'
        
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
            for i in range(2, len(actual_rates)):
                curr_ratio = actual_rates[i - 1]
                prev_ratio = actual_rates[i - 2]
                predicted_next_ratio = predicted_rates[i]
                actual_next_ratio = actual_rates[i]

                # Calculate the win/loss ratio for the current strategy
                win_loss_ratio = self.win_loss_ratio(strategy_name)
            
                # Calculate win probability
                p_i = self.win_probability(strategy_name)
                q_i = 1 - p_i
                
                # Calculate the Kelly fraction for the bet size
                f_i = self.kelly_criterion(p_i, q_i, win_loss_ratio)
                f_i = max(0, f_i)  # Ensure the fraction is non-negative

                # Determine the predicted and actual ratio changes
                predicted_ratio_change = predicted_next_ratio - curr_ratio
                base_ratio_change = curr_ratio - prev_ratio
            
                # Determine the trade direction based on the strategy and ratio changes
                trade_direction = self.determine_trade_direction(strategy_name, base_ratio_change, predicted_ratio_change)

                # Calculate the profit or loss for the trade
                profit = self.calculate_profit(strategy_name, trade_direction, curr_ratio, actual_next_ratio, f_i)
                self.total_profit_or_loss[strategy_name] += profit

                # Update win/loss counters and totals
                if profit >= 0:
                    self.num_wins[strategy_name] += 1
                    self.total_gains[strategy_name] += abs(profit)
                else:
                    self.num_losses[strategy_name] += 1
                    self.total_losses[strategy_name] += abs(profit)

        # Display the overall results after simulation
        self.display_total_profit()
        self.display_final_wallet_amount()
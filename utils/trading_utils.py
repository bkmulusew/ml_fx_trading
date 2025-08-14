import numpy as np

class TradingUtils:
    """Utility class for trading strategies."""

    @staticmethod
    def calculate_bollinger_bands_for_percentages(percentage_values, window=20, std_dev=2):
        """Calculate Bollinger bands for percentage increase values."""
        n = len(percentage_values)
        ma = []
        std = []
        upper_band = []
        lower_band = []
        
        for i in range(n):
            # Use all available data up to the window size
            start_idx = max(0, i - (window - 1))
            window_data = percentage_values[start_idx:i+1]
            actual_window = len(window_data)
            
            # Calculate moving average
            current_ma = sum(window_data) / actual_window
            ma.append(current_ma)
            
            # Calculate standard deviation if we have at least 2 points
            if actual_window >= 2:
                squared_diff = [(x - current_ma) ** 2 for x in window_data]
                current_std = (sum(squared_diff) / actual_window) ** 0.5
                std.append(current_std)
                
                # Calculate bands
                current_upper = current_ma + (current_std * std_dev)
                current_lower = current_ma - (current_std * std_dev)
            else:
                # With only 1 point, std is 0, so bands equal MA
                current_std = 0.0
                std.append(current_std)
                current_upper = current_ma
                current_lower = current_ma
                
            upper_band.append(current_upper)
            lower_band.append(current_lower)
        
        return ma, std, upper_band, lower_band
        
    @staticmethod
    def calculate_percentage_increases(actual_rates, predicted_rates):
        """Calculate percentage increases for Bollinger bands."""
        base_percentage_increases = [0.0]
        predicted_percentage_increases = [0.0]
        
        for i in range(1, len(actual_rates) - 1):
            curr_ratio = actual_rates[i]
            prev_ratio = actual_rates[i - 1]
            predicted_ratio = predicted_rates[i + 1]
            
            # Avoid division by zero
            if prev_ratio != 0 and curr_ratio != 0:
                base_percentage_increase = ((curr_ratio - prev_ratio) / prev_ratio) * 100
                predicted_percentage_increase = ((predicted_ratio - curr_ratio) / curr_ratio) * 100
            else:
                print(f"Division by zero at index {i}")
                base_percentage_increase = 0.0
                predicted_percentage_increase = 0.0
                
            base_percentage_increases.append(base_percentage_increase)
            predicted_percentage_increases.append(predicted_percentage_increase)
            
        return base_percentage_increases, predicted_percentage_increases
        
    @staticmethod
    def calculate_sharpe_ratio(trade_returns, risk_free_rate=0.0):
        """Calculate Sharpe ratio based on trade returns."""
        if len(trade_returns) > 1:
            excess_returns = [r - risk_free_rate for r in trade_returns]
            mean_return = np.mean(excess_returns)
            std_return = np.std(excess_returns)
            sharpe = mean_return / std_return if std_return != 0 else 0.0
            return sharpe
        else:
            return 0.0
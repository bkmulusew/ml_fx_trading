import numpy as np

class TradingUtils:
    """Utility class for trading strategies."""

    @staticmethod
    def calculate_pct_inc(actual_rates, pred_rates):
        """
        Calculate percentage increases.
        If pred_rates is:
            - list: return (base_pct_incs, pred_pct_incs)
            - dict of lists: return (base_pct_incs, {model: pred_pct_incs})
        """
        base_pct_incs = [0.0]

        # Case 1: pred_rates is a list
        if isinstance(pred_rates, list):
            pred_pct_incs = [0.0]

            for i in range(1, len(actual_rates) - 1):
                curr_ratio = actual_rates[i]
                prev_ratio = actual_rates[i - 1]
                pred_ratio = pred_rates[i + 1]

                if prev_ratio != 0 and curr_ratio != 0:
                    base_pct_inc = (curr_ratio - prev_ratio) / prev_ratio
                    pred_pct_inc = (pred_ratio - curr_ratio) / curr_ratio
                else:
                    base_pct_inc = 0.0
                    pred_pct_inc = 0.0

                base_pct_incs.append(base_pct_inc)
                pred_pct_incs.append(pred_pct_inc)

            return base_pct_incs, pred_pct_incs

        # Case 2: pred_rates is a dict
        elif isinstance(pred_rates, dict):

            # Create storage for each model
            pred_pct_incs_dict = {k: [0.0] for k in pred_rates.keys()}

            for i in range(1, len(actual_rates) - 1):
                curr_ratio = actual_rates[i]
                prev_ratio = actual_rates[i - 1]

                if prev_ratio != 0 and curr_ratio != 0:
                    base_pct_inc = (curr_ratio - prev_ratio) / prev_ratio
                else:
                    base_pct_inc = 0.0

                base_pct_incs.append(base_pct_inc)

                # Compute pred_pct_inc separately for each model
                for k, v in pred_rates.items():
                    pred_ratio = v[i + 1]  # v is the prediction list

                    if curr_ratio != 0:
                        pred_pct = (pred_ratio - curr_ratio) / curr_ratio
                    else:
                        pred_pct = 0.0

                    pred_pct_incs_dict[k].append(pred_pct)

            return base_pct_incs, pred_pct_incs_dict

        else:
            raise TypeError("pred_rates must be a list or a dict of lists.")

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
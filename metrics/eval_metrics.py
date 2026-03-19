import numpy as np

class ModelEvalMetrics:
    def calculate_smape(self, actual, predicted):
        """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
        actual, predicted = np.asarray(actual), np.asarray(predicted)
        denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
        diff = np.abs(actual - predicted) / denominator
        diff[denominator == 0] = 0.0  # avoid division by zero
        return 100 * np.mean(diff)

    def calculate_mape(self, actual, predicted):
        """Calculates the Mean Absolute Percentage Error (MAPE)."""
        actual, predicted = np.asarray(actual), np.asarray(predicted)
        diff = np.abs((actual - predicted) / actual)
        diff[actual == 0] = 0.0  # avoid division by zero
        return 100 * np.mean(diff)

    def calculate_mase(self, actual, predicted):
        """Calculates the Mean Absolute Scaled Error (MASE)."""
        actual, predicted = np.asarray(actual), np.asarray(predicted)
        n = len(actual)
        d = np.abs(np.diff(actual)).sum() / (n - 1)
        if d == 0:
            return 0.0
        errors = np.abs(actual - predicted)
        return errors.mean() / d

    def calculate_rmse(self, actual, predicted):
        """Calculates the Root Mean Squared Error (RMSE)."""
        actual, predicted = np.asarray(actual), np.asarray(predicted)
        return np.sqrt(((predicted - actual) ** 2).mean())

    def calculate_prediction_error(self, actual, predicted):
        """Calculates the prediction error."""
        smape = self.calculate_smape(actual, predicted)
        mape = self.calculate_mape(actual, predicted)
        mase = self.calculate_mase(actual, predicted)
        rmse = self.calculate_rmse(actual, predicted)
        return {'RMSE': rmse, 'MASE': mase, 'MAPE': mape, 'sMAPE': smape}
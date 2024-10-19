from models import FinancialForecastingModel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

class BiLSTMModel(nn.Module):
    """BiLSTM with attention model."""
    def __init__(self, num_features=1):
        super(BiLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(num_features, 128, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(256, num_heads=16, dropout=0.1, batch_first=True)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.attention(x, x, x)
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PytorchFinancialForecastingModel(FinancialForecastingModel):
    """A financial forecasting model based on the Pytorch library."""
    def __init__(self, mode_name, data_processor, model_config):
        self.data_processor = data_processor
        self.scaler = None
        self.model_config = model_config
        self.model = self.initalize_model(mode_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()

    def initalize_model(self, model_name):
        """Creates the model."""
        if model_name == "bilstm":
            return BiLSTMModel()

        else:
            raise ValueError("Invalid model name.")

    def split_and_scale_data(self, train_ratio=0.5, validation_ratio=0.1):
        """Splits the data into training, validation, and test sets and applies scaling."""
        series = self.data_processor.load_and_prepare_data()
        # Select the second column dynamically
        second_column_name = series.columns[1]
        series = series[second_column_name].values
        series = series.reshape(-1, 1)

        num_observations = len(series)
        train_end_index = int(num_observations * train_ratio)
        validation_end_index = int(num_observations * (train_ratio + validation_ratio))

        train_data = series[:train_end_index]
        valid_data = series[train_end_index:validation_end_index]
        test_data = series[validation_end_index:]

        self.scaler = MinMaxScaler((0, 1))
        train_data = self.scaler.fit_transform(train_data)
        valid_data = self.scaler.transform(valid_data)
        test_data = self.scaler.transform(test_data)

        X_train = []
        y_train = []
        X_valid = []
        y_valid = []
        X_test = []
        y_test = []

        for i in range(self.model_config.INPUT_CHUNK_LENGTH, len(train_data)):
            X_train.append(train_data[i - self.model_config.INPUT_CHUNK_LENGTH : i])
            y_train.append(train_data[i + (self.model_config.OUTPUT_CHUNK_LENGTH - 1)])

        for i in range(self.model_config.INPUT_CHUNK_LENGTH, len(valid_data)):
            X_valid.append(valid_data[i - self.model_config.INPUT_CHUNK_LENGTH : i])
            y_valid.append(valid_data[i + (self.model_config.OUTPUT_CHUNK_LENGTH - 1)])

        for i in range(self.model_config.INPUT_CHUNK_LENGTH, len(test_data)):
            X_test.append(test_data[i - self.model_config.INPUT_CHUNK_LENGTH : i])
            y_test.append(test_data[i + (self.model_config.OUTPUT_CHUNK_LENGTH - 1)])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_train, y_train = shuffle(X_train, y_train)

        X_valid, y_valid = np.array(X_valid), np.array(y_valid)
        X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))

        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        return {
            'x_train': torch.FloatTensor(X_train).to(self.device),
            'y_train': torch.FloatTensor(y_train).to(self.device),
            'x_valid': torch.FloatTensor(X_valid).to(self.device),
            'y_valid': torch.FloatTensor(y_valid).to(self.device),
            'x_test': torch.FloatTensor(X_test).to(self.device),
            'y_test': torch.FloatTensor(y_test).to(self.device)
        }

    def train(self, x_train, y_train, x_valid, y_valid):
        """Trains the model."""
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.model_config.BATCH_SIZE, shuffle=True)
        
        self.model.train()
        for epoch in range(self.model_config.N_EPOCHS):
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(x_valid)
                val_loss = self.criterion(val_outputs, y_valid)
            print(f'Epoch [{epoch+1}/{self.model_config.N_EPOCHS}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            self.model.train()

    def predict_future_values(self, x_test):
        """Makes future value predictions based on the test series."""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x_test)
        return predictions.cpu().numpy()

    def generate_predictions(self, x_test, y_test):
        """Generates predictions for each window of the test series."""
        pred_series = self.predict_future_values(x_test)
        pred_series = self.scaler.inverse_transform(pred_series.reshape(-1, 1)).flatten()
        
        # Storing the predictions
        predicted_values = pred_series.tolist()
        true_values = self.scaler.inverse_transform(y_test.cpu().numpy()).flatten().tolist()
        return {'predicted_values': predicted_values, 'true_values': true_values}
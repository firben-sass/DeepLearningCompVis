import torch.nn as nn

class TemporalStream(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 256)

    def forward(self, x):
        # x shape: (batch, time, features)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])
    
    def train(self, learning_rate):
        

    def predict(self):
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

class NNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def main():
    # Load data
    df = pd.read_csv('smart_health_tracker_data.csv')
    df.fillna(df.mode().iloc[0], inplace=True)

    X = df.drop('Sleep_Quality', axis=1).values.astype(np.float32)  # Adjust target column name
    y = df['Sleep_Quality'].values.astype(np.float32)

    # Convert to tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y).view(-1, 1)

    # Create dataset and loader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model, loss & optimizer
    input_dim = X.shape[1]
    model = NNet(input_dim=input_dim, hidden_dim=64, output_dim=1)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop (simplified)
    epochs = 20
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    # After training, you can add evaluation code here

if __name__ == "__main__":
    main()

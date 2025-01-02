import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import pandas as pd


# Define the model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Single linear layer
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Sigmoid for binary classification


def load_custom_data(csv_file):
    df = pd.read_csv(csv_file)
    # Assuming the last column is the target/label and the rest are features
    X = df.iloc[:, 1:-1].values  # Features (all columns except the last one)
    y = df.iloc[:, -1].values   # Labels (the last column)
    return X, y


if __name__ == '__main__':
    # Load and preprocess data
    csv_file = 'debug/data/2nd_tab/training_data.csv'
    X, y = load_custom_data(csv_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = X_train.shape[1]
    model = LogisticRegression(input_dim)

    # Define loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()  # Convert probabilities to binary predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Test Accuracy: {100 * correct / total:.2f}%')

    sys.exit()
    # Save the model
    model_save_path = 'logistic_regression_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    loaded_model = LogisticRegression(input_dim)
    loaded_model.load_state_dict(torch.load(model_save_path))
    loaded_model.eval()
    print(f'Model loaded from {model_save_path}')

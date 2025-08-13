import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_auc_score, 
                            ConfusionMatrixDisplay, roc_curve)
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Load and preprocess data
df = pd.read_csv('Health_NN\smart_health_tracker_data.csv')

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Daily_Steps'] = df['Daily_Steps'].fillna(df['Daily_Steps'].mean())
df['Resting_Heart_Rate'] = df['Resting_Heart_Rate'].fillna(df['Resting_Heart_Rate'].median())
df['Active_Heart_Rate'] = df['Active_Heart_Rate'].fillna(df['Active_Heart_Rate'].mean())
df['Hours_of_Sleep'] = df['Hours_of_Sleep'].fillna(df['Hours_of_Sleep'].mean())
df['Daily_Calorie_Intake'] = df['Daily_Calorie_Intake'].fillna(df['Daily_Calorie_Intake'].mean())
df['Sleep_Quality'] = df['Sleep_Quality'].fillna(df['Sleep_Quality'].mean())
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Stress_Level'] = df['Stress_Level'].fillna(df['Stress_Level'].mode()[0])
df['Daily_Activity_Type'] = df['Daily_Activity_Type'].fillna(df['Daily_Activity_Type'].mode()[0])
df['Mood'] = df['Mood'].fillna(df['Mood'].mode()[0])

# Create target variable
df['Well_Rested'] = (df['Hours_of_Sleep'] >= 7).astype(int)

# Select features and target
features = ['Sleep_Quality','Stress_Level','Daily_Calorie_Intake','Active_Heart_Rate','Resting_Heart_Rate']
X = df[features].values
y = df['Well_Rested'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)  # 0.125 x 0.8 = 0.1

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create data loaders
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define enhanced MLP model
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # Initialize weights
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        return self.net(x)

# Initialize model
model = MLP(X_train.shape[1])

# Handle class imbalance
pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training with early stopping
best_val_loss = float('inf')
patience = 10
patience_counter = 0

train_losses = []
val_losses = []

for epoch in range(200):
    # Training phase
    model.train()
    epoch_train_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * inputs.size(0)
    train_loss = epoch_train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_val_loss += loss.item() * inputs.size(0)
    val_loss = epoch_val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluation on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    test_probs = torch.sigmoid(test_outputs)
    test_preds = (test_probs >= 0.5).float()

# Convert to numpy
y_true = y_test_t.numpy()
y_pred_np = test_preds.numpy()
y_prob_np = test_probs.numpy()

# Calculate metrics
print("\nEvaluation Metrics:")
print("Accuracy:", accuracy_score(y_true, y_pred_np))
print("Precision:", precision_score(y_true, y_pred_np))
print("Recall:", recall_score(y_true, y_pred_np))
print("F1 Score:", f1_score(y_true, y_pred_np))
print("ROC AUC Score:", roc_auc_score(y_true, y_prob_np))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_np)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Rested', 'Well Rested'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob_np)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', 
         label=f'ROC Curve (AUC = {roc_auc_score(y_true, y_prob_np):.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Probability Distribution
plt.figure(figsize=(6, 4))
sns.histplot(y_prob_np, bins=30, kde=True, color='purple')
plt.title("Distribution of Predicted Probabilities")
plt.xlabel("Probability of Being Well Rested")
plt.ylabel("Count")
plt.show()

# Loss curve visualization
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True)
plt.show()
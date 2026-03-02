
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# -------------------------
# 1. Generate Synthetic Dataset
# -------------------------

np.random.seed(42)

N = 1000

hundred = np.random.uniform(10.5, 12.5, N)
four_hundred = np.random.uniform(47, 55, N)
height = np.random.uniform(66, 76, N)
weight = np.random.uniform(140, 210, N)
squat = np.random.uniform(225, 500, N)
age = np.random.uniform(16, 25, N)

rel_strength = squat / weight

score = (
    -1.5 * hundred
    - 0.8 * four_hundred
    + 1.2 * rel_strength
)

prob = 1 / (1 + np.exp(-0.01 * score))

labels = (prob > 0.5).astype(int)

X = np.column_stack([
    hundred,
    four_hundred,
    height,
    weight,
    squat,
    rel_strength,
    age
])

y = labels

# -------------------------
# 2. Normalize Features
# -------------------------

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# -------------------------
# 3. Define Neural Network
# -------------------------

class SprintNet(nn.Module):
    def __init__(self):
        super(SprintNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = SprintNet()

# -------------------------
# 4. Training Setup
# -------------------------

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50

# -------------------------
# 5. Training Loop
# -------------------------

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# -------------------------
# 6. Evaluation
# -------------------------

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predictions = (test_outputs > 0.5).float()
    accuracy = accuracy_score(y_test.numpy(), predictions.numpy())

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

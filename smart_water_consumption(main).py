import pandas as pd
import numpy as np
from datetime import datetime, timedelta

n_meters = 100
hours = 24 * 30
timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours)][::-1]

data = {
    'timestamp': np.tile(timestamps, n_meters),
    'meter_id': np.repeat(range(1, n_meters + 1), hours),
    'flow_rate': np.random.uniform(50, 500, hours * n_meters),
    'location': np.repeat(['Zone A', 'Zone B', 'Zone C', 'Zone D'], hours * n_meters // 4)
}

df = pd.DataFrame(data)

df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

df



"""## Data Preprocessing"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

le = LabelEncoder()
df['location'] = le.fit_transform(df['location'])

X = df[['meter_id', 'location', 'hour', 'day_of_week', 'month']]
y = df['flow_rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



"""# Implementation of Machine Learning Algorithms

## 1 - Random Forest Regressor
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Square Error (RMSE): {rmse}")

"""## 2- Gradient Boosting Regressor

"""

from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X_train_scaled, y_train)


y_pred_gbr = gbr.predict(X_test_scaled)

mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mean_squared_error(y_test, y_pred_gbr))

print(f"Gradient Boosting MAE: {mae_gbr}")
print(f"Gradient Boosting RMSE: {rmse_gbr}")

"""## 3- Neural Networks"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

class WaterConsumptionNN(nn.Module):
    def __init__(self):
        super(WaterConsumptionNN, self).__init__()
        self.fc1 = nn.Linear(X_train_tensor.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = WaterConsumptionNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
batch_size = 32

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size()[0])

    for i in range(0, X_train_tensor.size()[0], batch_size):
        optimizer.zero_grad()

        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy()

y_test_np = y_test_tensor.numpy()

mae_nn = mean_absolute_error(y_test_np, y_pred)
rmse_nn = np.sqrt(mean_squared_error(y_test_np, y_pred))

print(f"PyTorch Neural Network MAE: {mae_nn}")
print(f"PyTorch Neural Network RMSE: {rmse_nn}")

import matplotlib.pyplot as plt
import seaborn as sns
results = {
    'Model': ['Random Forest', 'Gradient Boosting', 'Neural Network'],
    'MAE': [mae, mae_gbr, mae_nn],
    'RMSE': [rmse, rmse_gbr, rmse_nn]
}
results_df = pd.DataFrame(results)
print(results_df)


plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MAE', data=results_df)
plt.title('Mean Absolute Error (MAE) Comparison')
plt.ylabel('MAE')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='RMSE', data=results_df)
plt.title('Root Mean Squared Error (RMSE) Comparison')
plt.ylabel('RMSE')
plt.show()


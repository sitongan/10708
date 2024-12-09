import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('cleaned.csv', index_col=0)

# For developing purpose
data = data.sample(n=2000, random_state=42)

print(data.head())  

X = data.drop(columns=['avgprice'])  
y = data['avgprice'] 

X = data.drop(columns=['avgprice']) 
y = data['avgprice'] 

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
X_raw = pd.concat([X_train_raw, X_test_raw])


X = pd.get_dummies(X_raw, columns=['origin', 'finaldest'], drop_first=True)

X_train = X.iloc[:len(X_train_raw)].reset_index(drop=True)
X_test = X.iloc[len(X_train_raw):].reset_index(drop=True)

print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

import torch
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, DynamicEdgeConv
from torch_geometric.nn.pool import global_mean_pool
import numpy as np

class DGCNN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, global_feature_dim, k=5):
        super(DGCNN, self).__init__()
        # DynamicEdgeConv dynamically computes edges based on feature similarity
        self.conv1 = DynamicEdgeConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(2 * node_feature_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
            ),
            k=k,
        )
        self.conv2 = DynamicEdgeConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(2 * 128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
            ),
            k=k,
        )
        self.conv3 = DynamicEdgeConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(2 * 32, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 8),
            ),
            k=k,
        )
        # Update fc1 to match the concatenated input size
        self.fc1 = torch.nn.Linear(8 + 8 + edge_feature_dim + global_feature_dim, 16)
        self.fc2 = torch.nn.Linear(16, 1)

    def forward(self, x, edge_index, edge_attr, year, quarter):
        # Apply DGCNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Concatenate dynamic edge features, global features, and node features
        x = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr, year, quarter], dim=1)

        # Fully connected layers for regression
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze()

print('cuda', torch.cuda.is_available())


airport_map = {airport: idx for idx, airport in enumerate(pd.concat([X_raw['origin'], X_raw['finaldest']]).unique())}
X_raw['origin_idx'] = X_raw['origin'].map(airport_map)
X_raw['finaldest_idx'] = X_raw['finaldest'].map(airport_map)

X_train = X_raw.iloc[:len(X_train_raw)].reset_index(drop=True)
X_test = X_raw.iloc[len(X_train_raw):].reset_index(drop=True)

edge_index_train = torch.tensor(np.array([X_train['origin_idx'].values, X_train['finaldest_idx'].values]), dtype=torch.long)
edge_index_test = torch.tensor(np.array([X_test['origin_idx'].values, X_test['finaldest_idx'].values]), dtype=torch.long)

# for edge attribute, it currently only stores the stops.
edge_attr_train = torch.tensor(X_train['stops'].values.reshape(-1, 1), dtype=torch.float)
edge_attr_test = torch.tensor(X_test['stops'].values.reshape(-1, 1), dtype=torch.float)

y_train = torch.tensor(y_train.values, dtype=torch.float)
y_test = torch.tensor(y_test.values, dtype=torch.float)

year_train = torch.tensor(X_train['year'].values, dtype=torch.float).reshape(-1, 1)
quarter_train = torch.tensor(X_train['quarter'].values, dtype=torch.float).reshape(-1, 1)
year_test = torch.tensor(X_test['year'].values, dtype=torch.float).reshape(-1, 1)
quarter_test = torch.tensor(X_test['quarter'].values, dtype=torch.float).reshape(-1, 1)

num_nodes = len(airport_map)
node_features = torch.eye(num_nodes)

#Node feature: onehot code + cooardinates

coordinates = pd.DataFrame({
    'airport': list(airport_map.keys()),
    'longitude': [X_raw[X_raw['origin'] == airport]['OriginLongitude'].values[0] if airport in X_raw['origin'].values else X_raw[X_raw['finaldest'] == airport]['DestLongitude'].values[0] for airport in airport_map],
    'latitude': [X_raw[X_raw['origin'] == airport]['OriginLatitude'].values[0] if airport in X_raw['origin'].values else X_raw[X_raw['finaldest'] == airport]['DestLatitude'].values[0] for airport in airport_map]
})

one_hot_nodes = torch.eye(len(airport_map))  
coordinates_tensor = torch.tensor(coordinates[['longitude', 'latitude']].values, dtype=torch.float)
node_features = torch.cat([one_hot_nodes, coordinates_tensor], dim=1)

node_feature_dim = node_features.shape[1] 
edge_feature_dim = edge_attr_train.shape[1]
global_feature_dim = 2  # Year and quarter as two additional features
model = DGCNN(node_feature_dim, edge_feature_dim, global_feature_dim,k=5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to()

from torch.optim import Adam

optimizer = Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss(reduction='mean')

def train():
    model.train()
    optimizer.zero_grad()
    output = model(node_features, edge_index_train, edge_attr_train, year_train, quarter_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        output = model(node_features, edge_index_test, edge_attr_test, year_test, quarter_test)
        loss = criterion(output, y_test)
    return loss.item()

num_epochs = 100
for epoch in range(num_epochs):
    train_loss = train()
    if epoch % 10 == 0:
        test_loss = test()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")



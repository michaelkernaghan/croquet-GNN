import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class CroquetGNN(torch.nn.Module):
    def __init__(self):
        super(CroquetGNN, self).__init__()
        self.conv1 = GCNConv(3, 16)  # 3 input features per node to 16 features
        self.conv2 = GCNConv(16, 1)  # Reduce to 1 output feature per node

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)  # Use sigmoid to predict probabilities

# Initialize the model
model = CroquetGNN()

# Node features and edges
node_features = torch.tensor([
    [1, 1, 0],  # Ball
    [5, 5, 1],  # Hoop 1
    [8, 3, 1]   # Hoop 2
], dtype=torch.float)
edge_index = torch.tensor([
    [0, 0],  # Source nodes (ball)
    [1, 2]   # Target nodes (hoops)
], dtype=torch.long)
data = Data(x=node_features, edge_index=edge_index)

# Predictions
model.eval()
with torch.no_grad():
    output = model(data.x, data.edge_index)

# Print the predicted probability of success for each edge
print("Predicted probabilities of success for each shot:")
for i, prob in enumerate(output):
    print(f"Shot to Hoop {i+1}: {prob.item():.2f}")

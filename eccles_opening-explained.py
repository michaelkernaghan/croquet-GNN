import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Node features: [x_position, y_position, identifier, action_state]
node_features = torch.tensor([
    [0, 0, 1, 0],  # Blue ball at the baulk (id=1, has_shot=0)
    [10, 0, 0, 0]  # Hoop 1 (id=0, has_been_passed=0)
], dtype=torch.float)

# Edge between Blue ball and Hoop 1
edge_index = torch.tensor([
    [0, 1]  # From Blue ball to Hoop 1
], dtype=torch.long).t().contiguous()

# Graph data object
data = Data(x=node_features, edge_index=edge_index)

class CroquetGNN(torch.nn.Module):
    def __init__(self):
        super(CroquetGNN, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, 2)  # Output could represent success and final position

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Instantiate and run the model
model = CroquetGNN()
out = model(data.x, data.edge_index)
print(out)

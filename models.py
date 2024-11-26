import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

class NodeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NodeClassifier, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, edge_dim=1, add_self_loops=True)
        
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # GATv2Conv supports edge attributes
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.sigmoid(x)


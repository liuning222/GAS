import torch
# import torch.nn as nn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp, GCNConv, GATConv
import torch.nn.functional as F
# from torch_scatter import scatter
from layers import GAS


def readout(x, batch):
    x_mean = gap(x, batch)
    x_max= gmp(x, batch)
    x_sum = gsp(x, batch)
    return torch.cat((x_sum, x_mean, x_max), dim=-1)

class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.ratio = args.topk_ratio
        self.dropout_ratio = args.dropout_ratio
        self.heads = args.attention_heads
        self.mode = args.mask_mode
        self.bn = args.bn

        self.layer1 = GAS(self.num_features, self.nhid, ratio=self.ratio[0], heads=self.heads, mode=self.mode)

        self.layers = torch.nn.ModuleList()
        for i in range(args.num_layers - 1):
            self.layers.append(GAS(self.nhid, self.nhid, ratio=self.ratio[i+1], heads=self.heads, mode=self.mode))
        self.lin1 = torch.nn.Linear(3 * self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x, edge_index, batch = self.layer1(x, edge_index, batch, self.bn)

        xs = readout(x, batch)
        for layer in self.layers:
            x, edge_index, batch = layer(x, edge_index, batch, self.bn)
            x = F.relu(x)
            xs += readout(x, batch)
        x = F.relu(self.lin1(xs))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        out = F.log_softmax(x, dim=-1)

        return out, 0


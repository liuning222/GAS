import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
import torch.nn.functional as F
from layers import GAS
from ogb.graphproppred.mol_encoder import AtomEncoder


def readout(x, batch):
    x_mean = gap(x, batch)
    x_max = gmp(x, batch)
    x_sum = gsp(x, batch)
    return torch.cat((x_sum, x_mean, x_max), dim=-1)


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_tasks = args.num_classes
        self.ratio = args.topk_ratio
        self.dropout_ratio = args.dropout_ratio
        self.heads = args.attention_heads
        self.mode = args.mask_mode
        self.bn = args.bn

        self.atom_encoder = AtomEncoder(self.nhid)

        self.layers = torch.nn.ModuleList()
        for i in range(args.num_layers):
            self.pools.append(GAS(self.nhid, self.nhid, ratio=self.ratio[i], heads=self.heads, mode=self.mode))
        self.classifier = nn.Sequential(
            nn.Linear(self.heads * 3 * self.nhid, self.nhid),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid // 2, self.num_tasks),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.atom_encoder(x)

        xh = []
        for head in range(self.heads):
            xs = 0
            for layer in self.layers:
                x, edge_index, batch = layer(x, edge_index, batch, self.bn)
                x = F.relu(x)
                xs += readout(x, batch)
            xh.append(xs)
        xh_out = torch.cat(xh, dim=-1)

        x_out = self.classifier(xh_out)

        return x_out

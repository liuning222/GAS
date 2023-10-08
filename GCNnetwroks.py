import torch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import BatchNorm1d
from ogb.graphproppred.mol_encoder import AtomEncoder
import torch.nn as nn


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_tasks = args.num_classes
        self.ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.heads = args.attention_heads
        self.mode = args.mask_mode
        self.bn = args.bn
        self.num_layer = args.num_layer
        self.atom_encoder = AtomEncoder(self.nhid)


        self.gcn1 = GCNConv(self.nhid, self.nhid)
        self.gcn2 = GCNConv(self.nhid, self.nhid)
        self.gcn3 = GCNConv(self.nhid, self.nhid)

        self.classifier = nn.Sequential(
            nn.Linear(self.nhid, self.nhid // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid // 2, self.num_tasks),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.atom_encoder(x)

        x1 = self.gcn1(x, edge_index)
        x1 = F.relu(x1)

        x2 = self.gcn2(x1, edge_index)
        x2 = F.relu(x2)

        x3 = self.gcn3(x2, edge_index)
        x3 = F.relu(x3)

        x_out = gap(x1, batch) + gap(x2, batch) + gap(x3, batch)

        x_out = self.classifier(x_out)

        return x_out, 0


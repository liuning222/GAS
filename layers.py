from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.nn import Parameter
from torch_scatter import scatter, scatter_max
import torch
from util import filter_dn, filter_de, filter_new, topk, degree_loss, glorot, zeros
from torch.nn import BatchNorm1d

import networkx as nx
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt


#delete node
class ELSA_DN(MessagePassing):
    def __init__(self, in_channels, out_channels, ratio=0.5, concat=True, negative_slope=0.2, dropout=0,  bias=True, **kwargs):
        super(PGAC_DN, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.bn_layer = BatchNorm1d(self.out_channels)

        self.weight = Parameter(torch.Tensor(in_channels,
                                             out_channels).cuda())
        self.att = Parameter(torch.Tensor(1, 2 * out_channels).cuda())

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        # zeros(self.bias)

    def forward(self, x, edge_index, batch, bn, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, batch, bn, size=size, x=x)

    def propagate(self, edge_index, batch, bn, size=None, **kwargs):
        mp_type = self.__get_mp_type__(edge_index)

        if mp_type == 'adj_t' and self.flow == 'target_to_source':
            raise ValueError(
                ('Flow direction "target_to_source" is invalid for message '
                 'propagation based on `torch_sparse.SparseTensor`. If you '
                 'really want to make use of a reverse message passing flow, '
                 'pass in the transposed sparse tensor to the message passing '
                 'module, e.g., `adj.t()`.'))

        if mp_type == 'edge_index':
            if size is None:
                size = [None, None]
            elif isinstance(size, int):
                size = [size, size]
            elif torch.is_tensor(size):
                size = size.tolist()
            elif isinstance(size, tuple):
                size = list(size)
        elif mp_type == 'adj_t':
            size = list(edge_index.sparse_sizes())[::-1]

        assert isinstance(size, list)
        assert len(size) == 2

        # We collect all arguments used for message passing in `kwargs`.
        kwargs = self.__collect__(edge_index, size, mp_type, kwargs)

        # Try to run `message_and_aggregate` first and see if it succeeds:
        if mp_type == 'adj_t' and self.__fuse__ is True:
            msg_aggr_kwargs = self.__distribute__(self.__msg_aggr_params__,
                                                  kwargs)
            out = self.message_and_aggregate(**msg_aggr_kwargs)
            if out == NotImplemented:
                self.__fuse__ = False

        # Otherwise, run both functions in separation.
        if mp_type == 'edge_index' or self.__fuse__ is False:
            msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)
            num_nodes = batch.size()[0]
            x_j, alpha, alpha_sum = self.message(**msg_kwargs)
            topk_pern_alpha, topk_pern_index, topk_batch = topk(alpha_sum, self.ratio, batch)
            edge_index_mask, x_j_mask, alpha_mask = filter_dn(
                edge_index, x_j, alpha, topk_pern_index, num_nodes)

            out = x_j_mask * alpha_mask.view(-1, 1)

            source, target = edge_index_mask

            out = scatter(out, target, dim=0,reduce=self.aggr)
            if bn:
                out = self.bn_layer(out)


        return out, edge_index_mask, topk_batch

    def message(self, edge_index_i, edge_index_j, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        #compute out edge attention sum for each node
        alpha_sum = scatter(alpha, edge_index_j, dim=0)

        return x_j, alpha, alpha_sum



    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

#delete edge, aggregate node to the most influenced neighbor
class ELSA_DE(MessagePassing):
    def __init__(self, in_channels, out_channels, ratio=0.5, concat=True, negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(PGAC_DE, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.bn_layer = BatchNorm1d(out_channels)

        self.weight = Parameter(torch.Tensor(in_channels,
                                             out_channels).cuda())
        self.att = Parameter(torch.Tensor(1, 2 * out_channels).cuda())

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        # zeros(self.bias)

    def forward(self, x, edge_index, batch, bn, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, batch, bn, size=size, x=x)

    def propagate(self, edge_index, batch, bn, size=None, **kwargs):
        mp_type = self.__get_mp_type__(edge_index)

        if mp_type == 'adj_t' and self.flow == 'target_to_source':
            raise ValueError(
                ('Flow direction "target_to_source" is invalid for message '
                 'propagation based on `torch_sparse.SparseTensor`. If you '
                 'really want to make use of a reverse message passing flow, '
                 'pass in the transposed sparse tensor to the message passing '
                 'module, e.g., `adj.t()`.'))

        if mp_type == 'edge_index':
            if size is None:
                size = [None, None]
            elif isinstance(size, int):
                size = [size, size]
            elif torch.is_tensor(size):
                size = size.tolist()
            elif isinstance(size, tuple):
                size = list(size)
        elif mp_type == 'adj_t':
            size = list(edge_index.sparse_sizes())[::-1]

        assert isinstance(size, list)
        assert len(size) == 2

        # We collect all arguments used for message passing in `kwargs`.
        kwargs = self.__collect__(edge_index, size, mp_type, kwargs)

        # Try to run `message_and_aggregate` first and see if it succeeds:
        if mp_type == 'adj_t' and self.__fuse__ is True:
            msg_aggr_kwargs = self.__distribute__(self.__msg_aggr_params__,
                                                  kwargs)
            out = self.message_and_aggregate(**msg_aggr_kwargs)
            if out == NotImplemented:
                self.__fuse__ = False

        # Otherwise, run both functions in separation.
        if mp_type == 'edge_index' or self.__fuse__ is False:
            msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)
            num_nodes = batch.size()[0]
            x_j, alpha, alpha_sum, max_index = self.message(**msg_kwargs)

            topk_pern_alpha, topk_pern_index, topk_batch = topk(alpha_sum, self.ratio, batch)
            mask, edge_index_mask, x_j_mask, alpha_mask = filter_de(
                edge_index, x_j, alpha, max_index, topk_pern_index, num_nodes)

            out = x_j_mask * alpha_mask.view(-1, 1)

            out_final, edge_index_final, new_index = filter_new(out, edge_index_mask, edge_index, x_j, alpha, topk_pern_index, max_index, num_nodes)


            if bn:
                out_final = self.bn_layer(out_final)


        return out_final, edge_index_final, topk_batch

    def message(self, edge_index_i, edge_index_j, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        #compute out edge attention sum for each node
        alpha_sum = scatter(alpha, edge_index_j, dim=0)

        _, max_index = scatter_max(alpha, edge_index_j, dim=0)

        return x_j, alpha, alpha_sum, max_index


    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class ELSA(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ratio=0.5, heads=1, mode=None):

        super(ELSA, self).__init__()

        if mode=='delete_node':
            self.attention = ELSA_DN(in_channels, out_channels, ratio)
        elif mode=='delete_edge':
            self.attention = ELSA_DE(in_channels, out_channels, ratio)

    def forward(self, x, edge_index, batch, bn):
        x, edge_index, batch = self.attention(x, edge_index, batch, bn)
        return x, edge_index, batch
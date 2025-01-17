from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.nn import Parameter
from torch_scatter import scatter, scatter_max
import torch
from util import filter_dn, filter_de_ours, filter_de_case2, filter_de_case3, filter_new, topk, glorot, zeros
from torch.nn import BatchNorm1d


# delete node
class AConv_DN(MessagePassing):
    def __init__(self, in_channels, out_channels, ratio=0.5, concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        super(AConv_DN, self).__init__(aggr='add', **kwargs)

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
        mutable_size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index,
                                  mutable_size, kwargs)
        msg_kwargs = self.inspector.collect_param_data(
            'message', coll_dict)
        num_nodes = batch.size()[0]
        x_j, alpha, alpha_sum = self.message(**msg_kwargs)

        topk_pern_alpha, topk_pern_index, topk_batch = topk(alpha_sum, self.ratio, batch)
        
        """Topk nodes multi-substructure, others are directly deleted"""
        edge_index_mask, x_j_mask, alpha_mask = filter_dn(
            edge_index, x_j, alpha, topk_pern_index, num_nodes)

        out = x_j_mask * alpha_mask.view(-1, 1)

        source, target = edge_index_mask

        out = scatter(out, target, dim=0, reduce=self.aggr)
        if bn:
            out = self.bn_layer(out)

        return out, edge_index_mask, topk_batch
        
        """topk node single-substructure，others are directly deleted"""
        edge_index_mask, x_j_mask, alpha_mask = filter_dn_case2(edge_index, x_j, alpha, max_index, topk_pern_index,num_nodes)
        
        out = x_j_mask * alpha_mask.view(-1, 1)
   
        out_final, edge_index_final, new_index = filter_new(out, edge_index_mask, edge_index, x_j, alpha,
                                                             topk_pern_index, max_index, num_nodes)
        
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
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

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


# delete edge, aggregate node to the most influenced neighbor
class AConv_DE(MessagePassing):
    def __init__(self, in_channels, out_channels, ratio=0.5, concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        super(AConv_DE, self).__init__(aggr='add', **kwargs)

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
        mutable_size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index,
                                  mutable_size, kwargs)
        msg_kwargs = self.inspector.collect_param_data(
            'message', coll_dict)
        num_nodes = batch.size()[0]

        x_j, alpha, alpha_sum, max_index = self.message(**msg_kwargs)

        topk_pern_alpha, topk_pern_index, topk_batch = topk(alpha_sum, self.ratio, batch)

        """GAS-ms(ours): topk node multi-substructure，others single-node"""
        mask, edge_index_mask, x_j_mask, alpha_mask = filter_de_ours(
            edge_index, x_j, alpha, max_index, topk_pern_index, num_nodes)

        out = x_j_mask * alpha_mask.view(-1, 1)

        out_final, edge_index_final, new_index = filter_new(out, edge_index_mask,
                                                            topk_pern_index, num_nodes)

        # """case1: GAS-mm"""
        # """high-influential nodes allow contributing to multiple substructures (correct)"""
        # """low-influential nodes all contributing to multiple substructures (fault)"""
        # x_j_mask = x_j
        # alpha_mask = alpha
        # edge_index_mask = edge_index
        #
        # out = x_j_mask * alpha_mask.view(-1, 1)
        #
        # out_final, edge_index_final, new_index = filter_new(out, edge_index_mask, edge_index, x_j, alpha,
        #                                                     topk_pern_index, max_index, num_nodes)

        # """case2: GAS-ss"""
        # """high-influential nodes allow contributing to single substructures (fault)"""
        # """low-influential nodes all contributing to single substructures (correct)"""
        # mask, edge_index_mask, x_j_mask, alpha_mask = filter_de_case2(
        #     edge_index, x_j, alpha, max_index, topk_pern_index, num_nodes)
        #
        # out = x_j_mask * alpha_mask.view(-1, 1)
        #
        # out_final, edge_index_final, new_index = filter_new(out, edge_index_mask, edge_index, x_j, alpha,
        #                                                     topk_pern_index, max_index, num_nodes)

        # """case3: GAS-sm"""
        # """high-influential nodes allow contributing to single substructures (fault)"""
        # """low-influential nodes all contributing to multiple substructures (fault)"""
        # mask, edge_index_mask, x_j_mask, alpha_mask = filter_de_case3(
        #     edge_index, x_j, alpha, max_index, topk_pern_index, num_nodes)
        #
        # out = x_j_mask * alpha_mask.view(-1, 1)
        #
        # out_final, edge_index_final, new_index = filter_new(out, edge_index_mask, edge_index, x_j, alpha,
        #                                                     topk_pern_index, max_index, num_nodes)

        if bn:
            out_final = self.bn_layer(out_final)

        return out_final, edge_index_final, topk_batch

    def message(self, edge_index_i, edge_index_j, x_i, x_j, size_i):
        """x_i: target node feature
        x_j: source node feature
        edge_index_i: target node index
        edge_index_j: source node index"""
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i,
                        num_nodes=size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

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


class GAS(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ratio=0.5, heads=1, mode=None):
        super(GAS, self).__init__()

        if mode == 'delete_node':
            self.attention = AConv_DN(in_channels, out_channels, ratio)
        elif mode == 'delete_edge':
            self.attention = AConv_DE(in_channels, out_channels, ratio)

    def forward(self, x, edge_index, batch, bn):
        x, edge_index, batch = self.attention(x, edge_index, batch, bn)
        return x, edge_index, batch

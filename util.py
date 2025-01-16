import torch
from torch_scatter import scatter_add, scatter_max, scatter_sum
from torch_geometric.data import Data
import math

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class Logger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def log(self, str):
        if self.lock:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')
        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def degree_as_tag(dataset):
    all_features = []
    all_degrees = []
    tagset = set([])

    for i in range(len(dataset)):
        edge_weight = torch.ones((dataset[i].edge_index.size(1),))
        degree = scatter_add(edge_weight, dataset[i].edge_index[0], dim=0)
        degree = degree.detach().numpy().tolist()
        tagset = tagset.union(set(degree))
        all_degrees.append(degree)
    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for i in range(len(dataset)):
        node_features = torch.zeros(len(all_degrees[i]), len(tagset))
        node_features[range(len(all_degrees[i])), [tag2index[tag] for tag in all_degrees[i]]] = 1
        all_features.append(node_features)
    return all_features, len(tagset)


def build_data_list(dataset):
    node_features, num_features = degree_as_tag(dataset)

    data_list = []
    for i in range(len(dataset)):
        old_data = dataset[i]
        new_data = Data(x=node_features[i], edge_index=old_data.edge_index, y=old_data.y)
        data_list.append(new_data)
    return data_list, num_features


def topk(x, ratio, batch, min_score=None, tol=1e-7):
    """🔠input
            x: influential score;
            ratio: high-influential ratio;
            batch
        output
            topk influential score，source node index，topk batch index"""
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0][batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = torch.nonzero(x > scores_min).view(-1)
    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ), -2)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        value, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        value = value.view(-1)

        k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]
        value = value[mask]

        mask_batch=[]
        num = 0
        for i in range(num_nodes.size()[0]):
            if i==0:
                mask_i = torch.arange(k[i], dtype=torch.long, device=x.device)
            else:
                mask_i = torch.arange(k[i], dtype=torch.long, device=x.device) + num
            num += num_nodes[i]
            mask_batch.append(mask_i)

        mask_batch = torch.cat(mask_batch, dim=0)
        topk_batch = batch[mask_batch]

    return value, perm, topk_batch

def degree_loss(alpha, edge_index, perm, num_nodes):
    # process alpha
    mask = perm.new_full((num_nodes,), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    source, _ = edge_index
    source = mask[source]
    mask_perm = (source >= 0)
    mask_others = (source < 0)

    alpha_perm = alpha[mask_perm]
    alpha_others = alpha[mask_others]

    att_perm = torch.sum(alpha_perm)/perm.size(0)
    att_others = torch.sum(alpha_others)/(num_nodes-perm.size(0))

    return att_others-att_perm
def filter_dn(edge_index, x_j, alpha, perm, num_nodes):
    #process x_j, alpha
    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if x_j is not None:
        x_j = x_j[mask]
    if alpha is not None:
        alpha = alpha[mask]

    return torch.stack([row, col], dim=0), x_j, alpha
def filter_de_ours(edge_index, x_j, alpha, max_index, perm, num_nodes):
    """input
            edge_index
            x_j: source node feature
            alpha: attention coefficient
            max_index: discriminative edge source index
            perm: topk influential node index
            num_nodes: number of nodes in a batch
        output
            mask
            edge_index_mask
            x_j_mask
            alpha_mask"""
    # process x_j, alpha
    mask = perm.new_full((num_nodes,), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    source, target = edge_index
    source_mask = mask[source]
    source_mask[max_index] = source[max_index]

    source_mask = source_mask>=0
    source = source[source_mask]
    target = target[source_mask]
    x_j_mask = x_j[source_mask]
    alpha_mask = alpha[source_mask]

    edge_index_mask = torch.stack([source, target])

    return mask, edge_index_mask, x_j_mask, alpha_mask
def filter_de_case2(edge_index, x_j, alpha, max_index, perm, num_nodes):
    # process x_j, alpha
    mask = perm.new_full((num_nodes,), -1)

    source, target = edge_index
    source_mask = mask[source]
    source_mask[max_index] = source[max_index]

    source_mask[-num_nodes:] = 1

    source_mask = source_mask>=0
    source = source[source_mask]
    target = target[source_mask]
    x_j_mask = x_j[source_mask]
    alpha_mask = alpha[source_mask]

    edge_index_mask = torch.stack([source, target])

    return mask, edge_index_mask, x_j_mask, alpha_mask
def filter_de_case3(edge_index, x_j, alpha, max_index, perm, num_nodes):
    # process x_j, alpha
    mask = perm.new_full((num_nodes,), -1)
    i = torch.arange(num_nodes, dtype=torch.long, device=perm.device)

    mask_perm = ~(i.view(-1, 1) == perm).any(dim=1)
    non_perm = i[mask_perm]

    source, target = edge_index
    source_non_perm = source[non_perm]
    all_index = torch.cat((max_index, source_non_perm), 0)
    union_all_indel = torch.unique(all_index)

    source_mask = mask[source]
    source_mask[union_all_indel] = source[union_all_indel]

    source_mask[-num_nodes:] = 1

    source_mask = source_mask>=0
    source = source[source_mask]
    target = target[source_mask]
    x_j_mask = x_j[source_mask]
    alpha_mask = alpha[source_mask]

    edge_index_mask = torch.stack([source, target])

    return mask, edge_index_mask, x_j_mask, alpha_mask

def filter_new(out, edge_index_mask, perm, num_nodes):
    mask = perm.new_full((num_nodes,), -1)
    new_index = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = new_index

    source, target = edge_index_mask
    source, target = mask[source], mask[target]
    mask = (target >= 0)
    target_new = target[mask]

    out = out[mask]

    out_final = scatter_sum(out, target_new, dim=0)

    mask_final = (source >=0) & (target >=0)
    source_final, target_final = source[mask_final], target[mask_final]
    edge_index_final = torch.stack([source_final, target_final])

    return out_final, edge_index_final, new_index
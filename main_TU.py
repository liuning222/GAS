import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.transforms import OneHotDegree
from TUnetworks import Net
import torch.nn.functional as F
import argparse
import os
import numpy as np
import random
from torch.utils.data import random_split
from util import num_graphs

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--dataset', type=str, default='IMDBMULTI',
                    help='MUTAG/IMDBBINARY/IMDBMULTI/COLLABDD/PROTEINS/DD')
parser.add_argument('--epochs', type=int, default=1000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=20,
                    help='patience for earlystopping')
parser.add_argument('--batch_size', type=int, default=512,
                    help='batch size')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--num_layers', type=int, default=2, help='number of AGCN layers')
parser.add_argument('--topk_ratio', nargs='+', type=float, default=[0.75])
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--bn', type=bool, default=False,
                    help='batch normalization')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('-lr_decay_step', dest='lr_decay_step', default=1000, type=int, help='lr decay step')
parser.add_argument('-lr_decay_factor', dest='lr_decay_factor', default=0.5, type=float, help='lr decay factor')
parser.add_argument('--attention_heads', type=int, default=1,
                    help='number of attention heads')
parser.add_argument('--mask_mode', type=str, default='delete_edge',
                    help='delete_node/delete_edge')

args = parser.parse_args()
args.device = 'cpu'
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'

max_degrees = {
    'IMDBBINARY': 135,
    'IMDBMULTI': 88,
    'COLLAB': 491,
}

transform = None
if args.dataset in max_degrees:
    transform = OneHotDegree(max_degrees[args.dataset])
print('transform:', transform)
dataset = TUDataset(os.path.join('TUDataset', args.dataset), transform=transform, name=args.dataset)
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)

num_training = int(len(dataset) * 0.8)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

def train(model):
    min_loss = 1e10
    patience = 0
    best_val_acc = 0

    train_loss = []
    val_loss = []

    for epoch in range(args.epochs):
        total_train_loss = 0
        print('epoch {}'.format(epoch))
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            try:
                out, mse_loss = model(data)
                loss = F.nll_loss(out, data.y)
                loss = loss + mse_loss
                total_train_loss += loss.item() * num_graphs(data)
                loss.backward()
                optimizer.step()
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
        train_avg_loss = total_train_loss / len(train_loader.dataset)
        acc_val, avg_loss_val = compute_test(val_loader)

        # lr_decay
        if epoch % args.lr_decay_step == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr_decay_factor * param_group['lr']

        train_loss.append(train_avg_loss)
        val_loss.append(avg_loss_val)

        print("Validation loss:{}\taccuracy:{}".format(avg_loss_val, acc_val))
        if acc_val > best_val_acc:
            best_val_acc = acc_val

        if avg_loss_val < min_loss:
            torch.save(model.state_dict(), 'latest.pth')
            print("Model saved at epoch{}".format(epoch))
            min_loss = avg_loss_val
            patience = 0
        else:
            patience += 1

        print('best val acc:', best_val_acc)

        if patience > args.patience:
            break

    model = Net(args).to(args.device)
    model.load_state_dict(torch.load('latest.pth'))
    test_acc, test_loss = compute_test(test_loader)
    print("Test accuarcy:{}".format(test_acc))

    return test_acc


def compute_test(loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        with torch.no_grad():
            out, mse_loss = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out, data.y, reduction='sum').item()
        loss += mse_loss
    return correct / len(loader.dataset), loss / len(loader.dataset)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    with open('acc_result.txt', 'a+') as f:
        f.write(str(args) + '\n')

    model = Net(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    all_test_acc = []
    for i in range(10):
        # Model training
        test_acc = train(model)
        all_test_acc.append(test_acc)
        with open('acc_result.txt', 'a+') as f:
            f.write(str(test_acc) + '\n')

    print('10 run average:', np.mean(all_test_acc))
    print('10 run std:', np.std(all_test_acc))
    # print(get_parameter_number(model))
    with open('acc_result.txt', 'a+') as f:
        f.write(str(np.mean(all_test_acc)) + '\n')
        f.write(str(np.std(all_test_acc)) + '\n')

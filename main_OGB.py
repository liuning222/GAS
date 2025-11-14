from OGBnetworks import Net
import argparse
from tqdm import tqdm, trange
import random
import numpy as np
import torch
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0015,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=1024,
                    help='hidden size')
parser.add_argument('--num_layers', type=int, default=1, help='number of GCN and pooling layers')
# parser.add_argument("--grad-norm", type=float, default=1.0)
parser.add_argument('-pooling_ratio', nargs='+', type=float, default=[0.8])
parser.add_argument('--dropout_ratio', type=float, default=0.1,
                    help='dropout ratio')
parser.add_argument('--bn', type=bool, default=False,
                    help='batch normalization')
parser.add_argument('--dataset', type=str, default='ogbg-moltoxcast',
                    help='ogbg-molhiv/ogbg-molpcba/ogbg-moltox21/ogbg-moltoxcast/ogbg-molbbbp/ogbg-molsider/ogbg-molbace/ogbg-molclintox')
parser.add_argument('--epochs', type=int, default=1000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=20,
                    help='patience for earlystopping')
parser.add_argument('--attention_heads', type=int, default=1,
                    help='number of attention heads')
parser.add_argument('--mask_mode', type=str, default='delete_edge',
                    help='delete_node/delete_edge')
parser.add_argument('-lr_decay_step', 	dest='lr_decay_step',default=10000, type=int, help='lr decay step')
parser.add_argument('-lr_decay_factor', dest='lr_decay_factor', default=0.5, type=float, help='lr decay factor')


args = parser.parse_args()

dataset = PygGraphPropPredDataset(name=args.dataset)

args.num_classes = dataset.num_tasks
args.num_features = dataset.num_features

split_idx = dataset.get_idx_split()

### automatic evaluator. takes dataset name as input
evaluator = Evaluator(args.dataset)

train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])

            # loss += mse_loss/num_graphs(batch)
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    with open('acc_result.txt', 'a+') as f:
        f.write(str(args) + '\n')
    ### automatic dataloading and splitting

    all_test_auc = []
    for i in range(10):
        print("run {}".format(i))
        random.seed(args.seed+i)
        np.random.seed(args.seed+i)
        torch.manual_seed(args.seed+i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed+i)
            args.device = 'cuda:0'

        model = Net(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val = 0

        for epoch in range(1, args.epochs + 1):
            print("=====Epoch {}".format(epoch))
            print('Training...')
            train(model, args.device, train_loader, optimizer, dataset.task_type)

            print('Evaluating...')
            train_perf = eval(model, args.device, train_loader, evaluator)
            valid_perf = eval(model, args.device, valid_loader, evaluator)
            test_perf = eval(model, args.device, test_loader, evaluator)

            print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

            if valid_perf[dataset.eval_metric] > best_val:
                torch.save(model.state_dict(), 'latest.pth')
                print("Model saved at epoch{}".format(epoch))
                best_val = valid_perf[dataset.eval_metric]
                patience = 0
            else:
                patience += 1
            if patience > args.patience:
                break

            # lr_decay
            if epoch % args.lr_decay_step == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr_decay_factor * param_group['lr']

        model = Net(args).to(args.device)
        model.load_state_dict(torch.load('latest.pth'))
        test = eval(model, args.device, test_loader, evaluator)
        print("Test accuarcy:{}".format(test[dataset.eval_metric]))

        print('Finished training!')


        with open('acc_result.txt', 'a+') as f:
            f.write(str(test[dataset.eval_metric]) + '\n')

        all_test_auc.append(test[dataset.eval_metric])

    print('10 run average:', np.mean(all_test_auc))
    print('10 run std:', np.std(all_test_auc))

    with open('acc_result.txt', 'a+') as f:
        f.write(str(np.mean(all_test_auc)) + '\n')
        f.write(str(np.std(all_test_auc)) + '\n')


if __name__ == "__main__":
    main()
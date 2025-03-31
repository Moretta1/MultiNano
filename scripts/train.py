from time import time
import argparse
import csv
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler, AdamW
from utils import *
from models import SignalTransformer_v2, SignalTransformer, NaiveNet, BahdanauAttention

torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

RMs = ["m6A", "m1A", "m5C", "hm5C", "I", "m7G", "psi"]
num_task = len(RMs)

# input feature dictionary, please change to your own data path
data_pth = '/data/fast5_data/ELIGOS/TandemMod_features/features/'

train_mod_dict = {
    "hm5C": data_pth + 'hm5C/hm5C.train.feature.tsv',
    "I": data_pth + 'I/I.train.feature.tsv',
    "m1A": data_pth + 'm1A/m1A.train.feature.tsv',
    "m5C": data_pth + 'm5C/m5C.train.feature.tsv',
    "m6A": data_pth + 'm6A/m6A.train.feature.tsv',
    "m7G": data_pth + 'm7G/m7G.train.feature.tsv',
    "psi": data_pth + 'psi/psi.train.feature.tsv',
}

test_mod_dict = {
    "hm5C": data_pth + 'hm5C/hm5C.test.feature.tsv',
    "I": data_pth + 'I/I.test.feature.tsv',
    "m1A": data_pth + 'm1A/m1A.test.feature.tsv',
    "m5C": data_pth + 'm5C/m5C.test.feature.tsv',
    "m6A": data_pth + 'm6A/m6A.test.feature.tsv',
    "m7G": data_pth + 'm7G/m7G.test.feature.tsv',
    "psi": data_pth + 'psi/psi.test.feature.tsv',
}

valid_mod_dict = {
    "hm5C": data_pth + 'hm5C/hm5C.valid.feature.tsv',
    "I": data_pth + 'I/I.valid.feature.tsv',
    "m1A": data_pth + 'm1A/m1A.valid.feature.tsv',
    "m5C": data_pth + 'm5C/m5C.valid.feature.tsv',
    "m6A": data_pth + 'm6A/m6A.valid.feature.tsv',
    "m7G": data_pth + 'm7G/m7G.valid.feature.tsv',
    "psi": data_pth + 'psi/psi.valid.feature.tsv',
}

train_unmod_dict = {
    "hm5C": data_pth + 'C/C.train.feature.tsv',
    "I": data_pth + 'G/G.train.feature.tsv',
    "m1A": data_pth + 'A/A.train.feature.tsv',
    "m5C": data_pth + 'C/C.train.feature.tsv',
    "m6A": data_pth + 'A/A.train.feature.tsv',
    "m7G": data_pth + 'G/G.train.feature.tsv',
    "psi": data_pth + 'U/U.train.feature.tsv',
}

test_unmod_dict = {
    "hm5C": data_pth + 'C/C.test.feature.tsv',
    "I": data_pth + 'G/G.test.feature.tsv',
    "m1A": data_pth + 'A/A.test.feature.tsv',
    "m5C": data_pth + 'C/C.test.feature.tsv',
    "m6A": data_pth + 'A/A.test.feature.tsv',
    "m7G": data_pth + 'G/G.test.feature.tsv',
    "psi": data_pth + 'U/U.test.feature.tsv',
}

valid_unmod_dict = {
    "hm5C": data_pth + 'C/C.valid.feature.tsv',
    "I": data_pth + 'G/G.valid.feature.tsv',
    "m1A": data_pth + 'A/A.valid.feature.tsv',
    "m5C": data_pth + 'C/C.valid.feature.tsv',
    "m6A": data_pth + 'A/A.valid.feature.tsv',
    "m7G": data_pth + 'G/G.valid.feature.tsv',
    "psi": data_pth + 'U/U.valid.feature.tsv',
}


class NN(SignalTransformer_v2):
    def __init__(self):
        """
        Initialize the NN class.
        Inherits from the SignalTransformer_v2 class.
        """
        super(NN, self).__init__()


# uncomment to different structure
'''
class NN(SignalTransformer):
    def __init__(self):
        """
        Initialize the NN class.
        Inherits from the SignalTransformer class.
        """
        super(NN, self).__init__()

class NN(NaiveNet):
    def __init__(self):
        """
        Initialize the NN class.
        Inherits from the NaiveNet class.
        """
        super(NN, self).__init__()
'''


def save_best(model, state, is_best, OutputDir):
    if is_best:
        print('=> Saving a new best from epoch %d"' % state['epoch'])
        torch.save(model, OutputDir + '/epoch%d.pkl' % state['epoch'])
    else:
        print("=> Validation Performance did not improve")


def construct_data(RMs, mode, len_train=3e2, len_test=2e2, len_val=2e2):
    x_seq = []
    label_each_type = []
    sub_type_len = []

    for i in range(len(RMs)):
        if mode == 'train':
            x, y = load_data(data_mod=train_mod_dict[RMs[i]], data_unmod=train_unmod_dict[RMs[i]],
                             data_length=len_train)
        elif mode == 'test':
            x, y = load_data(data_mod=test_mod_dict[RMs[i]], data_unmod=test_unmod_dict[RMs[i]], data_length=len_test)
        else:
            x, y = load_data(data_mod=test_mod_dict[RMs[i]], data_unmod=test_unmod_dict[RMs[i]], data_length=len_val)

        x_seq = x_seq + x
        label_each_type.append(y)
        sub_type_len.append(len(x))  # sample number in each modification
        print("length in modification " + str(RMs[i]) + str(" : ") + str(len(x)))

    row_num = np.sum(sub_type_len)
    col_num = len(RMs)
    labels = pd.DataFrame(np.zeros((row_num, col_num)), dtype=object, columns=RMs)
    y_list = []
    index_start = 0

    for i in range(len(RMs)):
        index_end = index_start + len(label_each_type[i])
        # print(f'index_start to index_end : {index_start}, {index_end}')
        labels.iloc[index_start:index_end, i] = label_each_type[i]
        # iloc for assign values
        y_list.append(label_each_type[i])
        index_start = index_end  # index updating

    print("load data finished")
    return x_seq, y_list, labels


def valid(model, test_loader, loss_weight):
    # !!! Valid
    model.eval()
    test_loss = 0
    metrics_dict = {"acc": 0,
                    "auc": 0,
                    "ap": 0}

    corrects = 0
    with torch.no_grad():
        for x, y_true in test_loader:
            test_num = len(y_true)
            y_true = y_true.cuda()
            signal, kmer, mean, std, intense, dwell, base_quality = x
            signal = Variable(signal.to(device)).to(torch.float32)
            kmer = Variable(kmer.to(device)).to(torch.long)
            mean = Variable(mean.to(device)).to(torch.float32)
            std = Variable(std.to(device)).to(torch.float32)
            intense = Variable(intense.to(device)).to(torch.float32)
            dwell = Variable(dwell.to(device)).to(torch.float32)
            base_quality = Variable(base_quality.to(device)).to(torch.float32)
            batch_size, features = signal.size()
            signal = signal.view(batch_size, 1, features)

            y_pred = model(signal, kmer, mean, std, intense, dwell, base_quality)
            # y_pred = model(signal)
            test_loss += naive_loss(y_pred, y_true, loss_weight)

            acc = 0
            auc = 0
            ap = 0
            correct = 0

            for i in range(num_task):
                label = y_true.cpu().numpy()[:, i]  # ith modification true label
                y_score = y_pred[i].cpu().detach().numpy()
                # y_score contains 2 elements，corresponding to the predict score of [type1,type2,...,type_{num_task}]
                y_pred_single = np.array([0 if instance < 0.5 else 1 for instance in y_score])
                correct += np.sum(y_pred_single == label)

                tp = 0
                fp = 0
                tn = 0
                fn = 0

                for index in range(test_num):
                    if label[index] == 1:
                        if label[index] == y_pred_single[index]:
                            tp = tp + 1
                        else:
                            fn = fn + 1
                    else:
                        if label[index] == y_pred_single[index]:
                            tn = tn + 1
                        else:
                            fp = fp + 1

                # print("true label distribution within modification:")
                # print(Counter(label))
                # print('tp\tfp\ttn\tfn')
                # print('{}\t{}\t{}\t{}'.format(tp, fp, tn, fn))

                acc += float(tp + tn) / test_num

                try:
                    auc += roc_auc_score(label, y_pred_single)  # calculate AUC
                except ValueError:
                    pass
                try:
                    ap += average_precision_score(label, y_pred_single)  # calculate AP
                except ValueError:
                    pass

            # average performance in totally num_task modifications
            metrics_dict['acc'] += acc / num_task
            metrics_dict["auc"] += auc / num_task
            metrics_dict["ap"] += ap / num_task
            corrects += correct  # total correct number in num_task modifications

        num_examples = len(test_loader.dataset)
        test_loss /= num_examples
        num_batches = num_examples // test_loader.batch_size + 1

        metrics_dict['acc'] /= num_batches
        metrics_dict['auc'] /= num_batches
        metrics_dict['ap'] /= num_batches

    print('Valid set: Average valid_loss: {:.4f}, Accuracy: {}/{} which is: {:.3f}%\n'.format(test_loss, corrects,
                                                                                              num_examples,
                                                                                              metrics_dict['acc']))
    return test_loss, metrics_dict


def binary_cross_entropy(x, y, focal=False):
    alpha = 0.75
    gamma = 2

    pt = x * y + (1 - x) * (1 - y)
    at = alpha * y + (1 - alpha) * (1 - y)

    # focal loss
    if focal:
        loss = -at * (1 - pt) ** (gamma) * (torch.log(x) * y + torch.log(1 - x) * (1 - y))
    else:
        loss = -(torch.log(x) * y + torch.log(1 - x) * (1 - y))
    return loss


def naive_loss(y_pred, y_true, loss_weight=None, ohem=True, focal=True):
    # num_task: modification types
    num_task = y_true.shape[-1]
    num_examples = y_true.shape[0]  # y_true: array [sample_num num_task]
    k = 0.7

    loss_output = torch.zeros(num_examples).cuda()
    for i in range(num_task):
        if loss_weight:
            out = loss_weight[i] * binary_cross_entropy(y_pred[i], y_true[:, i], focal)
            loss_output += out
        else:
            loss_output += binary_cross_entropy(y_pred[i], y_true[:, i], focal)

    # Online Hard Example Mining
    if ohem:
        val, idx = torch.topk(loss_output, int(k * num_examples))
        loss_output[loss_output < val[-1]] = 0

    loss = torch.sum(loss_output)

    return loss


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, train_loader, test_loader, args, lossWeight):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    gamma = float(args.lr) * 0.25
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    save_dir = args.output
    loss_weight_ = lossWeight

    if not os.path.exists(save_dir):
        print('%s does not exist, create it now' % save_dir + '-' * 30)
        os.mkdir(save_dir)

    logfile = open(save_dir + '/log.csv', 'w')

    logwriter = csv.DictWriter(logfile,
                               fieldnames=['epoch', 'loss', 'val_loss', 'val_acc', 'val_auc', 'val_precision'])
    logwriter.writeheader()

    t0 = time()

    best_val_acc = 0
    best_val_loss = 50
    patience = 0
    patience_limit = 20

    loss_weight = []
    for i in range(num_task):
        print("calculating weighted loss %d" % i)
        print(loss_weight_[i])
        loss_weight.append(torch.tensor(loss_weight_[i], requires_grad=True, device=device))

    print('Begin Training' + '-' * 70)
    for epoch in range(args.epoch):
        torch.cuda.empty_cache()
        training_loss = 0.0
        ti = time()
        model.train()

        for i, (x, y) in enumerate(train_loader):
            # load data to cuda
            y = y.cuda()
            signal, kmer, mean, std, intense, dwell, base_quality = x
            signal = Variable(signal.to(device)).to(torch.float32)
            kmer = Variable(kmer.to(device)).to(torch.long)
            mean = Variable(mean.to(device)).to(torch.float32)
            std = Variable(std.to(device)).to(torch.float32)
            intense = Variable(intense.to(device)).to(torch.float32)
            dwell = Variable(dwell.to(device)).to(torch.float32)
            base_quality = Variable(base_quality.to(device)).to(torch.float32)
            batch_size, features = signal.size()
            signal = signal.view(batch_size, 1, features)

            y_pred = model(signal, kmer, mean, std, intense, dwell,
                           base_quality)  # for NaiveNet or SignalTransformer-v2
            # y_pred = model(signal) # for SignalTransformer

            # get loss
            loss = naive_loss(y_pred, y, loss_weight)  # here y is a list instead of array
            optimizer.zero_grad()
            # gradient clipping
            clip_value = 1
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            loss.backward()
            optimizer.step()

            training_loss += loss.data.detach()  # training_loss calculating

            if i == 1:  # periodic check y_pred
                print("Sanity Checking, at epoch%02d, iter%02d, y_pred is" % (epoch, i),
                      [y_pred[j][1].cpu().detach() for j in range(num_task)])
                print("Learning rate: %.16f" % optimizer.state_dict()['param_groups'][0]['lr'])

        lr_decay.step()

        # adjust learning rate by hand
        if 19 <= epoch < 39:
            adjust_learning_rate(optimizer, optimizer.state_dict()['param_groups'][0]['lr'] / 10)
        elif 39 <= epoch < 59:
            adjust_learning_rate(optimizer, optimizer.state_dict()['param_groups'][0]['lr'] / (10 ** 2))
        elif 59 <= epoch < 79:
            adjust_learning_rate(optimizer, optimizer.state_dict()['param_groups'][0]['lr'] / (10 ** 3))
        elif epoch >= 79:
            adjust_learning_rate(optimizer, optimizer.state_dict()['param_groups'][0]['lr'] / (10 ** 4))

        # ==> valid/test
        val_loss, metrics_dict = valid(model, test_loader, loss_weight)

        logwriter.writerow(dict(epoch=epoch, loss=training_loss.cpu().numpy() / len(train_loader.dataset),
                                val_loss=val_loss.cpu().numpy(), val_acc=metrics_dict['acc'],
                                val_auc=metrics_dict["auc"],
                                val_precision=metrics_dict['ap']))

        print("===>Epoch %02d: loss=%.5f, val_loss=%.4f, val_acc=%.4f,val_auc=%.4f, val_ap=%.4f, time=%ds"
              % (epoch, training_loss / len(train_loader.dataset), val_loss,
                 metrics_dict["acc"], metrics_dict["auc"], metrics_dict["ap"],
                 time() - ti))

        is_best = bool(metrics_dict["acc"] > best_val_acc and metrics_dict["acc"] > 0.6)
        # this is an additional condiction to avoid the case that prediction result belongs to one certain type

        if is_best:  # update best validation acc and save model
            best_val_acc = metrics_dict["acc"]
            best_val_loss = val_loss
            save_best(model, {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_accuracy': best_val_acc,
                'optimizer': optimizer.state_dict()
            }, is_best, save_dir)
            patience = 0

        else:
            patience += 1
            if patience >= patience_limit:
                print("patience_limit achieved!")
                print("=> Validation Performance did not improve")
                break

    logfile.close()
    print("Total time = %ds" % (time() - t0))
    print('End Training' + '-' * 70)


def cal_loss_weight(dataset, beta=0.9):
    data, label = dataset[:]
    total_example = label.shape[0]
    num_task = label.shape[1]
    label = np.asarray(label)

    labels_dict = dict(zip(range(num_task), [sum(label[:, i]) for i in range(num_task)]))
    keys = labels_dict.keys()
    class_weight = dict()

    # Class-Balanced Loss Based on Effective Number of Samples
    for key in keys:
        effective_num = 1.0 - beta ** labels_dict[key]
        weights = (1.0 - beta) / effective_num
        class_weight[key] = weights

    weights_sum = sum(class_weight.values())

    # normalizing weights
    for key in keys:
        class_weight[key] = class_weight[key] / weights_sum * num_task

    return class_weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TandemMod train_mode, multiple types of RNA modification detection.')

    parser.add_argument('--output', type=str, required=True, help='New model file to be saved.')
    parser.add_argument('--epoch', type=int, required=False, default=10, help='Training epoch')
    parser.add_argument('--bs', type=int, required=False, default=64, help='batch_size')
    parser.add_argument('--lr', type=float, required=False, default=0.001, help='learning_rate')
    parser.add_argument('--decay', type=float, required=False, default=0.0025, help='learning_decay')

    args = parser.parse_args()

    print("train process.")

    x_train, y_train, y_train_df = construct_data(RMs, mode="train", len_train=6e3)
    x_valid, y_valid, y_valid_df = construct_data(RMs, mode="valid", len_val=12e2)
    # please modify data_length, this is just a demo

    train_dataset = RMdata(x_train, np.asarray(y_train_df, dtype=int))
    test_dataset = RMdata(x_valid, np.asarray(y_valid_df, dtype=int))
    print("MyDataset loaded.")

    # ------------------->>>
    train_num = len(x_train)
    valid_num = len(x_valid)
    print('Train data：', train_num)
    print('Valid data：', valid_num)
    # ---------------------------------->>
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=True)
    print("DataLoader loaded.")

    # model = NaiveNet(num_task=len(RMs), num_classes=2, vocab_size=5, embedding_size=4, seq_len=5).to(device) # NaiveNet
    # model = SignalTransformer(num_task=len(RMs)).to(device)   # SignalTransformer
    model = SignalTransformer_v2(num_task=len(RMs), vocab_size=5, embedding_size=4, seq_len=5).to(device)  # MultiNano
    print(model)

    loss_weight_ = cal_loss_weight(train_loader.dataset)

    # => training!
    train(model, train_loader, test_loader, args, lossWeight=loss_weight_)

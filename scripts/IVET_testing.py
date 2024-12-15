import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
from torch.utils.data import DataLoader

from utils import *
from torch.autograd import Variable
from models import TandemMod
from calculate_metrics import *

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device=", device)
RMs = ["m6A", "m1A", "m5C"]
num_task = len(RMs)

data_root_pth = '/data/home/grp-lizy/wangrulan/tandem/data/IVET/'

test_mod_dict = {
    "m1A": data_root_pth + 'm1A/m1A.test.feature.tsv',
    "m5C": data_root_pth + 'm5C/m5C.test.feature.tsv',
    "m6A": data_root_pth + 'm6A/m6A.test.feature.tsv',
}

test_unmod_dict = {
    "m1A": data_root_pth + 'unmod/A/A.test.feature.tsv',
    "m5C": data_root_pth + 'unmod/C/C.test.feature.tsv',
    "m6A": data_root_pth + 'unmod/A/A.test.feature.tsv',
}

train_mod_dict = {
    "m1A": data_root_pth + 'm1A/m1A.train.feature.tsv',
    "m5C": data_root_pth + 'm5C/m5C.train.feature.tsv',
    "m6A": data_root_pth + 'm6A/m6A.train.feature.tsv',
}

train_unmod_dict = {
    "m1A": data_root_pth + 'unmod/A/A.train.feature.tsv',
    "m5C": data_root_pth + 'unmod/C/C.train.feature.tsv',
    "m6A": data_root_pth + 'unmod/A/A.train.feature.tsv',
}


class NN(TandemMod):
    def __init__(self):
        """
        Initialize the NN class.
        Inherits from the TandemMod class.
        """
        super(NN, self).__init__()


def construct_data(RMs, mode='test'):
    x_seq = []
    label_each_type = []
    sub_type_len = []

    for i in range(len(RMs)):
        x, y = load_data(data_mod=test_mod_dict[RMs[i]], data_unmod=test_unmod_dict[RMs[i]])

        x_seq = x_seq + x
        label_each_type.append(y)
        sub_type_len.append(len(x))  # sample number in each modification

    row_num = np.sum(sub_type_len)
    col_num = len(RMs)
    labels = pd.DataFrame(np.zeros((row_num, col_num)), dtype=object)
    labels.columns = RMs
    y_list = []

    for i in range(len(RMs)):
        target = labels[RMs[i]]
        if i == 0:
            index_start = 0
        index_end = index_start + len(label_each_type[i])
        target[index_start:index_end] = label_each_type[i]
        labels[RMs[i]] = target
        y_list.append(target.tolist())
        index_start = index_end

    print("load data finished")
    return x_seq, y_list, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='TandemMod independent testing mode, multiple types of RNA modification detection.')
    parser.add_argument('--output', required=True, help='file save path, make sure end with a slash /.')
    parser.add_argument('--pretrained', required=True, help='pretrained model file')
    parser.add_argument('--bs', type=int, required=False, default=64, help='batch_size')
    args = parser.parse_args()
    if not os.path.exists(args.output):
        print('%s does not exist, create it now' % args.output + '-' * 30)
        os.makedirs(args.output)
        os.makedirs(args.output + str('Figs/'))

    x_test, y_test, y_df = construct_data(RMs, mode="test")
    test_dataset = RMdata(x_test, np.array(y_df, dtype=int))

    # ------------------->>>
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=True)
    print("DataLoader loaded.")

    test_num = len(x_test)
    num_batches = test_num // test_loader.batch_size + 1
    print('Test数据：', test_num)

    model = torch.load(args.pretrained)
    print('---------load_model: checkpoint.pkl----->>>>')

    # ---------------------------------->>
    model.eval()

    y_label_df = pd.DataFrame(columns=RMs)
    y_pred_score_df = pd.DataFrame(columns=RMs)
    metrics_dict = {"acc": 0,
                    "auc": 0,
                    "ap": 0}

    for step, (test_x, test_y) in enumerate(test_loader):
        # test_y: true_labels
        signal, kmer, mean, std, intense, dwell, base_quality = test_x
        signal = Variable(signal.to(device)).to(torch.float32)
        kmer = Variable(kmer.to(device)).to(torch.long)
        mean = Variable(mean.to(device)).to(torch.float32)
        std = Variable(std.to(device)).to(torch.float32)
        intense = Variable(intense.to(device)).to(torch.float32)
        dwell = Variable(dwell.to(device)).to(torch.float32)
        base_quality = Variable(base_quality.to(device)).to(torch.float32)
        batch_size, features = signal.size()
        signal = signal.view(batch_size, 1, features)

        out = model(signal, kmer, mean, std, intense, dwell, base_quality)
        test_y = test_y.cuda()
        df_tmp = pd.DataFrame(np.zeros((len(out[0]), num_task)), columns=RMs, dtype=object)
        df_score_tmp = pd.DataFrame(np.zeros((len(out[0]), num_task)), columns=RMs, dtype=object)

        acc = 0
        auc = 0
        ap = 0
        correct = 0

        for i in range(num_task):
            label = test_y.cpu().numpy()[:, i]  # true label
            y_score = out[i].cpu().detach().numpy()  # predicting score
            df_score_tmp[RMs[i]] = y_score  # score
            df_tmp[RMs[i]] = label  # true_label

            y_pred_single = np.array([0 if instance < 0.5 else 1 for instance in y_score])  # for assessment
            performance, roc_data, prc_data = calculate_metric_within_modification(pred_prob=y_score,
                                                                                   label_pred=y_pred_single,
                                                                                   label_real=label)
            # performance: [ACC, Sensitivity, Specificity, AUC, MCC]
            # roc_data: [FPR, TPR, AUC]
            # prc_data: [recall, precision, AP]
            # print for debug
            acc += performance[0]
            auc += roc_data[2]
            ap += prc_data[2]

            print("*" * 30 + "[ACC, Sensitivity, Specificity, AUC, MCC] in " + str(i) + "th modification, batch " + str(
                step) + " :" + "*" * 30)
            print(performance)

        print("batch " + str(step) + " finished.")
        y_label_df = pd.concat([y_label_df, df_tmp], ignore_index=True)
        y_pred_score_df = pd.concat([y_pred_score_df, df_score_tmp], ignore_index=True)
        # update dictionary: average performance in totally num_task modification

    # save
    y_pred_score = y_pred_score_df.to_numpy(dtype=float)
    y_pred_score_df.to_csv('%s/predicted_score.csv' % args.output, index=False)

    y_true = y_label_df.to_numpy(dtype=int)
    y_label_df.to_csv('%s/true_label.csv' % args.output, index=False)

    print("# ===========cal_metrics from saved dataframe ===========")
    y_pred = []
    for i in range(num_task):
        target = y_pred_score[:, i]
        y_pred.append(target)

    metrics_final, metrics_avg = cal_metrics(y_pred, y_true, plot=True, class_names=RMs, plot_name=None,
                                             save_dir=args.output)

    # save metrics_final as predict result:
    perform_df = pd.DataFrame(np.zeros((len(metrics_final), num_task)), dtype=object, columns=RMs,
                              index=metrics_final.keys())
    for criteria in metrics_final.keys():
        perform_df.loc[criteria] = metrics_final[criteria]

    perform_df.to_csv('%s/testing_result.csv' % args.output, index=True)

    # save average metrics_dict as average predict result:
    perform_df = pd.DataFrame(np.zeros((len(metrics_avg), num_task)), dtype=object, columns=RMs,
                              index=metrics_avg.keys())
    for criteria in metrics_avg.keys():
        perform_df.loc[criteria] = metrics_avg[criteria]
    # save to csv
    perform_df.to_csv('%s/average_metrics_dict.csv' % args.output, index=True)


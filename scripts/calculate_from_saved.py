import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import pandas as pd
from utils import *
from calculate_metrics import *

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device=", device)
# RMs = ["m6A", "m1A", "m5C", "hm5C", "I", "m7G", "psi"]
# RMs = ["hm5C", "I", "m1A", "m5C", "m6A", "m7G", "psi"]
RMs = ["m6A", "m1A", "m5C"]
num_task = len(RMs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='assessment from saved result.')
    parser.add_argument('--pth', required=True, help='file save path, make sure end with a slash /.')
    args = parser.parse_args()
    if not os.path.exists(args.pth):
        print('%s does not exist' % args.pth + '-' * 30)

    #import pdb;pdb.set_trace();
    y_pred_score = pd.read_csv('%s/predicted_score.csv' % args.pth)

    y_true = pd.read_csv('%s/true_label.csv' % args.pth)

    print("# ===========cal_metrics from saved dataframe ===========")
    y_pred = []
    y_pred_score = y_pred_score.to_numpy()
    for i in range(num_task):
        target = y_pred_score[:, i]
        y_pred.append(target)

    metrics_final, metrics_avg = cal_metrics(y_pred, y_true.to_numpy(), plot=True, class_names=RMs, plot_name=None,
                                             save_dir=args.pth)

    # save metrics_final as predict result:
    perform_df = pd.DataFrame(np.zeros((len(metrics_final), num_task)), dtype=object, columns=RMs,
                              index=metrics_final.keys())
    for criteria in metrics_final.keys():
        perform_df.loc[criteria] = metrics_final[criteria]

    perform_df.to_csv('%s/testing_result.csv' % args.pth, index=True)

    # save average metrics_dict as average predict result:
    perform_df = pd.DataFrame(np.zeros((len(metrics_avg), num_task)), dtype=object, columns=RMs,
                              index=metrics_avg.keys())
    for criteria in metrics_avg.keys():
        perform_df.loc[criteria] = metrics_avg[criteria]
    # save to csv
    perform_df.to_csv('%s/average_metrics_dict.csv' % args.pth, index=True)


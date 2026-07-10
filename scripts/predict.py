import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
from torch.utils import data as torch_data
from torch.utils.data import Dataset
from torch.autograd import Variable
from utils import *
import pandas as pd

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device=", device)

# predict dataset pth
# data_root_pth = '/data/home/grp-lizy/wangrulan/tandem/data/m6A_HEK293T'
TASK_MAP = {"m6A": 0, "m1A": 1, "m5C": 2, "hm5C": 3, "I": 4, "m7G": 5, "psi": 6}


class MyDataset(Dataset):
    """
    Dataset class that holds x and y data.

    Args:
        x (Any): The input data.
        y (Any): The target data.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def load_predict_data(file):
    """
    Load prediction data from the specified file.

    Notice here without true labels
    """

    X, Y = [], []
    with open(file) as f:
        for line in f:
            line = line.rstrip()
            items = line.split("\t")

            read_id = line.split("\t")[0]
            contig = line.split("\t")[1]
            position = line.split("\t")[2]
            motif = line.split("\t")[3]

            signals = "|".join(items[9:14]).split("|")
            signal = np.array([float(signal) for signal in signals])
            kmer = items[3]
            kmer = np.array([kmer_encode_dic[base] for base in kmer])
            mean = np.array([float(item) for item in items[4].split("|")])
            std = np.array([float(item) for item in items[5].split("|")])
            intense = np.array([float(item) for item in items[6].split("|")])
            dwell = np.array([float(item) for item in items[7].split("|")]) / 200
            base_quality = np.array([float(item) for item in items[8].split("|")]) / 40
            x = [signal, kmer, mean, std, intense, dwell, base_quality]
            X.append(x)
            Y.append("|".join([contig, position, motif, read_id]))

    return X, Y


def predict_certain_type(model, dataloader, typemod):
    predict_result = open(args.predict_result, "w")
    label_dict = {0: "unmod", 1: "mod"}

    task_idx = TASK_MAP[typemod]

    for i, (batch_x, batch_y) in enumerate(dataloader):

        signal, kmer, mean, std, intense, dwell, base_quality = batch_x
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
        probabilities = out[task_idx].cpu().detach().numpy()  # predicting score
        pred = np.array([0 if instance < 0.5 else 1 for instance in probabilities])  # for assessment

        for j in range(len(batch_y)):
            contig, position, motif, read_id = batch_y[j].split("|")
            # print(probabilities[j])
            print("%s\t%s\t%s\t%s\t%s\t%s" % (contig, position, motif, read_id, label_dict[pred[j]], probabilities[j]),
                  file=predict_result)

    predict_result.close()


def predict_several_type(model, dataloader):
    RMs = ["m6A", "m1A", "m5C", "hm5C", "I", "m7G", "psi"]
    num_task = len(RMs)

    all_probs = []  # 用于存储每个 batch 的概率矩阵 (batch_size, num_task)
    all_meta = []  # 可选：如果需要标识信息，也可以一并收集

    for step, (batch_x, batch_y) in enumerate(dataloader):
        signal, kmer, mean, std, intense, dwell, base_quality = batch_x
        signal = Variable(signal.to(device)).to(torch.float32)
        kmer = Variable(kmer.to(device)).to(torch.long)
        mean = Variable(mean.to(device)).to(torch.float32)
        std = Variable(std.to(device)).to(torch.float32)
        intense = Variable(intense.to(device)).to(torch.float32)
        dwell = Variable(dwell.to(device)).to(torch.float32)
        base_quality = Variable(base_quality.to(device)).to(torch.float32)
        batch_size, features = signal.size()
        signal = signal.view(batch_size, 1, features)

        out = model(signal, kmer, mean, std, intense, dwell, base_quality)  # list of 7 tensors

        # 收集当前 batch 的所有任务概率 (batch_size, num_task)
        probs_batch = np.column_stack([out[i].cpu().detach().numpy() for i in range(num_task)])
        all_probs.append(probs_batch)

        # 可选：收集标识信息（如果需要）
        meta_batch = [batch_y[j].split("|") for j in range(len(batch_y))]
        all_meta.extend(meta_batch)

        print(f"batch {step} finished.")

    # 合并所有 batch
    if all_probs:
        all_probs = np.vstack(all_probs)  # (total_samples, num_task)
    else:
        all_probs = np.array([])

    # 构建 
    meta_df = pd.DataFrame(all_meta, columns=["contig", "position", "motif", "read_id"])
    prob_df = pd.DataFrame(all_probs, columns=[f"{rm}_prob" for rm in RMs])
    result_df = pd.concat([meta_df, prob_df], axis=1)
    
    output_csv = args.predict_result 
    result_df.to_csv(output_csv, index=False, sep='\t')
    print(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TandemMod, multiple types of RNA modification detection.')
    parser.add_argument('--type', type=str, required=False, default='all', help='Pretrained model file.')
    parser.add_argument('--pretrained_model', required=True, help='Pretrained model file.')
    parser.add_argument('--feature_file', required=True, default='', help='File to be predicted.')
    parser.add_argument('--predict_result', required=True, default='', help='Predict results.')
    parser.add_argument('--bs', type=int, required=False, default=256, help='batch_size')

    args = parser.parse_args()
    file_name = args.feature_file
    print("load data")
    X, Y = load_predict_data(file_name)
    print(file_name)

    print("predict_process")
    dataset = MyDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.bs)
    model = torch.load(args.pretrained_model)

    if args.type != 'all':
        predict_certain_type(model, dataloader, args.type)
    else:
        predict_several_type(model, dataloader)


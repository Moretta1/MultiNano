import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
from torch.utils import data as torch_data
from torch.utils.data import Dataset
from torch.autograd import Variable
from utils import *

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device=", device)

# predict dataset pth
data_root_pth = '/data/home/grp-lizy/wangrulan/tandem/data/m6A_HEK293T'

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


def predict(model, dataloader):
    predict_result = open(args.predict_result, "w")
    label_dict = {0: "unmod", 1: "mod"}

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
        probabilities = out[0].cpu().detach().numpy()  # predicting score
        pred = np.array([0 if instance < 0.5 else 1 for instance in probabilities])  # for assessment

        for j in range(len(batch_y)):
            contig, position, motif, read_id = batch_y[j].split("|")
            print("%s\t%s\t%s\t%s\t%s\t%s" % (contig, position, motif, read_id, label_dict[pred[j]], probabilities[j]),
                  file=predict_result)

    predict_result.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TandemMod, multiple types of RNA modification detection.')
    parser.add_argument('--type', required=False, default='m6A', help='Pretrained model file.')
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
    predict(model, dataloader)




import torch.utils.data as data
import torch
import os
import sys
import numpy as np
from torch.utils.data import WeightedRandomSampler, Dataset
from collections import Counter

class RMdata(Dataset):
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
        x = self.x[index]
        y = self.y[index, ...]

        return x, y

    def __len__(self):
        return len(self.x)


def calculate_confusion_matrix(label_real, label_pred):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    print("true label distribution within modification:")
    print(Counter(label_real))
    print('tp\tfp\ttn\tfn')
    print('{}\t{}\t{}\t{}'.format(tp, fp, tn, fn))

    return tp, fp, tn, fn


def calculate_metric_within_modification(pred_prob, label_pred, label_real):
    from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score

    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    print("true label distribution within modification:")
    print(Counter(label_real))
    print('tp\tfp\ttn\tfn')
    print('{}\t{}\t{}\t{}'.format(tp, fp, tn, fn))

    # Accuracy
    ACC = float(tp + tn) / test_num

    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # ROC and AUC
    try:
        FPR, TPR, thresholds = roc_curve(label_real, pred_prob,
                                         pos_label=1)  # Default 1 is positive sample,FPR/TPR for plot
        AUC = auc(FPR, TPR)
    except ValueError:
        pass

    try:
        # PRC and AP
        precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
        AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)
    except ValueError:
        pass

    performance = [ACC, Sensitivity, Specificity, AUC, MCC]
    roc_data = [FPR, TPR, AUC]      # ROC(FPR, TPR, AUC)
    prc_data = [recall, precision, AP]    # PRC(Recall, Precision, AP)
    print("within single modification ACC, Sensitivity, Specificity, AUC, MCC, AP:")
    print(performance)
    print(prc_data[2])
    print("<======================>")
    return performance, roc_data, prc_data


def make_weights_for_balanced_classes(dataset):
    X, y = dataset[:]
    num_examples = len(y)
    nclasses = len(y) + 1
    count = np.zeros(nclasses)
    y = np.asarray(y)
    for i in range(num_examples):
        count[np.concatenate([np.squeeze(y[i, :]), np.array([0])]) == 1] += 1
    # negative class weight
    count[-1] = num_examples - np.sum([count[i] for i in range(nclasses)])
    weight_per_class = np.zeros(nclasses)
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * num_examples
    for i in range(num_examples):
        if not list(np.squeeze(y[i, :])) == list(np.zeros(len(y[1]))):
            weight[i] = np.mean(weight_per_class[np.concatenate([np.squeeze(y[i, :]), np.array([0])]) == 1])
        else:
            # negative cases
            weight[i] = weight_per_class[-1]
    return weight


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """
    Custom implementation of WeightedRandomSampler.
    This class extends the WeightedRandomSampler class provided by PyTorch.
    WeightedRandomSampler except allows for more than 2^24 samples to be sampled

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        None
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


def make_dataset(dir, class_to_idx):
    dataset = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                dataset.append(item)

    return dataset


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader):
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.root = root
        self.loader = loader

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str


kmer_encode_dic = {'A': 0, "C": 1, "G": 2, "T": 3}


def default_loader(path):
    """
    Load data from the specified path.

    Args:
        path (str): The path to the file.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing the loaded data:
                - signal (np.ndarray): The signals.
                - kmer (np.ndarray): The encoded kmers.
                - mean (np.ndarray): The mean values.
                - std (np.ndarray): The standard deviation values.
                - intense (np.ndarray): The intensity values.
                - dwell (np.ndarray): The dwell values.
                - base_quality (np.ndarray): The base quality values.

    Raises:
        None
    """
    with open(path) as f:
        line = f.readlines()[0]
        signals = "|".join(line.split("\t")[9:14]).split("|")
        signal = np.array([float(signal) for signal in signals])
        # signal=(signal-np.mean(signal))/np.std(signal)
        kmer = line.split("\t")[3]
        kmer = np.array([kmer_encode_dic[base] for base in kmer])
        mean = np.array([float(item) for item in line.split("\t")[4].split("|")])
        std = np.array([float(item) for item in line.split("\t")[5].split("|")])
        intense = np.array([float(item) for item in line.split("\t")[6].split("|")])
        dwell = np.array([float(item) for item in line.split("\t")[7].split("|")]) / 200
        base_quality = np.array([float(item) for item in line.split("\t")[8].split("|")]) / 40
    return signal, kmer, mean, std, intense, dwell, base_quality


def sequence_and_signal_loader(path):
    """
    Load sequence and signal data from the specified path.

    Args:
        path (str): The path to the file.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing the loaded data:
                - signal (np.ndarray): The signal data.
                - kmer (np.ndarray): The kmer data.
                - mean (np.ndarray): The mean values.
                - std (np.ndarray): The standard deviation values.
                - intense (np.ndarray): The intensity values.
                - dwell (np.ndarray): The dwell values.

    Raises:
        None
    """
    with open(path) as f:
        line = f.readline()
        line = line.rstrip()
        signal = np.array([float(item) for item in line.split(",")[0:500]])

    return signal


class TextFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        texts (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, loader=default_loader):
        super(TextFolder, self).__init__(root, loader)

        self.texts = self.samples


def load_data(data_mod, data_unmod):
    """
    Load data from the specified files.

    Args:
        data_mod (str): The path to the modified data file.
        data_unmod (str): The path to the unmodified data file.
        data_length (int): The desired length of the data.

    Returns:
        Tuple[List[List[np.ndarray]], List[int]]:
            A tuple containing the loaded data:
                - X (List[List[np.ndarray]]): A list of input samples, where each sample is a list of numpy arrays.
                - Y (List[int]): A list of class labels.

    Raises:
        None
    """
    X, Y = [], []
    for i, file in enumerate([data_unmod, data_mod]):
        # y = i = 0, y = i = 1
        with open(file) as f:
            count = 0
            for line in f:
                line = line.rstrip()
                items = line.split("\t")
                contig = items[1]

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
                Y.append(i)

                count += 1
                if count > 3e5:
                   break
    return X, Y


def load_predict_data(file):
    """
    Load prediction data from the specified file.

    Args:
        file (str): The path to the file.

    Returns:
        Tuple[List[List[np.ndarray]], List[str]]:
            A tuple containing the loaded data:
                - X (List[List[np.ndarray]]): A list of input samples, where each sample is a list of numpy arrays.
                - Y (List[str]): A list of identifiers for each sample.

    Raises:
        None
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
            # signal=(signal-np.mean(signal))/np.std(signal)
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


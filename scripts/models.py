import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as torch_data
from torch.utils.data import Dataset
from torch.autograd import Variable
from transformers import BertTokenizer, BertConfig, BertModel


class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention model, modified from MultiRM (https://github.com/Tsedao/MultiRM)

    Args:
        hidden_states (tensor): The hidden state from LSTM.
        values (tensor): The output from LSTM.

    Returns:
        tensor: context_vector, attention_weights.
    """

    def __init__(self, in_features, hidden_units, num_task):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.W2 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

    def forward(self, hidden_states, values):
        hidden_with_time_axis = torch.unsqueeze(hidden_states, dim=1)

        score = self.V(nn.Tanh()(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = nn.Softmax(dim=1)(score)
        values = torch.transpose(values, 1, 2)
        # transpose to make it suitable for matrix multiplication

        context_vector = torch.matmul(values, attention_weights)
        context_vector = torch.transpose(context_vector, 1, 2)
        return context_vector, attention_weights


class NaiveNet(nn.Module):
    """
    NaiveNet model。

    Args:
        Current level features (tensor): x.
        Event level features (tensor): kmer,mean,std,intense,dwell,base_quality.

    Returns:
        tensor: x, 2D probabilities.
    """

    def __init__(self, num_task, num_classes=2, vocab_size=5, embedding_size=4, seq_len=5):
        super(NaiveNet, self).__init__()

        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_zie, embedding_size)  # 将每个输入的整数索引映射到一个固定大小的嵌入向量
        self.num_task = num_task
        self.lstm_seq = nn.LSTM(input_size=4 + 5, hidden_size=128, batch_first=True, bidirectional=True)

        self.cnn_1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1),
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=True)
        self.attention = BahdanauAttention(in_features=256, hidden_units=10, num_task=num_task)
        for i in range(num_task):
            setattr(self, "NaiveFC%d" % i, nn.Sequential(
                nn.Linear(in_features=1536, out_features=1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=1024, out_features=512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=512, out_features=128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=128, out_features=1),
                nn.Sigmoid()
            ))

    def seq_to_digit(self, seq):
        return torch.Tensor([{'A': 0, "C": 1, "G": 2, "T": 3}[i] for i in list(seq)]).long()

    def forward(self, x, kmer, mean, std, intense, dwell, base_quality):
        kmer_embedded = self.embed(kmer)
        mean = torch.reshape(mean, (-1, self.seq_len, 1)).float()
        std = torch.reshape(std, (-1, self.seq_len, 1)).float()
        intense = torch.reshape(intense, (-1, self.seq_len, 1)).float()
        dwell = torch.reshape(dwell, (-1, self.seq_len, 1)).float()
        base_quality = torch.reshape(base_quality, (-1, self.seq_len, 1)).float()

        out_seq = torch.cat((kmer_embedded, mean, std, intense, dwell, base_quality), 2)
        out_seq, (h_n_seq, c_n_seq) = self.lstm_seq(out_seq)

        x = self.cnn_1d(x)

        batch_size, features, seq_len = x.size()
        x = x.view(batch_size, seq_len, features)  # parepare input for LSTM

        output, (h_n, c_n) = self.lstm(x)

        h_n = h_n.view(batch_size, output.size()[-1])  # pareprae input for Attention
        context_vector, attention_weights = self.attention(h_n, output)  # Attention (batch_size, num_task, unit)
        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, "NaiveFC%d" % i)
            out = torch.cat((out_seq[:, 0, :], out_seq[:, 1, :], out_seq[:, 2, :], out_seq[:, 3, :], out_seq[:, 4, :],
                             context_vector[:, i, :]), 1)
            out.view(out.size()[0], 1, out.size()[1])
            y = FClayer(out)
            y = torch.squeeze(y, dim=-1)

            outs.append(y)

        return outs


class SignalTransformer(nn.Module):
    """
    CNN + Transformer
    """

    def __init__(self, num_task):
        super(SignalTransformer, self).__init__()
        self.num_task = num_task

        # 1D-CNN feature extraction
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3)  # 500 -> 250
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # 250 -> 250
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # 250 -> 250

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # dimension reduction 250 -> 125

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc_input_size = 64 * 62
        self.shared_fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # multi-label output
        for i in range(self.num_task):
            setattr(self, f"NaiveFC{i}", nn.Sequential(
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ))

    def forward(self, x):
        """
        x: [batch_size, 500] -> CNN -> Transformer -> multi-label output
        """
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        # Transformer
        x = x.permute(2, 0, 1)  # [seq_len=125, batch_size, feature_dim=64]
        x = self.transformer(x)  # Transformer encoder

        x = x.permute(1, 0, 2).reshape(x.shape[1], -1)  # [batch_size, 64 * 125]
        shared_feature = self.shared_fc(x)  # [batch_size, 1024]

        # multi-label output
        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, f"NaiveFC{i}")
            y = FClayer(shared_feature)
            y = torch.squeeze(y, dim=-1)  # [batch_size]
            outs.append(y)

        return outs  # [batch_size, num_task]


class SignalTransformer_v2(nn.Module):
    # CNN + BiLSTM + Transformer + Multi-label Output
    def __init__(self, num_task, vocab_size=5, embedding_size=4, seq_len=5):
        super(SignalTransformer_v2, self).__init__()
        self.num_task = num_task
        self.seq_len = seq_len

        # for Base-level features (event-level features)
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.base_lstm = nn.LSTM(input_size=embedding_size + 5, hidden_size=128, batch_first=True, bidirectional=True)

        # for raw Signal (current-level features)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.signal_lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # shared FC layer
        self.shared_fc = nn.Sequential(
            nn.Linear(256 + 256, 1024),  # CNN+BiLSTM+Transformer concat Base-level BiLSTM
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # multi-label output
        for i in range(self.num_task):
            setattr(self, f"NaiveFC{i}", nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ))


    def forward(self, x, kmer, mean, std, intense, dwell, base_quality):
        """
        x: [batch_size, 500]  # raw signal (current-level features)
        kmer: [batch_size, seq_len]  # base-level feature: kmer
        mean, std, intense, dwell, base_quality: [batch_size, seq_len]  # other base-level features
        """

        # Base-level Feature Processing: BiLSTM
        kmer_embedded = self.embed(kmer)  # [batch_size, seq_len, embedding_size]
        mean = mean.unsqueeze(-1).float()
        std = std.unsqueeze(-1).float()
        intense = intense.unsqueeze(-1).float()
        dwell = dwell.unsqueeze(-1).float()
        base_quality = base_quality.unsqueeze(-1).float()

        base_features = torch.cat((kmer_embedded, mean, std, intense, dwell, base_quality), dim=2)
        base_out, _ = self.base_lstm(base_features)  # [batch_size, seq_len, 256]
        base_out = torch.mean(base_out, dim=1)  # [batch_size, 256]

        # Current-level Feature Processing:
        # CNN
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [batch_size, 256, seq_len=62]

        # BiLSTM
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, feature_dim]
        x, _ = self.signal_lstm(x)  # [batch_size, seq_len, 256]

        # Transformer
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, 256]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, 256]

        #
        x = torch.mean(x, dim=1)  # [batch_size, 256]

        # ====== Feature Concatenation & Multi-task Learning ======
        combined_features = torch.cat((x, base_out), dim=1)  # [batch_size, 256 + 256]
        shared_feature = self.shared_fc(combined_features)  # [batch_size, 512]

        # multi-label output
        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, f"NaiveFC{i}")
            y = FClayer(shared_feature)
            y = torch.squeeze(y, dim=-1)  # [batch_size]
            outs.append(y)

        return outs  # [batch_size, num_task]

import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda:0")

def satisfy(x):
    x = torch.tensor(x)
    x = torch.unsqueeze(x, 0)
    return x

def fdata_prepair(x):
    x = torch.tensor(x, dtype=torch.float)
    x = x.unsqueeze(0)
    x = x.unsqueeze(0)
    return x

def ndata_prepair(x):
    x = torch.tensor(x, dtype=torch.float)
    x = x.unsqueeze(0)
    x = x.unsqueeze(2)
    return x

def make_model():
    model = models.vgg16(pretrained=True)
    model = model.eval()
    return model

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = x.to(self.query.weight.dtype)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_dim))
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        output = self.fc(output)
        return output

class ResidualCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ResidualCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.residual_block = self.build_residual_block(64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def build_residual_block(self, channels):
        layers = []
        for _ in range(1):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.residual_block(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = out.unsqueeze(0)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.multihead_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, 20)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x1 = x
        x, _ = self.multihead_attention(x, x, x, attn_mask=None)
        x = x + x1
        x = self.dropout(self.layer_norm(x))
        x2 = x
        x = self.feedforward(x)
        x = x + x2
        x = self.dropout(self.layer_norm(x)).flatten()
        x = self.linear(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * torch.sqrt(torch.tensor(self.d_model))
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len]
        x = x + pe.to(x.device)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.lstm1 = nn.LSTM(1, 20, 2)
        self.lstm2 = nn.LSTM(1, 20, 2)

        self.self_attention = SelfAttention(104, 64, 100)

        self.linear1 = nn.Sequential(nn.Linear(1000, 256),
                                     nn.Linear(256, 64))

        self.decoder = TransformerEncoder(264, 128, 4, 0.1)

        self.linear2 = nn.Linear(100, 64)

    def forward(self, x):
        x = x.flatten().tolist()

        features = x[:1000]
        features = self.linear1(satisfy(features).cuda()).flatten().tolist()

        acc_before = x[1040:1240]

        acc = torch.tensor(x[1240:1260])


        speed, _ = self.lstm1(ndata_prepair(x[1000:1020]).cuda())
        speed = speed[:, -1, :].flatten().tolist()

        force, _ = self.lstm2(ndata_prepair(x[1020:1040]).cuda())
        force = force[:, -1, :].flatten().tolist()

        y = satisfy(features+speed+force)

        feature = self.self_attention(y.cuda())
        feature = feature.flatten().tolist()

        feature = self.linear2(satisfy(feature).cuda()).flatten().tolist()

        feature += acc_before

        acc_pre = self.decoder(fdata_prepair(feature).cuda())

        return acc, acc_pre
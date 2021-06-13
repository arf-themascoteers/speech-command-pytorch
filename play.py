import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader
import os


QUICK = False


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
        elif subset == "debug":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [self._walker[w] for w in range(len(self._walker)) if self._walker[w] not in excludes and w%10 == 0]
        elif subset == "dev":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [self._walker[w] for w in range(len(self._walker)) if self._walker[w] not in excludes and w%1000 == 0]


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.n_labels = 35
        self.net = nn.Sequential(
            nn.Linear(16000, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_labels)
        )

    def forward(self, x):
        x = self.net(x)
        return F.log_softmax(x, dim=1)


def refine(data):
    return [data[i] for i in range(len(data)) if data[i][0].shape[1] == 16000 ]


def train():
    NUM_EPOCHS = 20
    BATCH_SIZE = 100

    working_set = SubsetSC("training")
    if QUICK:
        working_set = SubsetSC("dev")
    working_set = refine(working_set)
    dataloader = DataLoader(working_set, batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleNet()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch  in range(1, NUM_EPOCHS + 1):
        for features, size, train_labels, serial, train_labels_indices in dataloader:
            optimizer.zero_grad()
            features = features.reshape(features.shape[0], features.shape[2])
            y_pred = model(features)
            loss = F.nll_loss(y_pred, train_labels_indices)
            loss.backward()
            optimizer.step()
            print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    return model


def test(model):
    working_set = SubsetSC("validation")
    if QUICK:
        working_set = SubsetSC("dev")
    working_set = refine(working_set)
    dataloader = DataLoader(working_set, batch_size=100, shuffle=True)
    model.eval()
    correct = 0
    with torch.no_grad():
        for features, size, train_labels, serial, train_labels_indices in dataloader:
            features = features.reshape(features.shape[0], features.shape[2])
            y_pred = model(features)
            pred = y_pred.data.max(1, keepdim=True)[1]
            correct += pred.eq(train_labels_indices.data.view_as(pred)).sum()
    print(f"{correct} correct among {len(working_set)}")


model = train()
test(model)

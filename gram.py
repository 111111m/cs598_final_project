import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import argparse
import os
import time
import torch.nn.functional as F

_TEST_RATIO = 0.2
_VALIDATION_RATIO = 0.1
# Custom dataset class
class VisitDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# GRAM model
class GRAM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, attention_dim, num_classes, num_levels):
        super(GRAM, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.attention_layers = nn.ModuleList([nn.Linear(hidden_dim, attention_dim) for _ in range(num_levels)])
        self.attention_context_vectors = nn.ParameterList([nn.Parameter(torch.randn(attention_dim)) for _ in range(num_levels)])
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.num_levels = num_levels

    def forward(self, x, leaves):
        x = self.embedding(x)
        x, _ = self.gru(x)

        attention_weights = []
        for i in range(self.num_levels):
            level_leaves = leaves[i]
            level_weights = F.softmax(torch.matmul(F.tanh(self.attention_layers[i](x)), self.attention_context_vectors[i]), dim=-1)
            level_output = torch.bmm(level_weights.unsqueeze(2), x).squeeze(2)
            level_output = torch.gather(level_output, 1, level_leaves.unsqueeze(2).expand(-1, -1, level_output.size(2)))
            attention_weights.append(level_weights)
            if i == 0:
                outputs = level_output
            else:
                outputs += level_output

        out = self.fc(outputs)
        return out, attention_weights



def get_input_dim_size(seq_file):
    with open(seq_file, 'rb') as f:
        sequences = pickle.load(f)
    return max([max(seq) for seq in sequences]) + 1


def get_num_classes(label_file):
    with open(label_file, 'rb') as f:
        labels = pickle.load(f)
    return max(labels) + 1


def get_rootCode(tree_file):
    with open(tree_file, 'rb') as f:
        tree = pickle.load(f)
    return max([max(subtree.keys()) for subtree in tree.values()]) + 1

def parse_arguments(parser):
    parser.add_argument('--seq_file', type=str, default='seqFile.txt', help='Path to the input sequence file.')
    parser.add_argument('--label_file', type=str, default='labelFile.txt', help='Path to the input label file.')
    parser.add_argument('--tree_file', type=str, default='tree.txt', help='Path to the input tree file.')
    return parser.parse_args()


def load_data(seqFile, labelFile, timeFile=''):
    with open(seqFile, 'rb') as f:
        sequences = np.array(pickle.load(f))
    with open(labelFile, 'rb') as f:
        labels = np.array(pickle.load(f))
    times = None
    if timeFile:
        with open(timeFile, 'rb') as f:
            times = np.array(pickle.load(f))

    np.random.seed(0)
    dataSize = len(labels)
    indices = np.random.permutation(dataSize)
    nTest = int(_TEST_RATIO * dataSize)
    nValid = int(_VALIDATION_RATIO * dataSize)

    test_indices, valid_indices, train_indices = indices[:nTest], indices[nTest:nTest + nValid], indices[nTest + nValid:]
    train_set_x, train_set_y, train_set_t = sequences[train_indices], labels[train_indices], times[train_indices] if times is not None else None
    test_set_x, test_set_y, test_set_t = sequences[test_indices], labels[test_indices], times[test_indices] if times is not None else None
    valid_set_x, valid_set_y, valid_set_t = sequences[valid_indices], labels[valid_indices], times[valid_indices] if times is not None else None

    def sort_by_seq_length(data, labels, times=None):
        sorted_index = sorted(range(len(data)), key=lambda x: len(data[x]))
        data, labels = [data[i] for i in sorted_index], [labels[i] for i in sorted_index]
        if times is not None:
            times = [times[i] for i in sorted_index]
        return data, labels, times

    train_set = sort_by_seq_length(train_set_x, train_set_y, train_set_t)
    valid_set = sort_by_seq_length(valid_set_x, valid_set_y, valid_set_t)
    test_set = sort_by_seq_length(test_set_x, test_set_y, test_set_t)

    return train_set, valid_set, test_set


def train(seq_file, label_file, tree_file, emb_file, out_file, input_dim_size, num_ancestors, emb_dim_size, hidden_dim_size, attention_dim_size, max_epochs, L2, num_class, batch_size, dropout_rate, log_eps, verbose):
    train_data, valid_data, test_data = load_data(seq_file, label_file)
    train_dataset = VisitDataset(train_data[0], train_data[1])
    valid_dataset = VisitDataset(valid_data[0], valid_data[1])
    test_dataset = VisitDataset(test_data[0], test_data[1])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GRAM(input_dim_size, emb_dim_size, hidden_dim_size, attention_dim_size, num_class, num_ancestors).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=L2)

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        valid_loss = 0
        for batch_x, batch_y in valid_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            valid_loss += loss.item()

        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        print('Epoch {}: Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch, train_loss, valid_loss))
        torch.save(model.state_dict(), out_file)

        # Evaluate on test data
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs, _ = model(batch_x)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()

        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / total
    print('Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_loss, accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    input_dim_size = get_input_dim_size(args.seq_file)
    num_class = get_num_classes(args.label_file)
    num_ancestors = get_rootCode(args.tree_file) - input_dim_size + 1

    train(
        seq_file=args.seq_file,
        label_file=args.label_file,
        tree_file=args.tree_file,
        emb_file=args.emb_file,
        out_file=args.out_file,
        input_dim_size=input_dim_size,
        num_ancestors=num_ancestors,
        emb_dim_size=100,
        hidden_dim_size=200,
        attention_dim_size=200,
        max_epochs=100,
        L2=0.,
        num_class=num_class,
        batch_size=100,
        dropout_rate=0.5,
        log_eps=1e-8,
        verbose=False
    )

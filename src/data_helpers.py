import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from derivator import Derivator


class SADataset(Dataset):

    def __init__(self, name, split, tok_type):

        self.tok = BertTokenizer.from_pretrained('bert-base-uncased')
        self.derivator = Derivator()

        data = pd.read_csv('{0}/csv/{0}_{1}.csv'.format(name, split))

        self.labels = list(data.label)
        self.derivatives = list(data.derivative.apply(tokenize, args=(self.tok, self.derivator, tok_type, )))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        label = self.labels[idx]
        derivative = self.derivatives[idx]

        return label, derivative


def tokenize(derivative, tok, derivator, tok_type):

    if tok_type == 'wordpiece':
        return tok.encode(derivative, add_special_tokens=True)

    elif tok_type == 'derivational':
        tokens = ['[CLS]']
        d = derivator.derive(derivative, mode='morphemes')
        if len(d[0]) > 0:
            for pfx in d[0]:
                tokens += [pfx, '-']
        tokens += [d[1]]
        if len(d[2]) > 0:
            for sfx in d[2]:
                tokens += ['##' + sfx]
        tokens += ['[SEP]']
        return tok.convert_tokens_to_ids(tokens)


def collate(batch):

    batch_size = len(batch)

    labels = torch.tensor([l for l, _ in batch]).float()
    derivatives = [d for _, d in batch]

    max_len = max(len(d) for d in derivatives)

    derivatives_pad = torch.zeros((batch_size, max_len)).long()
    masks_pad = torch.zeros((batch_size, max_len)).long()
    segs_pad = torch.zeros((batch_size, max_len)).long()

    for i, d in enumerate(derivatives):
        derivatives_pad[i, :len(d)] = torch.tensor(d)
        masks_pad[i, :len(d)] = 1

    return labels, derivatives_pad, masks_pad, segs_pad


def get_best(file):

    try:
        results = list()
        with open(file, 'r') as f:
            for l in f:
                if l.strip() == '':
                    continue
                results.append(float(l.strip().split()[1]))
        return max(results)

    except FileNotFoundError:
        return None

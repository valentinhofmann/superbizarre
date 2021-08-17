import pandas as pd
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

    elif tok_type == 'stem':
        tokens = ['[CLS]']
        d = derivator.derive(derivative, mode='morphemes')
        if len(d[0]) > 0:
            for _ in d[0]:
                tokens += ['[unused0]', '-']
        tokens += [d[1]]
        if len(d[2]) > 0:
            for _ in d[2]:
                tokens += ['[unused1]']
        tokens += ['[SEP]']
        return tok.convert_tokens_to_ids(tokens)

    elif tok_type == 'affixes':
        tokens = ['[CLS]']
        d = derivator.derive(derivative, mode='morphemes')
        if len(d[0]) > 0:
            for pfx in d[0]:
                tokens += [pfx, '-']
        tokens += ['[unused0]']
        if len(d[2]) > 0:
            for sfx in d[2]:
                tokens += ['##' + sfx]
        tokens += ['[SEP]']
        return tok.convert_tokens_to_ids(tokens)

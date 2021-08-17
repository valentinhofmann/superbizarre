import argparse
import logging
import os
import pickle
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import optim, nn
from torch.utils.data import DataLoader

from data_helpers import collate, get_best
from model import BertClassifier


def main():

    logging.disable(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')
    parser.add_argument('--lr', default=None, type=float, required=True, help='Learning rate.')
    parser.add_argument('--random_seed', default=None, type=int, required=True, help='Random seed.')
    parser.add_argument('--n_epochs', default=None, type=int, required=True, help='Number of epochs.')
    parser.add_argument('--data', default=None, type=str, required=True, help='Name of data.')
    parser.add_argument('--tok_type', default=None, type=str, required=True, help='Tokenization type.')
    parser.add_argument('--device', default=None, type=int, required=True, help='Selected CUDA device.')
    parser.add_argument('--hs', default=False, action='store_true', help='Hyperparameter search.')
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('trained'):
        os.makedirs('trained')

    print('Load training data...')
    with open('../data/{0}/pickle/{0}_{1}_train.p'.format(args.data, args.tok_type), 'rb') as f:
        train_dataset = pickle.load(f)
    print('Load development data...')
    with open('../data/{0}/pickle/{0}_{1}_dev.p'.format(args.data, args.tok_type), 'rb') as f:
        dev_dataset = pickle.load(f)
    print('Load test data...')
    with open('../data/{0}/pickle/{0}_{1}_test.p'.format(args.data, args.tok_type), 'rb') as f:
        test_dataset = pickle.load(f)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate)

    if args.hs:
        filename = '{}_{}_hs'.format(args.data, args.tok_type)
    else:
        filename = '{}_{}_{:02d}'.format(args.data, args.tok_type, args.random_seed)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    model = BertClassifier().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    best_f1 = get_best('results/{}.txt'.format(filename))
    print('Best F1 so far: {}'.format(best_f1))

    print('Train classifier...')
    for epoch in range(1, args.n_epochs + 1):

        model.train()

        for i, batch in enumerate(train_loader):

            if i % 1000 == 0:
                print('Processed {} examples...'.format(i * args.batch_size))

            labels, derivatives, masks, segs = batch

            labels = labels.to(device)
            derivatives = derivatives.to(device)
            masks = masks.to(device)
            segs = segs.to(device)

            optimizer.zero_grad()

            output = model(derivatives, masks, segs)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        print('Evaluate classifier...')
        model.eval()

        y_true = list()
        y_pred = list()

        with torch.no_grad():

            for batch in dev_loader:

                labels, derivatives, masks, segs = batch

                labels = labels.to(device)
                derivatives = derivatives.to(device)
                masks = masks.to(device)
                segs = segs.to(device)

                output = model(derivatives, masks, segs)

                y_true.extend(labels.tolist())
                y_pred.extend(torch.round(output).tolist())

        f1_dev = f1_score(y_true, y_pred, average='macro')

        y_true = list()
        y_pred = list()

        with torch.no_grad():

            for batch in test_loader:

                labels, derivatives, masks, segs = batch

                labels = labels.to(device)
                derivatives = derivatives.to(device)
                masks = masks.to(device)
                segs = segs.to(device)

                output = model(derivatives, masks, segs)

                y_true.extend(labels.tolist())
                y_pred.extend(torch.round(output).tolist())

        f1_test = f1_score(y_true, y_pred, average='macro')

        print(f1_dev, f1_test)

        with open('results/{}.txt'.format(filename), 'a+') as f:
            f.write('{:.0e}\t{}\t{}\n'.format(args.lr, f1_dev, f1_test))

        if best_f1 is None or f1_dev > best_f1:

            best_f1 = f1_dev
            torch.save(model.state_dict(), 'trained/{}.torch'.format(filename))


if __name__ == '__main__':

    start_time = time.time()

    main()

    print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))
    print()

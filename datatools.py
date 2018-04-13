import os
import torch
from torch.autograd import Variable
import torch.utils.data
import numpy as np
from wsj_loader import WSJ
os.environ['WSJ_PATH'] = '../data'

# load raw data


def init_vocab():
    vocab = dict()
    vocab["<eos>"] = 0
    vocab["<unk>"] = 1
    return vocab


def to_string_char(Y, vocab):
    reverse_index = {v: k for k, v in vocab.items()}
    l = []
    for q in Y:
        words = []
        for e in q:
            words.append(reverse_index[e])
        l.append("".join(words[1:-1]))
        # print(q)
    return np.array(l)


def one_hot_minibatch(v, dim):
    u = np.zeros(shape=(len(v), dim))
    for i in range(len(v)):
        u[i, int(v[i])] = 1
    return to_tensor(u)


def to_one_hot(Y, nLabels):
    l = list()
    for y in Y:
        b = np.zeros((len(y), nLabels))
        b[np.arange(len(y)), y] = 1
        l.append(b)
    return l


def load_train_data_char(vocab):
    loader = WSJ()
    c = len(vocab)
    trainX, trainY = loader.train
    l = []
    # print(trainY[1000])
    for s in trainY:
        words = list(s)
        q = []
        q.append(vocab["<eos>"])
        for w in words:
            if w in vocab.keys():
                q.append(vocab[w])
            else:
                vocab[w] = c
                q.append(c)
                c += 1
        q.append(vocab["<eos>"])
        l.append(np.array(q))
    trainY = np.array(l)
    return trainX, trainY


def load_dev_data_char(vocab):
    loader = WSJ()
    devX, devY = loader.dev
    l = []
    for s in devY:
        words = list(s)
        q = []
        q.append(vocab["<eos>"])
        for w in words:
            if w in vocab.keys():
                q.append(vocab[w])
            else:
                q.append(vocab["<unk>"])
        q.append(vocab["<eos>"])
        l.append(np.array(q))
    devY = np.array(l)
    return devX, devY


def load_test_data():
    loader = WSJ()
    testX, _ = loader.test
    return testX

# helpers


def to_tensor(narray):
    return torch.from_numpy(narray)


def to_variable(tensor, cuda=True):
    if cuda and torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)
